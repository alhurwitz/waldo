# Clustering Speed Optimization

**Date:** 2026-04-11
**Scope:** Full auto-cluster pipeline performance — clustering, matching, and embedding extraction parallelism.

## Problem

The auto-cluster pipeline processes faces sequentially at every level: embeddings are compared against centroids one-at-a-time in a Python loop, matching uses per-face iteration over references, and photo detection runs single-threaded. This leaves significant performance on the table, especially as the number of faces, clusters, and media files grows.

## Design

### 1. Vectorized Clustering (`cluster.py`)

Replace the list-of-arrays centroid store with a single NumPy matrix and use BLAS-optimized matrix multiplication for similarity computation.

**Changes to `FaceCluster`:**

- `_centroids: list[np.ndarray]` becomes `_centroid_matrix: np.ndarray` of shape `(k, 512)`. Grows dynamically (double-on-resize or append+rebuild).
- `_counts: list[int]` becomes `_counts: np.ndarray` of shape `(k,)`.
- `assign(embedding)`: computes `sims = self._centroid_matrix[:k] @ embedding` — one BLAS call, then argmax + threshold check. Centroid update is a row operation on the matrix.
- `assign_batch(embeddings: np.ndarray)`: takes `(n, 512)` input, iterates over n calling the vectorized `assign()` internally. Order-dependent centroid updates prevent full parallelism, but each step is BLAS-fast.
- `centroids_as_refs()` slices the matrix directly instead of zipping lists.

**Growth strategy:** Pre-allocate `_centroid_matrix` with capacity 64. When full, double the capacity and copy. `n_clusters` tracks the used portion.

### 2. Vectorized Matching (`matcher.py`)

Introduce a `RefIndex` class that pre-stacks reference embeddings for batch matching.

**New class `RefIndex`:**

```
RefIndex(refs: dict[str, list[np.ndarray]])
```

- On construction: stack all embeddings into a `(total_refs, 512)` matrix. Build a parallel list mapping each row index to its person name.
- `classify(embedding, threshold) -> str | None`: one matmul `matrix @ embedding`, argmax, threshold check, return person name or None.
- `classify_batch(embeddings: np.ndarray, threshold) -> list[str | None]`: `matrix @ embeddings.T` gives `(total_refs, n)`, columnwise argmax + threshold.

**Integration:**

- `_process_video()`: build `RefIndex` once from `refs`, use `classify_batch()` per sample (batch all faces in a single frame).
- `_process_photo()`: same pattern.
- The existing `classify_face()` function is deleted. All call sites switch to `RefIndex` methods.

### 3. Parallel Embedding Extraction in Auto-Cluster (`pipeline.py`)

Separate detection (I/O + GPU-bound) from clustering (CPU-bound) and parallelize detection.

**Video auto-cluster:**

- Run `_ensure_cache()` calls across multiple videos using `ThreadPoolExecutor(max_workers=cfg.workers)`. InsightFace's ONNX runtime releases the GIL, so threading provides real concurrency for inference.
- After all caches are populated, collect all embeddings and call `cluster.assign_batch()`.

**Photo auto-cluster:**

- Use `ThreadPoolExecutor(max_workers=cfg.workers)` to read images and run `embedder.detect()` in parallel.
- Collect all detected embeddings into a single array, then `cluster.assign_batch()`.

**Backward compatibility:** `cfg.workers` defaults to 1, preserving current sequential behavior. Users opt in with `--workers N`.

### 4. Optional FAISS Backend (`cluster.py`)

FAISS as a drop-in acceleration layer, imported opportunistically.

**Dependency:** Optional extra in `pyproject.toml` under `[project.optional-dependencies]`:
```toml
fast = ["faiss-cpu>=1.7"]
```

**Runtime selection:**

- `FaceCluster.__init__()` attempts `import faiss`. If available, uses `faiss.IndexFlatIP` as the centroid store. If not, uses the NumPy matrix from Section 1.
- On `assign()`: FAISS path does `index.search(embedding.reshape(1, -1), 1)` for nearest centroid lookup, then threshold check. On centroid update, the index is rebuilt from the matrix (cheap for <10K centroids).
- `RefIndex` follows the same pattern: FAISS backend if available, NumPy fallback otherwise.

**Same public API** — callers don't know or care which backend is active.

## Files Modified

| File | Changes |
|------|---------|
| `src/waldo/cluster.py` | Matrix-based centroids, `assign_batch()`, optional FAISS backend |
| `src/waldo/matcher.py` | `RefIndex` class replaces `classify_face()` |
| `src/waldo/pipeline.py` | Parallel detection in `_auto_cluster()`, batch assignment, `RefIndex` usage |
| `pyproject.toml` | `faiss-cpu` optional dependency |
| `tests/test_cluster.py` | Update for new API, add batch assignment tests |
| `tests/test_matcher.py` | Update for `RefIndex`, add batch classify tests |

## Non-Goals

- GPU-accelerated embedding (InsightFace already uses CoreML on Apple Silicon; GPU FAISS is out of scope)
- Changing the clustering algorithm itself (greedy online works well for face clustering)
- Async I/O for video frame decoding (OpenCV doesn't support it well)

## Testing

- Existing unit tests updated to exercise new APIs
- New tests for `assign_batch()` and `RefIndex.classify_batch()`
- Verify FAISS and NumPy backends produce identical results
- No performance benchmarks in CI (manual validation)

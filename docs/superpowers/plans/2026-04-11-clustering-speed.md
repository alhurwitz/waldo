# Clustering Speed Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Speed up the full auto-cluster pipeline by vectorizing clustering/matching operations, parallelizing embedding extraction, and adding optional FAISS acceleration.

**Architecture:** Replace Python-loop similarity computation with BLAS-optimized matrix multiplies in both `FaceCluster` and a new `RefIndex` class. Parallelize `_auto_cluster()` detection across files using `ThreadPoolExecutor`. Add FAISS as an optional backend that activates automatically when installed.

**Tech Stack:** NumPy (matrix ops), FAISS (optional, `faiss-cpu`), ThreadPoolExecutor (parallelism)

---

## File Structure

| File | Role |
|------|------|
| `src/waldo/cluster.py` | Matrix-based `FaceCluster` with `assign()` and `assign_batch()`, optional FAISS backend |
| `src/waldo/matcher.py` | `RefIndex` class with `classify()` and `classify_batch()`, optional FAISS backend |
| `src/waldo/pipeline.py` | Parallel detection in `_auto_cluster()`, batch APIs at call sites |
| `pyproject.toml` | `fast` optional dependency group for `faiss-cpu` |
| `tests/test_cluster.py` | Updated + new tests for matrix-based cluster and batch assignment |
| `tests/test_matcher.py` | Updated + new tests for `RefIndex` single and batch classify |

---

### Task 1: Vectorized FaceCluster — Tests

**Files:**
- Modify: `tests/test_cluster.py`

- [ ] **Step 1: Write failing tests for matrix-based FaceCluster and assign_batch**

Add these tests to `tests/test_cluster.py`, keeping all existing tests unchanged:

```python
def test_assign_batch_matches_sequential():
    """assign_batch must produce identical results to calling assign() in sequence."""
    embeddings = np.array([
        _norm([1, 0, 0]),
        _norm([0.95, 0.05, 0]),
        _norm([0, 1, 0]),
        _norm([0, 0.9, 0.1]),
        _norm([0, 0, 1]),
    ])
    # Sequential
    seq = FaceCluster(threshold=0.5)
    seq_ids = [seq.assign(e) for e in embeddings]
    # Batch
    batch = FaceCluster(threshold=0.5)
    batch_ids = batch.assign_batch(embeddings)
    assert batch_ids == seq_ids
    assert batch.n_clusters == seq.n_clusters


def test_assign_batch_empty():
    c = FaceCluster(threshold=0.5)
    assert c.assign_batch(np.zeros((0, 3), dtype=np.float32)) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cluster.py -v`
Expected: FAIL — `FaceCluster` has no `assign_batch` method.

---

### Task 2: Vectorized FaceCluster — Implementation

**Files:**
- Modify: `src/waldo/cluster.py`

- [ ] **Step 1: Rewrite FaceCluster with matrix-based centroid storage**

Replace the entire contents of `src/waldo/cluster.py` with:

```python
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

_INITIAL_CAPACITY = 64


class FaceCluster:
    """Greedy online clustering of L2-normalised face embeddings.

    Centroids are stored in a pre-allocated matrix for BLAS-optimised
    similarity computation.  Falls back gracefully when the matrix needs
    to grow.
    """

    def __init__(self, threshold: float = 0.55, *, dim: int = 512) -> None:
        self.threshold = threshold
        self._dim = dim
        self._capacity = _INITIAL_CAPACITY
        self._centroid_matrix = np.empty((self._capacity, dim), dtype=np.float32)
        self._counts = np.empty(self._capacity, dtype=np.int64)
        self._size = 0

    @property
    def n_clusters(self) -> int:
        return self._size

    def _grow(self) -> None:
        new_cap = self._capacity * 2
        new_mat = np.empty((new_cap, self._dim), dtype=np.float32)
        new_mat[: self._size] = self._centroid_matrix[: self._size]
        self._centroid_matrix = new_mat
        new_counts = np.empty(new_cap, dtype=np.int64)
        new_counts[: self._size] = self._counts[: self._size]
        self._counts = new_counts
        self._capacity = new_cap

    def assign(self, embedding: np.ndarray) -> int:
        """Return the cluster index for this embedding (creating one if needed)."""
        if self._size == 0:
            self._centroid_matrix[0] = embedding
            self._counts[0] = 1
            self._size = 1
            return 0

        sims = self._centroid_matrix[: self._size] @ embedding
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim >= self.threshold:
            n = int(self._counts[best_idx])
            new_c = (self._centroid_matrix[best_idx] * n + embedding) / (n + 1)
            norm = np.linalg.norm(new_c)
            if norm > 0:
                new_c /= norm
            self._centroid_matrix[best_idx] = new_c
            self._counts[best_idx] = n + 1
            return best_idx

        if self._size >= self._capacity:
            self._grow()
        self._centroid_matrix[self._size] = embedding
        self._counts[self._size] = 1
        self._size += 1
        return self._size - 1

    def assign_batch(self, embeddings: np.ndarray) -> list[int]:
        """Assign a batch of embeddings. Order-dependent — processes sequentially
        but each similarity computation is BLAS-optimised."""
        if len(embeddings) == 0:
            return []
        return [self.assign(e) for e in embeddings]

    def centroids_as_refs(self, min_count: int = 2) -> dict[str, list[np.ndarray]]:
        """Export clusters as a refs dict (same shape as load_references output).

        Clusters with fewer than ``min_count`` faces are dropped (likely noise).
        """
        refs: dict[str, list[np.ndarray]] = {}
        for i in range(self._size):
            n = int(self._counts[i])
            if n < min_count:
                log.debug("cluster person_%d has only %d face(s), skipping", i + 1, n)
                continue
            refs[f"person_{i + 1}"] = [self._centroid_matrix[i].copy()]
        return refs
```

- [ ] **Step 2: Run all cluster tests**

Run: `uv run pytest tests/test_cluster.py -v`
Expected: All 6 tests PASS (4 existing + 2 new).

- [ ] **Step 3: Commit**

```bash
git add src/waldo/cluster.py tests/test_cluster.py
git commit -m "perf: vectorize FaceCluster with matrix-based centroid storage"
```

---

### Task 3: RefIndex — Tests

**Files:**
- Modify: `tests/test_matcher.py`

- [ ] **Step 1: Write failing tests for RefIndex**

Replace the entire contents of `tests/test_matcher.py` with:

```python
import numpy as np

from waldo.matcher import RefIndex


def _norm(v):
    a = np.array(v, dtype=np.float32)
    return a / np.linalg.norm(a)


def test_classify_picks_best():
    grandma = _norm([1, 0, 0])
    grandpa = _norm([0, 1, 0])
    refs = {"grandma": [grandma], "grandpa": [grandpa]}
    idx = RefIndex(refs)
    assert idx.classify(_norm([0.9, 0.1, 0]), threshold=0.5) == "grandma"
    assert idx.classify(_norm([0.1, 0.9, 0]), threshold=0.5) == "grandpa"


def test_classify_below_threshold():
    refs = {"grandma": [_norm([1, 0, 0])]}
    idx = RefIndex(refs)
    assert idx.classify(_norm([0, 1, 0]), threshold=0.5) is None


def test_classify_batch():
    grandma = _norm([1, 0, 0])
    grandpa = _norm([0, 1, 0])
    refs = {"grandma": [grandma], "grandpa": [grandpa]}
    idx = RefIndex(refs)
    embeddings = np.array([
        _norm([0.9, 0.1, 0]),
        _norm([0.1, 0.9, 0]),
        _norm([0, 0, 1]),  # no match
    ])
    results = idx.classify_batch(embeddings, threshold=0.5)
    assert results == ["grandma", "grandpa", None]


def test_classify_batch_empty():
    refs = {"a": [_norm([1, 0, 0])]}
    idx = RefIndex(refs)
    assert idx.classify_batch(np.zeros((0, 3), dtype=np.float32), threshold=0.5) == []


def test_multiple_refs_per_person():
    refs = {
        "grandma": [_norm([1, 0, 0]), _norm([0.9, 0.1, 0])],
        "grandpa": [_norm([0, 1, 0])],
    }
    idx = RefIndex(refs)
    assert idx.classify(_norm([0.95, 0.05, 0]), threshold=0.5) == "grandma"
    assert idx.classify(_norm([0.1, 0.9, 0]), threshold=0.5) == "grandpa"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_matcher.py -v`
Expected: FAIL — `RefIndex` does not exist.

---

### Task 4: RefIndex — Implementation

**Files:**
- Modify: `src/waldo/matcher.py`

- [ ] **Step 1: Replace matcher.py with RefIndex implementation**

Replace the entire contents of `src/waldo/matcher.py` with:

```python
from __future__ import annotations

import numpy as np


class RefIndex:
    """Pre-stacked reference embeddings for fast batch matching.

    All reference embeddings are stacked into a single matrix so that
    classification becomes a single BLAS matmul + argmax.
    """

    def __init__(self, refs: dict[str, list[np.ndarray]]) -> None:
        names: list[str] = []
        rows: list[np.ndarray] = []
        for name, embs in refs.items():
            for e in embs:
                names.append(name)
                rows.append(e)
        self._names = names
        self._matrix = np.vstack(rows).astype(np.float32) if rows else np.empty((0, 0), dtype=np.float32)

    @property
    def n_refs(self) -> int:
        return len(self._names)

    def classify(self, embedding: np.ndarray, threshold: float) -> str | None:
        """Return the person whose reference is closest, or None if below threshold."""
        if self.n_refs == 0:
            return None
        sims = self._matrix @ embedding
        best_idx = int(np.argmax(sims))
        if float(sims[best_idx]) >= threshold:
            return self._names[best_idx]
        return None

    def classify_batch(
        self, embeddings: np.ndarray, threshold: float
    ) -> list[str | None]:
        """Classify multiple embeddings in one matmul."""
        n = len(embeddings)
        if n == 0 or self.n_refs == 0:
            return [None] * n
        # (total_refs, n) similarity matrix
        sim_matrix = self._matrix @ embeddings.T
        best_indices = np.argmax(sim_matrix, axis=0)
        best_sims = sim_matrix[best_indices, np.arange(n)]
        return [
            self._names[int(idx)] if float(sim) >= threshold else None
            for idx, sim in zip(best_indices, best_sims)
        ]
```

- [ ] **Step 2: Run all matcher tests**

Run: `uv run pytest tests/test_matcher.py -v`
Expected: All 5 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add src/waldo/matcher.py tests/test_matcher.py
git commit -m "perf: add RefIndex for vectorized face matching"
```

---

### Task 5: Pipeline Integration — Wire RefIndex and assign_batch

**Files:**
- Modify: `src/waldo/pipeline.py:1-30` (imports)
- Modify: `src/waldo/pipeline.py:96-132` (`_process_video`)
- Modify: `src/waldo/pipeline.py:139-171` (`_process_photo`)
- Modify: `src/waldo/pipeline.py:178-218` (`_auto_cluster`)
- Modify: `src/waldo/pipeline.py:237-297` (`run`)

- [ ] **Step 1: Update imports in pipeline.py**

Replace the import line:

```python
from .matcher import classify_face
```

with:

```python
from .matcher import RefIndex
```

- [ ] **Step 2: Update _process_video to use RefIndex**

Replace the `_process_video` function signature and body. Change:

```python
def _process_video(
    video: Path,
    cfg: Config,
    refs: dict[str, list[np.ndarray]],
    embedder: Embedder,
    progress: Progress | None = None,
) -> list[Path]:
```

to:

```python
def _process_video(
    video: Path,
    cfg: Config,
    ref_index: RefIndex,
    persons: list[str],
    embedder: Embedder,
    progress: Progress | None = None,
) -> list[Path]:
```

And replace the matching loop (lines 105-113) — change:

```python
    per_person: dict[str, list[tuple[float, bool]]] = {p: [] for p in refs}
    for sample in cache.samples:
        present_now = {p: False for p in refs}
        for face in sample.faces:
            who = classify_face(face.embedding, refs, cfg.threshold)
            if who:
                present_now[who] = True
        for p, present in present_now.items():
            per_person[p].append((sample.t, present))
```

to:

```python
    per_person: dict[str, list[tuple[float, bool]]] = {p: [] for p in persons}
    for sample in cache.samples:
        present_now = {p: False for p in persons}
        if sample.faces:
            embeddings = np.array([f.embedding for f in sample.faces])
            matches = ref_index.classify_batch(embeddings, cfg.threshold)
            for who in matches:
                if who:
                    present_now[who] = True
        for p, present in present_now.items():
            per_person[p].append((sample.t, present))
```

- [ ] **Step 3: Update _process_photo to use RefIndex**

Replace the `_process_photo` function signature. Change:

```python
def _process_photo(
    img_path: Path,
    refs: dict[str, list[np.ndarray]],
    output_dir: Path,
    embedder: Embedder,
    threshold: float,
) -> str | None:
```

to:

```python
def _process_photo(
    img_path: Path,
    ref_index: RefIndex,
    output_dir: Path,
    embedder: Embedder,
    threshold: float,
) -> str | None:
```

And replace the matching loop inside `_process_photo` — change:

```python
    found: set[str] = set()
    for face in faces:
        who = classify_face(face.embedding, refs, threshold)
        if who:
            found.add(who)
```

to:

```python
    embeddings = np.array([f.embedding for f in faces])
    matches = ref_index.classify_batch(embeddings, threshold)
    found: set[str] = {who for who in matches if who}
```

- [ ] **Step 4: Update _auto_cluster to use assign_batch**

Replace the video clustering loop in `_auto_cluster` (lines 191-200) — change:

```python
            task = progress.add_task("clustering videos", total=len(videos))
            for video in videos:
                try:
                    cache = _ensure_cache(video, cfg, embedder, progress=progress)
                    for sample in cache.samples:
                        for face in sample.faces:
                            cluster.assign(face.embedding)
                except Exception as e:
                    log.warning("skipping %s: %s", video.name, e)
                progress.advance(task)
```

to:

```python
            task = progress.add_task("clustering videos", total=len(videos))
            for video in videos:
                try:
                    cache = _ensure_cache(video, cfg, embedder, progress=progress)
                    embs = [
                        f.embedding
                        for s in cache.samples
                        for f in s.faces
                    ]
                    if embs:
                        cluster.assign_batch(np.array(embs))
                except Exception as e:
                    log.warning("skipping %s: %s", video.name, e)
                progress.advance(task)
```

Replace the photo clustering loop (lines 203-213) — change:

```python
            task = progress.add_task("clustering photos", total=len(photos))
            for img_path in photos:
                try:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        for face in embedder.detect(img):
                            cluster.assign(face.embedding)
                except Exception as e:
                    log.warning("skipping %s: %s", img_path.name, e)
                progress.advance(task)
```

to:

```python
            task = progress.add_task("clustering photos", total=len(photos))
            for img_path in photos:
                try:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        faces = embedder.detect(img)
                        if faces:
                            embs = np.array([f.embedding for f in faces])
                            cluster.assign_batch(embs)
                except Exception as e:
                    log.warning("skipping %s: %s", img_path.name, e)
                progress.advance(task)
```

- [ ] **Step 5: Update run() to build RefIndex and pass it to callers**

In the `run()` function, after `refs_arrays` is determined (around line 249), add RefIndex construction and update all call sites. Change:

```python
    # Discover all media files recursively.
```

to:

```python
    ref_index = RefIndex(refs_arrays)
    persons = list(refs_arrays.keys())

    # Discover all media files recursively.
```

Update the `_do_video` closure — change:

```python
            def _do_video(v: Path) -> None:
                try:
                    _process_video(v, cfg, refs_arrays, embedder, progress=progress)
                except Exception as e:
```

to:

```python
            def _do_video(v: Path) -> None:
                try:
                    _process_video(v, cfg, ref_index, persons, embedder, progress=progress)
                except Exception as e:
```

Update the `_do_photo` closure — change:

```python
            def _do_photo(p: Path) -> None:
                try:
                    _process_photo(p, refs_arrays, cfg.output_dir, embedder, cfg.threshold)
                except Exception as e:
```

to:

```python
            def _do_photo(p: Path) -> None:
                try:
                    _process_photo(p, ref_index, cfg.output_dir, embedder, cfg.threshold)
                except Exception as e:
```

- [ ] **Step 6: Run all tests**

Run: `uv run pytest -v`
Expected: All tests PASS.

- [ ] **Step 7: Commit**

```bash
git add src/waldo/pipeline.py
git commit -m "perf: wire RefIndex and assign_batch into pipeline"
```

---

### Task 6: Parallel Detection in Auto-Cluster

**Files:**
- Modify: `src/waldo/pipeline.py:178-218` (`_auto_cluster`)

- [ ] **Step 1: Parallelize video cache building in _auto_cluster**

Replace the video section of `_auto_cluster` (the `if videos:` block) with:

```python
        if videos:
            task = progress.add_task("scanning videos", total=len(videos))
            caches: list[VideoCache] = []

            def _cache_video(v: Path) -> VideoCache | None:
                try:
                    return _ensure_cache(v, cfg, embedder, progress=progress)
                except Exception as e:
                    log.warning("skipping %s: %s", v.name, e)
                    return None
                finally:
                    progress.advance(task)

            if cfg.workers <= 1:
                caches = [c for v in videos if (c := _cache_video(v)) is not None]
            else:
                with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
                    caches = [c for c in pool.map(_cache_video, videos) if c is not None]

            all_embs = [
                f.embedding
                for cache in caches
                for s in cache.samples
                for f in s.faces
            ]
            if all_embs:
                cluster.assign_batch(np.array(all_embs))
```

- [ ] **Step 2: Parallelize photo detection in _auto_cluster**

Replace the photo section of `_auto_cluster` (the `if photos:` block) with:

```python
        if photos:
            task = progress.add_task("scanning photos", total=len(photos))
            photo_embs: list[np.ndarray] = []

            def _detect_photo(p: Path) -> list[np.ndarray]:
                try:
                    img = cv2.imread(str(p))
                    if img is not None:
                        return [f.embedding for f in embedder.detect(img)]
                except Exception as e:
                    log.warning("skipping %s: %s", p.name, e)
                finally:
                    progress.advance(task)
                return []

            if cfg.workers <= 1:
                for p in photos:
                    photo_embs.extend(_detect_photo(p))
            else:
                with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
                    for embs in pool.map(_detect_photo, photos):
                        photo_embs.extend(embs)

            if photo_embs:
                cluster.assign_batch(np.array(photo_embs))
```

- [ ] **Step 3: Run all tests**

Run: `uv run pytest -v`
Expected: All tests PASS.

- [ ] **Step 4: Commit**

```bash
git add src/waldo/pipeline.py
git commit -m "perf: parallelize embedding extraction in auto-cluster"
```

---

### Task 7: Optional FAISS Backend — Tests

**Files:**
- Modify: `tests/test_cluster.py`
- Modify: `tests/test_matcher.py`

- [ ] **Step 1: Add FAISS parity tests to test_cluster.py**

Add at the bottom of `tests/test_cluster.py`:

```python
import pytest

faiss = pytest.importorskip("faiss")


def test_faiss_assign_matches_numpy():
    """FAISS backend must produce identical cluster assignments."""
    embeddings = np.array([
        _norm([1, 0, 0]),
        _norm([0.95, 0.05, 0]),
        _norm([0, 1, 0]),
        _norm([0, 0.9, 0.1]),
        _norm([0, 0, 1]),
    ])
    np_cluster = FaceCluster(threshold=0.5, dim=3, backend="numpy")
    np_ids = np_cluster.assign_batch(embeddings)

    faiss_cluster = FaceCluster(threshold=0.5, dim=3, backend="faiss")
    faiss_ids = faiss_cluster.assign_batch(embeddings)

    assert faiss_ids == np_ids
    assert faiss_cluster.n_clusters == np_cluster.n_clusters
```

- [ ] **Step 2: Add FAISS parity tests to test_matcher.py**

Add at the bottom of `tests/test_matcher.py`:

```python
import pytest

faiss = pytest.importorskip("faiss")


def test_faiss_refindex_matches_numpy():
    grandma = _norm([1, 0, 0])
    grandpa = _norm([0, 1, 0])
    refs = {"grandma": [grandma], "grandpa": [grandpa]}

    np_idx = RefIndex(refs, backend="numpy")
    faiss_idx = RefIndex(refs, backend="faiss")

    embeddings = np.array([
        _norm([0.9, 0.1, 0]),
        _norm([0.1, 0.9, 0]),
        _norm([0, 0, 1]),
    ])
    np_results = np_idx.classify_batch(embeddings, threshold=0.5)
    faiss_results = faiss_idx.classify_batch(embeddings, threshold=0.5)
    assert faiss_results == np_results
```

- [ ] **Step 3: Run tests — FAISS tests should fail**

Run: `uv run pytest tests/test_cluster.py tests/test_matcher.py -v`
Expected: Non-FAISS tests PASS. FAISS tests either SKIP (if faiss not installed) or FAIL (backend param not recognized).

---

### Task 8: Optional FAISS Backend — Implementation

**Files:**
- Modify: `src/waldo/cluster.py`
- Modify: `src/waldo/matcher.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add `fast` optional dependency to pyproject.toml**

Add after the existing `dev` optional dependency:

```toml
fast = ["faiss-cpu>=1.7"]
```

So the section becomes:

```toml
[project.optional-dependencies]
dev = ["pytest>=8.0"]
fast = ["faiss-cpu>=1.7"]
```

- [ ] **Step 2: Add FAISS backend to FaceCluster**

Replace `src/waldo/cluster.py` with:

```python
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

_INITIAL_CAPACITY = 64


def _faiss_available() -> bool:
    try:
        import faiss  # noqa: F401
        return True
    except ImportError:
        return False


class FaceCluster:
    """Greedy online clustering of L2-normalised face embeddings.

    Centroids are stored in a pre-allocated matrix for BLAS-optimised
    similarity computation.  When ``faiss-cpu`` is installed, an
    ``IndexFlatIP`` is used for nearest-centroid lookup instead.
    """

    def __init__(
        self,
        threshold: float = 0.55,
        *,
        dim: int = 512,
        backend: str = "auto",
    ) -> None:
        self.threshold = threshold
        self._dim = dim
        self._capacity = _INITIAL_CAPACITY
        self._centroid_matrix = np.empty((self._capacity, dim), dtype=np.float32)
        self._counts = np.empty(self._capacity, dtype=np.int64)
        self._size = 0

        if backend == "auto":
            self._use_faiss = _faiss_available()
        else:
            self._use_faiss = backend == "faiss"

        self._index = None
        if self._use_faiss:
            import faiss
            self._index = faiss.IndexFlatIP(dim)

    @property
    def n_clusters(self) -> int:
        return self._size

    def _grow(self) -> None:
        new_cap = self._capacity * 2
        new_mat = np.empty((new_cap, self._dim), dtype=np.float32)
        new_mat[: self._size] = self._centroid_matrix[: self._size]
        self._centroid_matrix = new_mat
        new_counts = np.empty(new_cap, dtype=np.int64)
        new_counts[: self._size] = self._counts[: self._size]
        self._counts = new_counts
        self._capacity = new_cap

    def _rebuild_index(self) -> None:
        if self._index is not None:
            self._index.reset()
            if self._size > 0:
                self._index.add(np.ascontiguousarray(self._centroid_matrix[: self._size]))

    def assign(self, embedding: np.ndarray) -> int:
        """Return the cluster index for this embedding (creating one if needed)."""
        if self._size == 0:
            self._centroid_matrix[0] = embedding
            self._counts[0] = 1
            self._size = 1
            self._rebuild_index()
            return 0

        if self._index is not None:
            _, I = self._index.search(embedding.reshape(1, -1), 1)
            best_idx = int(I[0, 0])
            best_sim = float(np.dot(self._centroid_matrix[best_idx], embedding))
        else:
            sims = self._centroid_matrix[: self._size] @ embedding
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])

        if best_sim >= self.threshold:
            n = int(self._counts[best_idx])
            new_c = (self._centroid_matrix[best_idx] * n + embedding) / (n + 1)
            norm = np.linalg.norm(new_c)
            if norm > 0:
                new_c /= norm
            self._centroid_matrix[best_idx] = new_c
            self._counts[best_idx] = n + 1
            self._rebuild_index()
            return best_idx

        if self._size >= self._capacity:
            self._grow()
        self._centroid_matrix[self._size] = embedding
        self._counts[self._size] = 1
        self._size += 1
        self._rebuild_index()
        return self._size - 1

    def assign_batch(self, embeddings: np.ndarray) -> list[int]:
        """Assign a batch of embeddings. Order-dependent — processes sequentially
        but each similarity computation is BLAS-optimised."""
        if len(embeddings) == 0:
            return []
        return [self.assign(e) for e in embeddings]

    def centroids_as_refs(self, min_count: int = 2) -> dict[str, list[np.ndarray]]:
        """Export clusters as a refs dict (same shape as load_references output).

        Clusters with fewer than ``min_count`` faces are dropped (likely noise).
        """
        refs: dict[str, list[np.ndarray]] = {}
        for i in range(self._size):
            n = int(self._counts[i])
            if n < min_count:
                log.debug("cluster person_%d has only %d face(s), skipping", i + 1, n)
                continue
            refs[f"person_{i + 1}"] = [self._centroid_matrix[i].copy()]
        return refs
```

- [ ] **Step 3: Add FAISS backend to RefIndex**

Replace `src/waldo/matcher.py` with:

```python
from __future__ import annotations

import numpy as np


def _faiss_available() -> bool:
    try:
        import faiss  # noqa: F401
        return True
    except ImportError:
        return False


class RefIndex:
    """Pre-stacked reference embeddings for fast batch matching.

    All reference embeddings are stacked into a single matrix so that
    classification becomes a single BLAS matmul + argmax.  When
    ``faiss-cpu`` is installed, uses ``IndexFlatIP`` for the lookup.
    """

    def __init__(
        self,
        refs: dict[str, list[np.ndarray]],
        backend: str = "auto",
    ) -> None:
        names: list[str] = []
        rows: list[np.ndarray] = []
        for name, embs in refs.items():
            for e in embs:
                names.append(name)
                rows.append(e)
        self._names = names
        self._matrix = (
            np.vstack(rows).astype(np.float32) if rows else np.empty((0, 0), dtype=np.float32)
        )

        if backend == "auto":
            use_faiss = _faiss_available() and self.n_refs > 0
        else:
            use_faiss = backend == "faiss" and self.n_refs > 0

        self._index = None
        if use_faiss:
            import faiss
            self._index = faiss.IndexFlatIP(self._matrix.shape[1])
            self._index.add(np.ascontiguousarray(self._matrix))

    @property
    def n_refs(self) -> int:
        return len(self._names)

    def classify(self, embedding: np.ndarray, threshold: float) -> str | None:
        """Return the person whose reference is closest, or None if below threshold."""
        if self.n_refs == 0:
            return None

        if self._index is not None:
            D, I = self._index.search(embedding.reshape(1, -1), 1)
            best_idx = int(I[0, 0])
            best_sim = float(D[0, 0])
        else:
            sims = self._matrix @ embedding
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])

        if best_sim >= threshold:
            return self._names[best_idx]
        return None

    def classify_batch(
        self, embeddings: np.ndarray, threshold: float
    ) -> list[str | None]:
        """Classify multiple embeddings in one operation."""
        n = len(embeddings)
        if n == 0 or self.n_refs == 0:
            return [None] * n

        if self._index is not None:
            D, I = self._index.search(np.ascontiguousarray(embeddings), 1)
            return [
                self._names[int(I[i, 0])] if float(D[i, 0]) >= threshold else None
                for i in range(n)
            ]

        sim_matrix = self._matrix @ embeddings.T
        best_indices = np.argmax(sim_matrix, axis=0)
        best_sims = sim_matrix[best_indices, np.arange(n)]
        return [
            self._names[int(idx)] if float(sim) >= threshold else None
            for idx, sim in zip(best_indices, best_sims)
        ]
```

- [ ] **Step 4: Run all tests**

Run: `uv run pytest -v`
Expected: All tests PASS. FAISS tests SKIP if faiss not installed, PASS if installed.

- [ ] **Step 5: Commit**

```bash
git add src/waldo/cluster.py src/waldo/matcher.py pyproject.toml
git commit -m "perf: add optional FAISS backend for clustering and matching"
```

---

### Task 9: Final Verification

**Files:** None (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest -v`
Expected: All tests PASS.

- [ ] **Step 2: Run type check if available**

Run: `uv run python -c "from waldo.cluster import FaceCluster; from waldo.matcher import RefIndex; print('imports ok')"`
Expected: `imports ok`

- [ ] **Step 3: Verify no regressions in pipeline imports**

Run: `uv run python -c "from waldo.pipeline import run; print('pipeline ok')"`
Expected: `pipeline ok`

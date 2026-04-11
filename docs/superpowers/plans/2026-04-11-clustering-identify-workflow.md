# Clustering & Identify Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the waldo pipeline into four CLI commands (`scan`, `identify`, `extract`, `run`) with a folder-based refs interface between steps.

**Architecture:** The current monolithic `run()` in pipeline.py is split into `scan_faces()` (cluster + save crops), `extract_clips()` (match + cut), with a new `identify` module for the interactive naming flow. The refs folder (containing face crop JPEGs organized by person) is the only interface between stages. A new `cropper.py` handles face crop extraction. A new `identify.py` handles HTML generation and terminal prompts.

**Tech Stack:** Python, Typer (CLI), OpenCV (image cropping), Pydantic (config), Rich (progress), webbrowser (HTML display), base64 (inline images in HTML)

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/waldo/cropper.py` | Create | Extract face crops from frames using bboxes with padding |
| `src/waldo/identify.py` | Create | Generate HTML page for clusters, run terminal prompt loop |
| `src/waldo/pipeline.py` | Modify | Split `run()` into `scan_faces()` and `extract_clips()`, add `run_all()` |
| `src/waldo/cli.py` | Modify | Replace single `scan` command with `scan`, `identify`, `extract`, `run` |
| `src/waldo/config.py` | Modify | Make `output_dir` optional (not needed for `scan` or `identify`) |
| `src/waldo/cluster.py` | Modify | Track which faces belong to which cluster (for crop extraction) |
| `tests/test_cropper.py` | Create | Unit tests for face crop extraction |
| `tests/test_identify.py` | Create | Unit tests for HTML generation and identify flow |

---

### Task 1: Add face-to-cluster tracking in FaceCluster

Currently `FaceCluster.assign()` returns a cluster index but doesn't track which faces (frames + bboxes) belong to each cluster. We need this to save representative face crops.

**Files:**
- Modify: `src/waldo/cluster.py`
- Test: `tests/test_cluster.py`

- [ ] **Step 1: Read existing cluster tests**

Read `tests/test_cluster.py` to understand test patterns.

- [ ] **Step 2: Write failing test for face tracking**

In `tests/test_cluster.py`, add:

```python
def test_cluster_tracks_face_sources():
    """FaceCluster.assign() with source info stores (source_id, bbox) per cluster."""
    cluster = FaceCluster(threshold=0.5)

    emb_a = _random_emb(42)
    emb_b = _perturbation(emb_a, 0.05)  # same person
    emb_c = _random_emb(99)  # different person

    cluster.assign(emb_a, source_id="vid1_t0.5", bbox=(10, 20, 50, 60))
    cluster.assign(emb_b, source_id="vid1_t1.0", bbox=(12, 22, 52, 62))
    cluster.assign(emb_c, source_id="vid1_t1.5", bbox=(100, 100, 200, 200))

    sources = cluster.get_sources()
    # Two clusters: cluster 0 has 2 faces, cluster 1 has 1 face
    assert len(sources) == 2
    assert len(sources[0]) == 2
    assert sources[0][0] == ("vid1_t0.5", (10, 20, 50, 60))
    assert sources[0][1] == ("vid1_t1.0", (12, 22, 52, 62))
    assert len(sources[1]) == 1
    assert sources[1][0] == ("vid1_t1.5", (100, 100, 200, 200))
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd /Users/alberthurwitz/Projects/waldo && python -m pytest tests/test_cluster.py::test_cluster_tracks_face_sources -v`
Expected: FAIL — `assign()` doesn't accept `source_id` / `bbox` kwargs

- [ ] **Step 4: Implement face tracking in FaceCluster**

In `src/waldo/cluster.py`, update `FaceCluster`:

```python
class FaceCluster:
    def __init__(self, threshold: float = 0.55) -> None:
        self.threshold = threshold
        self._centroids: list[np.ndarray] = []
        self._counts: list[int] = []
        self._sources: list[list[tuple[str, tuple[float, float, float, float]]]] = []

    def assign(
        self,
        embedding: np.ndarray,
        source_id: str | None = None,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> int:
        """Return the cluster index for this embedding (creating one if needed)."""
        if not self._centroids:
            self._centroids.append(embedding.copy())
            self._counts.append(1)
            self._sources.append([])
            if source_id is not None and bbox is not None:
                self._sources[0].append((source_id, bbox))
            return 0

        sims = np.array([float(np.dot(embedding, c)) for c in self._centroids])
        best_idx = int(np.argmax(sims))
        best_sim = sims[best_idx]

        if best_sim >= self.threshold:
            n = self._counts[best_idx]
            new_c = (self._centroids[best_idx] * n + embedding) / (n + 1)
            norm = np.linalg.norm(new_c)
            if norm > 0:
                new_c /= norm
            self._centroids[best_idx] = new_c
            self._counts[best_idx] = n + 1
            if source_id is not None and bbox is not None:
                self._sources[best_idx].append((source_id, bbox))
            return best_idx

        self._centroids.append(embedding.copy())
        self._counts.append(1)
        self._sources.append([])
        if source_id is not None and bbox is not None:
            self._sources[-1].append((source_id, bbox))
        return self.n_clusters - 1

    def get_sources(self) -> list[list[tuple[str, tuple[float, float, float, float]]]]:
        """Return per-cluster list of (source_id, bbox) tuples."""
        return self._sources
```

Note: `source_id` and `bbox` are optional so existing callers (which pass only `embedding`) continue to work unchanged.

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /Users/alberthurwitz/Projects/waldo && python -m pytest tests/test_cluster.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add src/waldo/cluster.py tests/test_cluster.py
git commit -m "feat: track face sources (source_id, bbox) in FaceCluster"
```

---

### Task 2: Create cropper module

Extracts a face crop from a frame given a bounding box, with padding and resize.

**Files:**
- Create: `src/waldo/cropper.py`
- Create: `tests/test_cropper.py`

- [ ] **Step 1: Write failing test for crop_face**

Create `tests/test_cropper.py`:

```python
import numpy as np
import pytest

from waldo.cropper import crop_face


def test_crop_face_basic():
    """crop_face extracts a padded region from a frame."""
    # 100x200 BGR frame (height=100, width=200)
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    # Draw a white rectangle where the "face" is
    frame[30:60, 80:120] = 255

    # bbox: x1=80, y1=30, x2=120, y2=60
    crop = crop_face(frame, (80.0, 30.0, 120.0, 60.0), pad_fraction=0.2)

    # Should be a non-empty image
    assert crop.shape[2] == 3  # BGR
    assert crop.shape[0] > 0
    assert crop.shape[1] > 0


def test_crop_face_clamps_to_frame():
    """Padding near the edge clamps to frame boundaries."""
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    # bbox near top-left corner
    crop = crop_face(frame, (0.0, 0.0, 20.0, 20.0), pad_fraction=0.5)

    assert crop.shape[0] > 0
    assert crop.shape[1] > 0


def test_crop_face_target_size():
    """crop_face resizes to target_size when specified."""
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    crop = crop_face(frame, (50.0, 50.0, 150.0, 150.0), target_size=64)

    assert crop.shape[0] == 64
    assert crop.shape[1] == 64
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/alberthurwitz/Projects/waldo && python -m pytest tests/test_cropper.py -v`
Expected: FAIL — ModuleNotFoundError: No module named 'waldo.cropper'

- [ ] **Step 3: Implement crop_face**

Create `src/waldo/cropper.py`:

```python
from __future__ import annotations

import cv2
import numpy as np


def crop_face(
    frame: np.ndarray,
    bbox: tuple[float, float, float, float],
    pad_fraction: float = 0.2,
    target_size: int | None = None,
) -> np.ndarray:
    """Extract a face crop from a frame with padding.

    Args:
        frame: BGR image (H, W, 3).
        bbox: (x1, y1, x2, y2) face bounding box.
        pad_fraction: Fraction of bbox size to pad on each side (0.2 = 20%).
        target_size: If set, resize the crop to (target_size, target_size).

    Returns:
        BGR image of the cropped face region.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1

    pad_x = bw * pad_fraction
    pad_y = bh * pad_fraction

    cx1 = max(0, int(x1 - pad_x))
    cy1 = max(0, int(y1 - pad_y))
    cx2 = min(w, int(x2 + pad_x))
    cy2 = min(h, int(y2 + pad_y))

    crop = frame[cy1:cy2, cx1:cx2]

    if target_size is not None:
        crop = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_AREA)

    return crop
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/alberthurwitz/Projects/waldo && python -m pytest tests/test_cropper.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/waldo/cropper.py tests/test_cropper.py
git commit -m "feat: add cropper module for extracting face crops from frames"
```

---

### Task 3: Create save_crops function

Saves representative face crops from clusters to the refs folder structure.

**Files:**
- Modify: `src/waldo/cropper.py`
- Modify: `tests/test_cropper.py`

- [ ] **Step 1: Write failing test for save_crops**

Add to `tests/test_cropper.py`:

```python
from pathlib import Path
from waldo.cropper import crop_face, save_crops


def test_save_crops_creates_folders(tmp_path):
    """save_crops creates person_N folders with face crop JPEGs."""
    # Simulate 2 clusters, each with 2 frames and bboxes
    frames = {
        "vid1_t0.5": np.zeros((100, 100, 3), dtype=np.uint8),
        "vid1_t1.0": np.zeros((100, 100, 3), dtype=np.uint8),
        "vid2_t0.0": np.zeros((100, 100, 3), dtype=np.uint8),
    }
    # cluster_sources: per-cluster list of (source_id, bbox)
    cluster_sources = [
        [("vid1_t0.5", (10, 10, 50, 50)), ("vid1_t1.0", (12, 12, 52, 52))],
        [("vid2_t0.0", (20, 20, 60, 60))],
    ]
    # cluster_counts: number of faces per cluster (may be > len(sources) if some had no source tracking)
    cluster_counts = [5, 3]

    refs_dir = tmp_path / "refs"
    save_crops(refs_dir, cluster_sources, cluster_counts, frames, max_per_cluster=5, min_count=2)

    # person_1 should exist (count=5 >= min_count=2)
    assert (refs_dir / "person_1").is_dir()
    jpgs = list((refs_dir / "person_1").glob("*.jpg"))
    assert len(jpgs) == 2  # 2 source faces available

    # person_2 should exist (count=3 >= min_count=2)
    assert (refs_dir / "person_2").is_dir()
    jpgs = list((refs_dir / "person_2").glob("*.jpg"))
    assert len(jpgs) == 1


def test_save_crops_filters_small_clusters(tmp_path):
    """Clusters below min_count are not saved."""
    frames = {"vid1_t0.5": np.zeros((100, 100, 3), dtype=np.uint8)}
    cluster_sources = [[("vid1_t0.5", (10, 10, 50, 50))]]
    cluster_counts = [1]  # below min_count

    refs_dir = tmp_path / "refs"
    save_crops(refs_dir, cluster_sources, cluster_counts, frames, max_per_cluster=5, min_count=2)

    assert not (refs_dir / "person_1").exists()


def test_save_crops_max_per_cluster(tmp_path):
    """At most max_per_cluster crops are saved."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frames = {f"v_t{i}": frame for i in range(10)}
    sources = [(f"v_t{i}", (10, 10, 50, 50)) for i in range(10)]
    cluster_sources = [sources]
    cluster_counts = [10]

    refs_dir = tmp_path / "refs"
    save_crops(refs_dir, cluster_sources, cluster_counts, frames, max_per_cluster=5, min_count=2)

    jpgs = list((refs_dir / "person_1").glob("*.jpg"))
    assert len(jpgs) == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/alberthurwitz/Projects/waldo && python -m pytest tests/test_cropper.py::test_save_crops_creates_folders -v`
Expected: FAIL — ImportError: cannot import name 'save_crops'

- [ ] **Step 3: Implement save_crops**

Add to `src/waldo/cropper.py`:

```python
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def save_crops(
    refs_dir: Path,
    cluster_sources: list[list[tuple[str, tuple[float, float, float, float]]]],
    cluster_counts: list[int],
    frames: dict[str, np.ndarray],
    max_per_cluster: int = 5,
    min_count: int = 2,
) -> list[str]:
    """Save face crops to refs_dir/person_N/ folders.

    Args:
        refs_dir: Root directory for reference crops.
        cluster_sources: Per-cluster list of (source_id, bbox) from FaceCluster.
        cluster_counts: Total face count per cluster (from FaceCluster._counts).
        frames: Dict mapping source_id → BGR frame.
        max_per_cluster: Maximum crops to save per person.
        min_count: Minimum faces in a cluster to save it.

    Returns:
        List of person folder names that were created (e.g. ["person_1", "person_3"]).
    """
    created: list[str] = []
    for i, (sources, count) in enumerate(zip(cluster_sources, cluster_counts)):
        if count < min_count:
            log.debug("cluster person_%d has only %d face(s), skipping", i + 1, count)
            continue

        name = f"person_{i + 1}"
        person_dir = refs_dir / name
        person_dir.mkdir(parents=True, exist_ok=True)

        # Pick evenly spaced sources if we have more than max_per_cluster
        if len(sources) > max_per_cluster:
            step = len(sources) / max_per_cluster
            indices = [int(step * j) for j in range(max_per_cluster)]
            selected = [sources[idx] for idx in indices]
        else:
            selected = sources

        for j, (source_id, bbox) in enumerate(selected):
            frame = frames.get(source_id)
            if frame is None:
                continue
            crop = crop_face(frame, bbox)
            out_path = person_dir / f"face_{j + 1:03d}.jpg"
            cv2.imwrite(str(out_path), crop)

        created.append(name)
        log.info("saved %d crop(s) for %s", len(selected), name)

    return created
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/alberthurwitz/Projects/waldo && python -m pytest tests/test_cropper.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/waldo/cropper.py tests/test_cropper.py
git commit -m "feat: add save_crops to write face crops to refs folder structure"
```

---

### Task 4: Create identify module — HTML generation

Generates a self-contained HTML page showing face crops per cluster.

**Files:**
- Create: `src/waldo/identify.py`
- Create: `tests/test_identify.py`

- [ ] **Step 1: Write failing test for generate_html**

Create `tests/test_identify.py`:

```python
import base64
from pathlib import Path

import cv2
import numpy as np

from waldo.identify import generate_html


def _make_refs(tmp_path: Path, names: list[str], faces_per: int = 2) -> Path:
    """Create a fake refs folder with JPEG face crops."""
    refs_dir = tmp_path / "refs"
    for name in names:
        d = refs_dir / name
        d.mkdir(parents=True)
        for i in range(faces_per):
            img = np.zeros((64, 64, 3), dtype=np.uint8)
            cv2.imwrite(str(d / f"face_{i + 1:03d}.jpg"), img)
    return refs_dir


def test_generate_html_contains_sections(tmp_path):
    """HTML has a section per cluster with the folder name as heading."""
    refs_dir = _make_refs(tmp_path, ["person_1", "person_2"])
    html_path = generate_html(refs_dir, tmp_path / "identify.html")

    assert html_path.exists()
    content = html_path.read_text()
    assert "person_1" in content
    assert "person_2" in content


def test_generate_html_embeds_images(tmp_path):
    """Face crops are embedded as base64 data URIs."""
    refs_dir = _make_refs(tmp_path, ["person_1"], faces_per=1)
    html_path = generate_html(refs_dir, tmp_path / "identify.html")

    content = html_path.read_text()
    assert "data:image/jpeg;base64," in content


def test_generate_html_empty_refs(tmp_path):
    """Empty refs folder produces valid HTML with a 'no clusters' message."""
    refs_dir = tmp_path / "refs"
    refs_dir.mkdir()
    html_path = generate_html(refs_dir, tmp_path / "identify.html")

    content = html_path.read_text()
    assert "No clusters" in content or "no clusters" in content
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/alberthurwitz/Projects/waldo && python -m pytest tests/test_identify.py -v`
Expected: FAIL — ModuleNotFoundError: No module named 'waldo.identify'

- [ ] **Step 3: Implement generate_html**

Create `src/waldo/identify.py`:

```python
from __future__ import annotations

import base64
import logging
from pathlib import Path

from .config import IMG_EXTS

log = logging.getLogger(__name__)


def generate_html(refs_dir: Path, out_path: Path) -> Path:
    """Generate a self-contained HTML page showing face crops per cluster.

    Args:
        refs_dir: Directory containing person_N/ subfolders with face crop JPEGs.
        out_path: Where to write the HTML file.

    Returns:
        Path to the written HTML file.
    """
    folders = sorted(
        d for d in refs_dir.iterdir() if d.is_dir()
    ) if refs_dir.is_dir() else []

    sections: list[str] = []
    for folder in folders:
        images = sorted(
            f for f in folder.iterdir()
            if f.suffix.lower() in IMG_EXTS
        )
        if not images:
            continue

        img_tags: list[str] = []
        for img_path in images:
            data = base64.b64encode(img_path.read_bytes()).decode("ascii")
            ext = img_path.suffix.lower().lstrip(".")
            mime = "jpeg" if ext in ("jpg", "jpeg") else ext
            img_tags.append(
                f'<img src="data:image/{mime};base64,{data}" '
                f'style="width:128px;height:128px;object-fit:cover;border-radius:8px;margin:4px;">'
            )

        sections.append(
            f"<div style='margin-bottom:32px;'>"
            f"<h2>{folder.name}</h2>"
            f"<div style='display:flex;flex-wrap:wrap;'>"
            f"{''.join(img_tags)}"
            f"</div></div>"
        )

    if not sections:
        body = "<p>No clusters found. Run <code>waldo scan</code> first.</p>"
    else:
        body = "\n".join(sections)

    html = (
        "<!DOCTYPE html><html><head>"
        "<meta charset='utf-8'>"
        "<title>Waldo — Identify Faces</title>"
        "<style>body{font-family:system-ui,sans-serif;max-width:900px;margin:40px auto;padding:0 20px;}"
        "h1{border-bottom:2px solid #333;padding-bottom:8px;}"
        "h2{color:#555;}</style>"
        "</head><body>"
        "<h1>Waldo — Identify Faces</h1>"
        f"{body}"
        "</body></html>"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    return out_path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/alberthurwitz/Projects/waldo && python -m pytest tests/test_identify.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/waldo/identify.py tests/test_identify.py
git commit -m "feat: add HTML generation for face cluster identification"
```

---

### Task 5: Create identify module — terminal prompt loop

Implements the interactive rename/skip/delete flow.

**Files:**
- Modify: `src/waldo/identify.py`
- Modify: `tests/test_identify.py`

- [ ] **Step 1: Write failing test for run_identify_prompts**

Add to `tests/test_identify.py`:

```python
from unittest.mock import patch
from waldo.identify import generate_html, run_identify_prompts


def test_identify_rename(tmp_path):
    """Typing a name renames the folder."""
    refs_dir = _make_refs(tmp_path, ["person_1"])

    with patch("builtins.input", return_value="grandma"):
        stats = run_identify_prompts(refs_dir)

    assert not (refs_dir / "person_1").exists()
    assert (refs_dir / "grandma").is_dir()
    assert stats == {"identified": 1, "skipped": 0, "deleted": 0}


def test_identify_skip(tmp_path):
    """Typing 'skip' leaves the folder unchanged."""
    refs_dir = _make_refs(tmp_path, ["person_1"])

    with patch("builtins.input", return_value="skip"):
        stats = run_identify_prompts(refs_dir)

    assert (refs_dir / "person_1").is_dir()
    assert stats == {"identified": 0, "skipped": 1, "deleted": 0}


def test_identify_delete(tmp_path):
    """Typing 'delete' removes the folder."""
    refs_dir = _make_refs(tmp_path, ["person_1"])

    with patch("builtins.input", return_value="delete"):
        stats = run_identify_prompts(refs_dir)

    assert not (refs_dir / "person_1").exists()
    assert stats == {"identified": 0, "skipped": 0, "deleted": 1}


def test_identify_merge_collision(tmp_path):
    """Typing an existing folder name merges crops into it."""
    refs_dir = _make_refs(tmp_path, ["grandma", "person_2"], faces_per=2)

    # When prompted for person_2, type "grandma" to merge
    # grandma already exists so only person_2 gets prompted
    with patch("builtins.input", return_value="grandma"):
        stats = run_identify_prompts(refs_dir)

    assert (refs_dir / "grandma").is_dir()
    assert not (refs_dir / "person_2").exists()
    # grandma now has 4 crops (2 original + 2 merged)
    assert len(list((refs_dir / "grandma").glob("*.jpg"))) == 4
    assert stats == {"identified": 1, "skipped": 0, "deleted": 0}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/alberthurwitz/Projects/waldo && python -m pytest tests/test_identify.py::test_identify_rename -v`
Expected: FAIL — ImportError: cannot import name 'run_identify_prompts'

- [ ] **Step 3: Implement run_identify_prompts**

Add to `src/waldo/identify.py`:

```python
import shutil


def run_identify_prompts(refs_dir: Path) -> dict[str, int]:
    """Prompt the user to name each cluster folder.

    Args:
        refs_dir: Directory containing person subfolders.

    Returns:
        Dict with counts: {"identified": N, "skipped": N, "deleted": N}
    """
    folders = sorted(d for d in refs_dir.iterdir() if d.is_dir())
    stats = {"identified": 0, "skipped": 0, "deleted": 0}

    for folder in folders:
        answer = input(f"Who is {folder.name}? (name / skip / delete): ").strip()

        if answer.lower() == "skip" or answer == "":
            stats["skipped"] += 1
            continue

        if answer.lower() == "delete":
            shutil.rmtree(folder)
            stats["deleted"] += 1
            log.info("deleted %s", folder.name)
            continue

        # Rename / merge
        target = refs_dir / answer
        if target.exists() and target != folder:
            # Merge: move all files from folder into target
            for f in folder.iterdir():
                dest = target / f.name
                n = 1
                while dest.exists():
                    dest = target / f"{f.stem}_{n}{f.suffix}"
                    n += 1
                shutil.move(str(f), str(dest))
            folder.rmdir()
            log.info("merged %s into %s", folder.name, answer)
        else:
            folder.rename(target)
            log.info("renamed %s to %s", folder.name, answer)

        stats["identified"] += 1

    return stats
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/alberthurwitz/Projects/waldo && python -m pytest tests/test_identify.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/waldo/identify.py tests/test_identify.py
git commit -m "feat: add interactive identify prompts with rename/skip/delete/merge"
```

---

### Task 6: Split pipeline.py — extract scan_faces()

Extract the clustering + crop-saving logic into a `scan_faces()` function.

**Files:**
- Modify: `src/waldo/pipeline.py`

- [ ] **Step 1: Implement scan_faces()**

In `src/waldo/pipeline.py`, add the `scan_faces()` function. This refactors `_auto_cluster` to also track face sources and save crops. The function:
1. Discovers all media files
2. Scans frames, detects/embeds faces
3. Clusters them with source tracking
4. Saves face crops to refs_dir

```python
from .cropper import crop_face, save_crops


def scan_faces(cfg: Config) -> list[str]:
    """Cluster all faces in input media and save crops to refs_dir.

    Returns list of person folder names created (e.g. ["person_1", "person_2"]).
    """
    embedder = Embedder()
    all_files = sorted(
        p for p in cfg.input_dir.rglob("*")
        if p.is_file() and not any(part.startswith(".") for part in p.relative_to(cfg.input_dir).parts)
    )
    videos = [f for f in all_files if f.suffix.lower() in VIDEO_EXTS]
    photos = [f for f in all_files if f.suffix.lower() in IMG_EXTS]

    if not videos and not photos:
        log.warning("no videos or images found in %s", cfg.input_dir)
        return []

    cluster = FaceCluster(threshold=cfg.threshold)
    # frames dict: source_id → BGR frame (for cropping later)
    frames: dict[str, np.ndarray] = {}

    with _make_progress() as progress:
        if videos:
            task = progress.add_task("scanning videos", total=len(videos))
            for video in videos:
                try:
                    cache = _ensure_cache(video, cfg, embedder, progress=progress)
                    # We need original frames for cropping — re-read just the ones with faces
                    frames_needed: list[tuple[float, tuple[float, float, float, float]]] = []
                    for sample in cache.samples:
                        for face in sample.faces:
                            source_id = f"{video.stem}_t{sample.t:.3f}"
                            cluster.assign(face.embedding, source_id=source_id, bbox=face.bbox)
                            frames_needed.append((sample.t, face.bbox))

                    # Re-read frames that had faces to get crops
                    if frames_needed:
                        needed_times = {t for t, _ in frames_needed}
                        for t, frame in sample_frames(video, cfg.fps):
                            if t in needed_times:
                                source_id = f"{video.stem}_t{t:.3f}"
                                if source_id not in frames:
                                    frames[source_id] = _downscale(frame)
                            if len(frames) >= len(needed_times) * 2:
                                # Heuristic: stop early if we have enough
                                break
                except Exception as e:
                    log.warning("skipping %s: %s", video.name, e)
                progress.advance(task)

        if photos:
            task = progress.add_task("scanning photos", total=len(photos))
            for img_path in photos:
                try:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        for face in embedder.detect(img):
                            source_id = f"{img_path.stem}"
                            cluster.assign(face.embedding, source_id=source_id, bbox=face.bbox)
                            if source_id not in frames:
                                frames[source_id] = img
                except Exception as e:
                    log.warning("skipping %s: %s", img_path.name, e)
                progress.advance(task)

    if cluster.n_clusters == 0:
        log.warning("no faces detected in any media")
        cfg.refs_dir.mkdir(parents=True, exist_ok=True)
        return []

    created = save_crops(
        cfg.refs_dir,
        cluster.get_sources(),
        cluster._counts,
        frames,
        max_per_cluster=5,
        min_count=2,
    )

    log.info("scan complete: %d cluster(s), %d saved to %s", cluster.n_clusters, len(created), cfg.refs_dir)
    return created
```

Note: The `needed_times` matching uses float equality which works because timestamps come from the same `frame_idx / src_fps` calculation in the cache and in `sample_frames`. Both paths use the identical arithmetic.

- [ ] **Step 2: Run existing tests to verify nothing broke**

Run: `cd /Users/alberthurwitz/Projects/waldo && python -m pytest -v`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add src/waldo/pipeline.py
git commit -m "feat: add scan_faces() to pipeline — cluster + save crops to refs folder"
```

---

### Task 7: Split pipeline.py — extract extract_clips()

Rename the extraction portion of `run()` into `extract_clips()`.

**Files:**
- Modify: `src/waldo/pipeline.py`

- [ ] **Step 1: Implement extract_clips()**

In `src/waldo/pipeline.py`, add `extract_clips()`. This is essentially the current `run()` body from the `load_references()` call onward:

```python
def extract_clips(cfg: Config) -> None:
    """Match faces against references and cut clips / copy photos."""
    check_ffmpeg()
    embedder = Embedder()

    from .references import load_references
    refs_arrays = load_references(cfg.refs_dir, embedder)

    all_files = sorted(
        p for p in cfg.input_dir.rglob("*")
        if p.is_file() and not any(part.startswith(".") for part in p.relative_to(cfg.input_dir).parts)
    )
    videos = [f for f in all_files if f.suffix.lower() in VIDEO_EXTS]
    photos = [f for f in all_files if f.suffix.lower() in IMG_EXTS]

    if not videos and not photos:
        log.warning("no videos or images found in %s", cfg.input_dir)
        return

    with _make_progress() as progress:
        if videos:
            log.info("found %d video(s)", len(videos))
            videos_task = progress.add_task("videos", total=len(videos))

            def _do_video(v: Path) -> None:
                try:
                    _process_video(v, cfg, refs_arrays, embedder, progress=progress)
                except Exception as e:
                    log.exception("failed processing %s: %s", v, e)
                progress.advance(videos_task)

            if cfg.workers <= 1:
                for v in videos:
                    _do_video(v)
            else:
                with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
                    list(pool.map(_do_video, videos))

        if photos:
            log.info("found %d photo(s)", len(photos))
            photos_task = progress.add_task("photos", total=len(photos))

            def _do_photo(p: Path) -> None:
                try:
                    _process_photo(p, refs_arrays, cfg.output_dir, embedder, cfg.threshold)
                except Exception as e:
                    log.exception("failed processing %s: %s", p, e)
                progress.advance(photos_task)

            if cfg.workers <= 1:
                for p in photos:
                    _do_photo(p)
            else:
                with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
                    list(pool.map(_do_photo, photos))
```

- [ ] **Step 2: Update run() to call scan_faces() + extract_clips()**

Update the existing `run()` function to be a thin orchestrator:

```python
def run(cfg: Config, auto: bool = False) -> None:
    """Run the full pipeline: scan → (identify) → extract.

    If auto=True, skips the identify step and uses generic person_N names.
    If auto=False, runs interactive identify between scan and extract.
    """
    if cfg.refs_dir is not None and cfg.refs_dir.is_dir() and any(cfg.refs_dir.iterdir()):
        # Refs already exist — skip scan, go straight to extract
        extract_clips(cfg)
        return

    scan_faces(cfg)

    if not auto:
        import webbrowser
        from .identify import generate_html, run_identify_prompts

        html_path = generate_html(cfg.refs_dir, cfg.refs_dir / ".identify.html")
        webbrowser.open(html_path.as_uri())
        stats = run_identify_prompts(cfg.refs_dir)
        print(f"Done! {stats['identified']} identified, {stats['skipped']} skipped, {stats['deleted']} deleted.")

    extract_clips(cfg)
```

- [ ] **Step 3: Run existing tests**

Run: `cd /Users/alberthurwitz/Projects/waldo && python -m pytest -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add src/waldo/pipeline.py
git commit -m "refactor: split pipeline into scan_faces(), extract_clips(), and run()"
```

---

### Task 8: Update config.py — make output_dir optional

`scan` and `identify` don't need `output_dir`. Make it optional.

**Files:**
- Modify: `src/waldo/config.py`

- [ ] **Step 1: Update Config model**

In `src/waldo/config.py`, change `output_dir` to optional and update the validator:

```python
class Config(BaseModel):
    model_config = {"frozen": True}

    input_dir: Path
    refs_dir: Path | None = None
    output_dir: Path | None = None
    fps: float = Field(default=2.0, gt=0)
    threshold: float = Field(default=0.5, ge=0, le=1)
    gap: float = Field(default=2.0, ge=0)
    min_len: float = Field(default=0.5, ge=0)
    pad: float = Field(default=15.0, ge=0)
    workers: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _check_dirs(self) -> "Config":
        if self.output_dir is not None and self.input_dir.resolve() == self.output_dir.resolve():
            raise ValueError("--input and --output must be different directories")
        return self
```

- [ ] **Step 2: Run existing tests**

Run: `cd /Users/alberthurwitz/Projects/waldo && python -m pytest -v`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add src/waldo/config.py
git commit -m "refactor: make output_dir optional in Config for scan/identify commands"
```

---

### Task 9: Rewrite CLI with four commands

Replace the single `scan` command with `scan`, `identify`, `extract`, `run`.

**Files:**
- Modify: `src/waldo/cli.py`

- [ ] **Step 1: Rewrite cli.py**

Replace the contents of `src/waldo/cli.py`:

```python
from __future__ import annotations

import logging
import webbrowser
from pathlib import Path
from typing import Optional

import typer

from .config import Config

app = typer.Typer(
    add_completion=False,
    help="Where's Waldo, but for real faces. Find people in videos and photos using face recognition.",
)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@app.command()
def scan(
    input: Path = typer.Option(
        ..., exists=True, file_okay=False, dir_okay=True,
        help="Folder of media to scan for faces.",
    ),
    refs: Path = typer.Option(
        ..., file_okay=False, dir_okay=True,
        help="Where to save clustered face crops (e.g. refs/).",
    ),
    fps: float = typer.Option(2.0, help="Frames per second to sample."),
    threshold: float = typer.Option(0.5, help="Clustering similarity threshold (0-1)."),
    workers: int = typer.Option(1, help="Concurrent files to process."),
) -> None:
    """Detect and cluster all faces in input media. Saves face crops to --refs folder."""
    _setup_logging()

    if refs.is_dir() and any(refs.iterdir()):
        overwrite = typer.confirm(f"Refs folder {refs} already exists. Overwrite?")
        if not overwrite:
            raise typer.Abort()
        import shutil
        shutil.rmtree(refs)

    cfg = Config(input_dir=input, refs_dir=refs, fps=fps, threshold=threshold, workers=workers)

    from .pipeline import scan_faces
    created = scan_faces(cfg)

    if created:
        typer.echo(f"\nSaved {len(created)} cluster(s) to {refs}/")
        typer.echo(f"Next: rename folders manually, or run: waldo identify --refs {refs}")
    else:
        typer.echo("No faces found.")


@app.command()
def identify(
    refs: Path = typer.Option(
        ..., exists=True, file_okay=False, dir_okay=True,
        help="Refs folder with face crop subfolders to identify.",
    ),
) -> None:
    """Interactively name face clusters. Opens an HTML page and prompts in the terminal."""
    _setup_logging()

    from .identify import generate_html, run_identify_prompts

    html_path = generate_html(refs, refs / ".identify.html")
    webbrowser.open(html_path.as_uri())

    stats = run_identify_prompts(refs)
    typer.echo(f"\nDone! {stats['identified']} identified, {stats['skipped']} skipped, {stats['deleted']} deleted.")
    typer.echo(f"Next: waldo extract --input <media> --refs {refs} --output <output>")


@app.command()
def extract(
    input: Path = typer.Option(
        ..., exists=True, file_okay=False, dir_okay=True,
        help="Folder of media to process.",
    ),
    refs: Path = typer.Option(
        ..., exists=True, file_okay=False, dir_okay=True,
        help="Refs folder with named subfolders (one per person).",
    ),
    output: Path = typer.Option(
        ..., file_okay=False, dir_okay=True,
        help="Where to write clips and photos.",
    ),
    fps: float = typer.Option(2.0, help="Frames per second to sample."),
    threshold: float = typer.Option(0.5, help="Match similarity threshold (0-1)."),
    gap: float = typer.Option(2.0, help="Bridge gaps shorter than this (seconds)."),
    min_len: float = typer.Option(0.5, help="Minimum appearance duration (seconds)."),
    pad: float = typer.Option(15.0, help="Padding around appearances (seconds)."),
    workers: int = typer.Option(1, help="Concurrent files to process."),
) -> None:
    """Match faces against references and cut clips / copy photos."""
    _setup_logging()
    cfg = Config(
        input_dir=input, refs_dir=refs, output_dir=output,
        fps=fps, threshold=threshold, gap=gap, min_len=min_len, pad=pad, workers=workers,
    )
    output.mkdir(parents=True, exist_ok=True)

    from .pipeline import extract_clips
    extract_clips(cfg)


@app.command()
def run(
    input: Path = typer.Option(
        ..., exists=True, file_okay=False, dir_okay=True,
        help="Folder of media to process.",
    ),
    refs: Path = typer.Option(
        ..., file_okay=False, dir_okay=True,
        help="Refs folder — will be created by scan, used by extract.",
    ),
    output: Path = typer.Option(
        ..., file_okay=False, dir_okay=True,
        help="Where to write clips and photos.",
    ),
    fps: float = typer.Option(2.0, help="Frames per second to sample."),
    threshold: float = typer.Option(0.5, help="Similarity threshold (0-1)."),
    gap: float = typer.Option(2.0, help="Bridge gaps shorter than this (seconds)."),
    min_len: float = typer.Option(0.5, help="Minimum appearance duration (seconds)."),
    pad: float = typer.Option(15.0, help="Padding around appearances (seconds)."),
    workers: int = typer.Option(1, help="Concurrent files to process."),
    auto: bool = typer.Option(False, help="Skip interactive identification — use generic person_N names."),
) -> None:
    """Run the full pipeline: scan → identify → extract."""
    _setup_logging()
    cfg = Config(
        input_dir=input, refs_dir=refs, output_dir=output,
        fps=fps, threshold=threshold, gap=gap, min_len=min_len, pad=pad, workers=workers,
    )
    output.mkdir(parents=True, exist_ok=True)

    from .pipeline import run as run_pipeline
    run_pipeline(cfg, auto=auto)


if __name__ == "__main__":
    app()
```

- [ ] **Step 2: Verify CLI help works**

Run: `cd /Users/alberthurwitz/Projects/waldo && python -m waldo.cli --help`
Expected: Shows four commands: scan, identify, extract, run

- [ ] **Step 3: Run all tests**

Run: `cd /Users/alberthurwitz/Projects/waldo && python -m pytest -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add src/waldo/cli.py
git commit -m "feat: replace monolithic scan command with scan/identify/extract/run"
```

---

### Task 10: Integration smoke test

End-to-end test verifying scan → identify → extract flow works.

**Files:**
- Create: `tests/test_workflow.py`

- [ ] **Step 1: Write integration test**

Create `tests/test_workflow.py`:

```python
"""Smoke test for the scan → identify → extract workflow.

This test mocks the Embedder to avoid requiring the InsightFace model,
and verifies the folder-based interface between pipeline stages.
"""
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from waldo.config import Config
from waldo.embedder import Face


def _make_face(embedding: np.ndarray, bbox=(10, 10, 50, 50)) -> Face:
    return Face(bbox=bbox, embedding=embedding)


def _random_emb(seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v = rng.randn(512).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def media_dir(tmp_path):
    """Create a tiny video (10 frames of black) for testing."""
    d = tmp_path / "media"
    d.mkdir()
    video_path = d / "test.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 10.0, (64, 64))
    for _ in range(10):
        writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
    writer.release()
    return d


def test_scan_creates_refs_folder(media_dir, tmp_path):
    """waldo scan creates refs/person_N/ folders with face crops."""
    refs_dir = tmp_path / "refs"

    emb_a = _random_emb(42)
    emb_b = _random_emb(42)  # same seed = same person

    mock_embedder = MagicMock()
    mock_embedder.detect.return_value = [_make_face(emb_a)]

    cfg = Config(input_dir=media_dir, refs_dir=refs_dir, fps=2.0, threshold=0.5, workers=1)

    with patch("waldo.pipeline.Embedder", return_value=mock_embedder):
        from waldo.pipeline import scan_faces
        created = scan_faces(cfg)

    assert len(created) >= 1
    assert (refs_dir / "person_1").is_dir()
    assert len(list((refs_dir / "person_1").glob("*.jpg"))) > 0


def test_identify_renames_folders(tmp_path):
    """waldo identify renames person_N folders based on user input."""
    refs_dir = tmp_path / "refs"
    p1 = refs_dir / "person_1"
    p1.mkdir(parents=True)
    # Create a dummy face crop
    cv2.imwrite(str(p1 / "face_001.jpg"), np.zeros((64, 64, 3), dtype=np.uint8))

    from waldo.identify import run_identify_prompts

    with patch("builtins.input", return_value="grandma"):
        stats = run_identify_prompts(refs_dir)

    assert (refs_dir / "grandma").is_dir()
    assert not (refs_dir / "person_1").exists()
    assert stats["identified"] == 1
```

- [ ] **Step 2: Run integration tests**

Run: `cd /Users/alberthurwitz/Projects/waldo && python -m pytest tests/test_workflow.py -v`
Expected: ALL PASS

- [ ] **Step 3: Run full test suite**

Run: `cd /Users/alberthurwitz/Projects/waldo && python -m pytest -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_workflow.py
git commit -m "test: add integration smoke tests for scan → identify → extract workflow"
```

"""Smoke tests for the scan → identify → extract workflow."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from waldo.config import Config
from waldo.embedder import Face
from waldo.identify import run_identify_prompts


def _random_emb(seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v = rng.randn(512).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_face(embedding: np.ndarray, bbox=(10, 10, 50, 50)) -> Face:
    return Face(bbox=bbox, embedding=embedding)


@pytest.fixture
def media_dir(tmp_path):
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
    refs_dir = tmp_path / "refs"
    emb_a = _random_emb(42)

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
    refs_dir = tmp_path / "refs"
    p1 = refs_dir / "person_1"
    p1.mkdir(parents=True)
    cv2.imwrite(str(p1 / "face_001.jpg"), np.zeros((64, 64, 3), dtype=np.uint8))

    with patch("builtins.input", return_value="grandma"):
        stats = run_identify_prompts(refs_dir)

    assert (refs_dir / "grandma").is_dir()
    assert not (refs_dir / "person_1").exists()
    assert stats["identified"] == 1

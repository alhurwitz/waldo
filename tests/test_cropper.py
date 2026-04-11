import numpy as np
import pytest

from waldo.cropper import crop_face, save_crops


def test_crop_face_basic():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    frame[30:60, 80:120] = 255
    crop = crop_face(frame, (80.0, 30.0, 120.0, 60.0), pad_fraction=0.2)
    assert crop.shape[2] == 3
    assert crop.shape[0] > 0
    assert crop.shape[1] > 0


def test_crop_face_clamps_to_frame():
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    crop = crop_face(frame, (0.0, 0.0, 20.0, 20.0), pad_fraction=0.5)
    assert crop.shape[0] > 0
    assert crop.shape[1] > 0


def test_crop_face_target_size():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    crop = crop_face(frame, (50.0, 50.0, 150.0, 150.0), target_size=64)
    assert crop.shape[0] == 64
    assert crop.shape[1] == 64


def test_save_crops_creates_folders(tmp_path):
    frames = {
        "vid1_t0.5": np.zeros((100, 100, 3), dtype=np.uint8),
        "vid1_t1.0": np.zeros((100, 100, 3), dtype=np.uint8),
        "vid2_t0.0": np.zeros((100, 100, 3), dtype=np.uint8),
    }
    cluster_sources = [
        [("vid1_t0.5", (10, 10, 50, 50)), ("vid1_t1.0", (12, 12, 52, 52))],
        [("vid2_t0.0", (20, 20, 60, 60))],
    ]
    cluster_counts = [5, 3]

    refs_dir = tmp_path / "refs"
    save_crops(refs_dir, cluster_sources, cluster_counts, frames, max_per_cluster=5, min_count=2)

    assert (refs_dir / "person_1").is_dir()
    assert len(list((refs_dir / "person_1").glob("*.jpg"))) == 2
    assert (refs_dir / "person_2").is_dir()
    assert len(list((refs_dir / "person_2").glob("*.jpg"))) == 1


def test_save_crops_filters_small_clusters(tmp_path):
    frames = {"vid1_t0.5": np.zeros((100, 100, 3), dtype=np.uint8)}
    cluster_sources = [[("vid1_t0.5", (10, 10, 50, 50))]]
    cluster_counts = [1]

    refs_dir = tmp_path / "refs"
    save_crops(refs_dir, cluster_sources, cluster_counts, frames, max_per_cluster=5, min_count=2)
    assert not (refs_dir / "person_1").exists()


def test_save_crops_max_per_cluster(tmp_path):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frames = {f"v_t{i}": frame for i in range(10)}
    sources = [(f"v_t{i}", (10, 10, 50, 50)) for i in range(10)]
    cluster_sources = [sources]
    cluster_counts = [10]

    refs_dir = tmp_path / "refs"
    save_crops(refs_dir, cluster_sources, cluster_counts, frames, max_per_cluster=5, min_count=2)
    assert len(list((refs_dir / "person_1").glob("*.jpg"))) == 5

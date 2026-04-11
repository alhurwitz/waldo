from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from waldo.identify import generate_html, run_identify_prompts


def _make_refs(tmp_path: Path, names: list[str], faces_per: int = 2) -> Path:
    refs_dir = tmp_path / "refs"
    for name in names:
        d = refs_dir / name
        d.mkdir(parents=True)
        for i in range(faces_per):
            img = np.zeros((64, 64, 3), dtype=np.uint8)
            cv2.imwrite(str(d / f"face_{i + 1:03d}.jpg"), img)
    return refs_dir


def test_generate_html_contains_sections(tmp_path):
    refs_dir = _make_refs(tmp_path, ["person_1", "person_2"])
    html_path = generate_html(refs_dir, tmp_path / "identify.html")
    assert html_path.exists()
    content = html_path.read_text()
    assert "person_1" in content
    assert "person_2" in content


def test_generate_html_embeds_images(tmp_path):
    refs_dir = _make_refs(tmp_path, ["person_1"], faces_per=1)
    html_path = generate_html(refs_dir, tmp_path / "identify.html")
    content = html_path.read_text()
    assert "data:image/jpeg;base64," in content


def test_generate_html_empty_refs(tmp_path):
    refs_dir = tmp_path / "refs"
    refs_dir.mkdir()
    html_path = generate_html(refs_dir, tmp_path / "identify.html")
    content = html_path.read_text()
    assert "No clusters" in content


def test_identify_rename(tmp_path):
    refs_dir = _make_refs(tmp_path, ["person_1"])
    with patch("builtins.input", return_value="grandma"):
        stats = run_identify_prompts(refs_dir)
    assert not (refs_dir / "person_1").exists()
    assert (refs_dir / "grandma").is_dir()
    assert stats == {"identified": 1, "skipped": 0, "deleted": 0}


def test_identify_skip(tmp_path):
    refs_dir = _make_refs(tmp_path, ["person_1"])
    with patch("builtins.input", return_value="skip"):
        stats = run_identify_prompts(refs_dir)
    assert (refs_dir / "person_1").is_dir()
    assert stats == {"identified": 0, "skipped": 1, "deleted": 0}


def test_identify_delete(tmp_path):
    refs_dir = _make_refs(tmp_path, ["person_1"])
    with patch("builtins.input", return_value="delete"):
        stats = run_identify_prompts(refs_dir)
    assert not (refs_dir / "person_1").exists()
    assert stats == {"identified": 0, "skipped": 0, "deleted": 1}


def test_identify_merge_collision(tmp_path):
    refs_dir = _make_refs(tmp_path, ["grandma", "person_2"], faces_per=2)
    # Only person_2 will be prompted (grandma comes first alphabetically but is not person_N)
    # Actually both get prompted. Mock returns "grandma" for both.
    answers = iter(["skip", "grandma"])
    with patch("builtins.input", side_effect=answers):
        stats = run_identify_prompts(refs_dir)
    assert (refs_dir / "grandma").is_dir()
    assert not (refs_dir / "person_2").exists()
    assert len(list((refs_dir / "grandma").glob("*.jpg"))) == 4
    assert stats == {"identified": 1, "skipped": 1, "deleted": 0}

"""CLI surface tests for `waldo compile`. assemble() is mocked."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from waldo.cli import app

runner = CliRunner()


def _make_output_tree(root: Path, layout: dict[str, list[str]]) -> None:
    for folder, files in layout.items():
        d = root / folder
        d.mkdir(parents=True, exist_ok=True)
        for name in files:
            (d / name).write_bytes(b"")


def test_compile_persons_invokes_assemble(tmp_path):
    _make_output_tree(tmp_path, {
        "grandma": ["a.mp4", "b.mp4"],
        "dad": ["c.mp4"],
    })
    with patch("waldo.assembler.assemble") as mock_assemble:
        result = runner.invoke(app, [
            "compile", "--output", str(tmp_path),
            "--persons", "grandma,dad",
            "--transition", "cut",
        ])
    assert result.exit_code == 0, result.output
    mock_assemble.assert_called_once()
    args, kwargs = mock_assemble.call_args
    clips = args[0]
    assert len(clips) == 3
    assert {c.name for c in clips} == {"a.mp4", "b.mp4", "c.mp4"}
    out_path = args[1]
    assert out_path == tmp_path / "_compilations" / "grandma_dad.mp4"
    assert kwargs.get("transition", args[2] if len(args) > 2 else None) == "cut"


def test_compile_all_invokes_assemble(tmp_path):
    _make_output_tree(tmp_path, {
        "grandma": ["a.mp4"],
        "dad": ["b.mp4"],
        "together": ["t.mp4"],
        "_compilations": ["old.mp4"],
    })
    with patch("waldo.assembler.assemble") as mock_assemble:
        result = runner.invoke(app, [
            "compile", "--output", str(tmp_path), "--all",
        ])
    assert result.exit_code == 0, result.output
    args, _ = mock_assemble.call_args
    clips = args[0]
    assert {c.name for c in clips} == {"a.mp4", "b.mp4"}
    out_path = args[1]
    assert out_path == tmp_path / "_compilations" / "all.mp4"


def test_compile_persons_and_all_errors(tmp_path):
    _make_output_tree(tmp_path, {"grandma": ["a.mp4"]})
    result = runner.invoke(app, [
        "compile", "--output", str(tmp_path),
        "--persons", "grandma", "--all",
    ])
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output.lower() or "cannot" in result.output.lower()


def test_compile_neither_persons_nor_all_errors(tmp_path):
    result = runner.invoke(app, ["compile", "--output", str(tmp_path)])
    assert result.exit_code != 0


def test_compile_no_clips_found_exits_nonzero(tmp_path):
    (tmp_path / "ghost").mkdir()  # exists but empty
    with patch("waldo.assembler.assemble") as mock_assemble:
        result = runner.invoke(app, [
            "compile", "--output", str(tmp_path),
            "--persons", "ghost",
        ])
    assert result.exit_code != 0
    mock_assemble.assert_not_called()

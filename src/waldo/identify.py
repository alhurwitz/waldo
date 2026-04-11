from __future__ import annotations

import base64
import logging
import shutil
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


def run_identify_prompts(refs_dir: Path) -> dict[str, int]:
    """Prompt the user to name each cluster folder.

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

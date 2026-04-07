# Face Finder — Design

## Goal
Given a folder of long MP4 videos and reference photos of two people (grandma, grandpa), find every scene where they appear and save those scenes as individual clips on disk, organized by person.

## Inputs
- `videos/` — flat folder of `.mp4` files.
- `refs/grandma/`, `refs/grandpa/` — each containing one or more reference photos (JPG/PNG). Multiple photos per person improve matching accuracy across angles, lighting, and age.

## Outputs
```
output/
  grandma/  <video-stem>_<HH-MM-SS>_<HH-MM-SS>.mp4
  grandpa/  <video-stem>_<HH-MM-SS>_<HH-MM-SS>.mp4
  both/     <video-stem>_<HH-MM-SS>_<HH-MM-SS>.mp4
```
- A scene where only grandma appears → `grandma/`.
- A scene where only grandpa appears → `grandpa/`.
- A scene where both appear → `both/` (single copy, not duplicated).

## Architecture
Single Python CLI tool, processing one video at a time. Pipeline:

```
videos/*.mp4  ─┐
               ├─►  [1] Sample frames  ─►  [2] Detect+embed faces (InsightFace)
refs/grandma/  │                                        │
refs/grandpa/  ┘                                        ▼
                                          [3] Match against references
                                                        │
                                                        ▼
                                          [4] Build appearance timeline
                                                        │
                                                        ▼
                                          [5] Snap to scene boundaries (PySceneDetect)
                                                        │
                                                        ▼
                                          [6] Cut clips with ffmpeg → output/{grandma,grandpa,both}/
```

## Components

### 1. Frame sampler
Decodes the video at a configurable rate (default **2 fps**). Faces persist across frames so per-frame processing is unnecessary.

### 2. Face detector + embedder
Uses **InsightFace `buffalo_l`** via ONNX Runtime with the **CoreML execution provider** for Apple Silicon acceleration. For each sampled frame, returns a list of (bounding box, 512-d embedding).

### 3. Reference matcher
On startup, computes embeddings for every reference photo in `refs/grandma/` and `refs/grandpa/`. For each detected face in a frame, computes cosine similarity to the nearest reference embedding for each person; assigns to the person whose nearest reference exceeds a threshold (default **0.5**, tunable). If neither passes, the face is ignored.

### 4. Timeline builder
For each person, produces a list of (timestamp, matched_bool) from sampled frames, then:
- Merges consecutive matched samples into intervals.
- Bridges gaps shorter than **2s** (one continuous appearance).
- Drops blips shorter than **0.5s** (false positives).

Result: per-person list of intervals `[(start_s, end_s), …]`.

### 5. Scene snapper
Runs **PySceneDetect** once per video to obtain scene boundaries. Each appearance interval is expanded to cover all scenes it overlaps. After expansion, overlapping intervals (across persons) are unioned into a single set of "clip scenes," each tagged with which persons appear in it (grandma, grandpa, or both).

### 6. Clip cutter
For each clip scene, runs `ffmpeg -ss <start> -to <end> -i <video> -c copy <out>` (stream copy → fast, lossless, no re-encode). Routes the file to `grandma/`, `grandpa/`, or `both/` based on the scene's tag.

## Data flow & caching
Per-video sidecar JSON cache stores:
- Scene boundaries from PySceneDetect.
- Sampled-frame face embeddings + timestamps.

This makes re-runs after threshold/parameter tweaks fast — only steps 3–6 re-execute.

Cache path: `<videos>/.face-finder-cache/<video-stem>.json` (or similar).

## CLI
```
face-finder run \
  --videos ./videos \
  --refs ./refs \
  --output ./output \
  --fps 2 \
  --threshold 0.5
```
Defaults are reasonable; only `--videos`, `--refs`, `--output` are required.

## Error handling
- Unreadable / corrupt videos: log warning, skip, continue with next video.
- Reference photo with 0 detected faces: warn and ignore that photo.
- Reference photo with >1 detected face: warn (ambiguous) and ignore that photo.
- A person with zero usable reference photos after the above: hard error before processing starts.
- Ctrl+C is safe: caches are written incrementally so a re-run resumes cleanly.

## Testing
- **Unit tests** on pure functions:
  - Timeline merging (gap bridging, blip dropping).
  - Scene snapping (interval-to-scene expansion, multi-person tagging).
- **Smoke test**: a small sample video with known appearances of two known faces, verifying clips land in the right folders with roughly the right timestamps.

## Tech stack
- Python (project already uses `pyproject.toml` + `uv`).
- `insightface` + `onnxruntime` (with CoreML provider).
- `scenedetect` (PySceneDetect).
- `opencv-python` for frame decoding.
- `ffmpeg` (system binary) for cutting.
- `typer` or `click` for the CLI.
- `pytest` for tests.

## Out of scope
- Identifying anyone other than the two reference people.
- Re-encoding / transcoding clips.
- A GUI.
- Cloud / remote execution.

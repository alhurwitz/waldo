# Clustering & Identify Workflow — Design

## Goal

Split the waldo pipeline into discrete CLI commands so users can cluster faces, interactively name them, and then extract clips — with a folder-based interface between steps.

## CLI Commands

### `waldo scan`

Clusters all faces found in input media and saves representative face crops to a refs folder.

```
waldo scan --input ./videos --refs ./refs
```

**Behavior:**
1. Sample frames from all videos (reuses Sampler)
2. Detect and embed faces (reuses Embedder)
3. Cluster faces (reuses FaceCluster)
4. For each cluster with >= `min_count` faces, save up to 5 representative face crops as JPEGs to `refs/person_N/face_001.jpg`, etc.
5. Stop. Print summary and suggest next command.

**Face crops:** Extracted from the original frame using the bounding box with padding (~20% on each side) so the crop shows hair/neck/context, not just the face. Saved as JPEG, roughly 200x200px (scaled from bbox).

**Options:**
- `--input` (required): Media directory
- `--refs` (required): Where to write the refs folder
- `--fps` (default 2.0): Frame sampling rate
- `--threshold` (default 0.5): Clustering similarity threshold
- `--workers` (default 1): Concurrent file processing

**Edge cases:**
- No faces found: warn "No faces detected in any media", create empty refs folder
- Refs folder already exists: prompt "Refs folder already exists. Overwrite? (y/n)"

### `waldo identify`

Interactive command that helps users name the clustered face folders.

```
waldo identify --refs ./refs
```

**Behavior:**
1. Generate an HTML page with one section per cluster folder, showing all face crop images in a grid
2. Open the HTML page in the default browser
3. Prompt in the terminal, one cluster at a time:
   ```
   Who is person_1? (name / skip / delete): grandma
   Who is person_2? (name / skip / delete): grandpa
   Who is person_3? (name / skip / delete): delete
   ```
4. Actions:
   - **Type a name** → rename folder from `person_N/` to `<name>/`
   - **Type `skip`** → leave folder as-is
   - **Type `delete`** → remove the folder entirely
5. **Collision handling:** If user types a name that already exists as a folder, merge the face crops into the existing folder (two clusters were actually the same person)
6. Print summary: `Done! 2 identified, 0 skipped, 1 deleted.`
7. Print suggested next command: `Run: waldo extract --input ./videos --refs ./refs --output ./output`

**Options:**
- `--refs` (required): Refs folder to identify

**Edge cases:**
- All clusters deleted: warn "No clusters remaining — nothing to extract"
- Re-running identify: works — re-opens HTML with current folders, prompts for all folders

### `waldo extract`

Matches faces against named references and cuts clips. This is the current pipeline from step 3 onward.

```
waldo extract --input ./videos --refs ./refs --output ./output
```

**Behavior:**
1. Load reference embeddings from refs folder (reuses `load_references()`)
2. For each video: classify faces, build timelines, snap to scenes, cut clips
3. For each photo: classify faces, copy to output
4. Route output to `output/<person>/` and `output/together/`

**Options:**
- `--input` (required): Media directory
- `--refs` (required): Refs folder with named subfolders
- `--output` (required): Where to write clips
- `--threshold` (default 0.5): Match similarity threshold
- `--gap` (default 2.0): Gap bridging in seconds
- `--min-len` (default 0.5): Minimum appearance duration
- `--pad` (default 15.0): Padding around appearances in seconds
- `--workers` (default 1): Concurrent file processing
- `--fps` (default 2.0): Frame sampling rate (for cache lookup)

**Edge cases:**
- Unnamed clusters (person_N): works fine — clips go to `output/person_1/`, etc.
- Mixed named/unnamed: fine — `output/grandma/`, `output/person_2/` coexist

### `waldo run`

Runs the full pipeline: scan → identify → extract.

```
waldo run --input ./videos --refs ./refs --output ./output
waldo run --input ./videos --refs ./refs --output ./output --auto
```

**Behavior:**
- Without `--auto`: runs scan, then identify (interactive), then extract
- With `--auto`: runs scan, skips identify, runs extract with generic `person_N` names

**Options:** Union of all options from scan + extract, plus `--auto`.

## Refs Folder Structure

After `scan`:
```
refs/
  person_1/
    face_001.jpg
    face_002.jpg
    face_003.jpg
  person_2/
    face_001.jpg
    face_002.jpg
```

After `identify` (or manual renaming):
```
refs/
  grandma/
    face_001.jpg
    face_002.jpg
    face_003.jpg
  grandpa/
    face_001.jpg
    face_002.jpg
```

The refs folder IS the interface between steps. No database, no state files — just folders and images. Users can also rename folders manually in Finder/terminal instead of using `waldo identify`.

## HTML Page for Identify

Simple self-contained HTML file (no external dependencies):
- One section per cluster, with the folder name as heading
- Grid of face crop images (inline base64 or file:// references)
- Serves as a visual reference while the user answers terminal prompts

Generated as a temp file and opened with `webbrowser.open()`.

## Changes to Existing Code

### What stays the same
- Embedder, Sampler, Matcher, Timeline, Snapper, Cutter, Cache — no changes
- `load_references()` — no changes (already reads `refs/<name>/<photo>` structure)

### What changes
- **`pipeline.py`**: Split `run()` into `scan()`, `extract()`, and `run()`. Move `_auto_cluster` logic into `scan()` with added face-crop saving.
- **`cli.py`**: Add `scan`, `identify`, `extract`, `run` commands. Remove or deprecate old `scan` command.
- **`config.py`**: May need separate config models per command, or make fields optional based on which command is running.

### New code
- **Face crop extraction**: Function to crop a face from a frame using bbox + padding, resize, save as JPEG
- **HTML generation**: Function to generate the identify HTML page from a refs folder
- **Identify prompts**: Terminal input loop with rename/skip/delete logic
- **`identify` command**: Orchestrates HTML generation + prompts

## Testing

- **Face crop extraction**: Unit test — given a frame and bbox, verify crop dimensions and padding
- **HTML generation**: Unit test — given a refs folder, verify HTML contains expected sections and images
- **Identify flow**: Integration test — create temp refs folder, simulate input, verify renames/deletes
- **Command integration**: Smoke test — scan → identify → extract on sample data

# Compile: Auto-Assembled Compilation Videos with Transitions

## Goal

Add a `waldo compile` subcommand that stitches per-person clips produced by
`waldo extract` into a single compilation video, with configurable transitions
between clips.

## User-facing surface

### New subcommand: `waldo compile`

```
waldo compile \
  --output ./out \
  --persons grandma,dad \
  --transition crossfade
```

Flags:

- `--output PATH` (required) — the directory passed to a previous
  `waldo extract`. Must contain per-person subfolders with `.mp4` clips.
- `--persons LIST` — comma-separated list of person names. Each name must
  correspond to a subfolder under `--output`. Mutually exclusive with `--all`.
- `--all` — compile every person folder under `--output`, excluding
  `together/` and `_compilations/`. Mutually exclusive with `--persons`.
- `--transition {crossfade,fade,cut,random}` — defaults to `crossfade`.
  - `crossfade`: video dissolves between clips; audio crossfaded.
  - `fade`: fade-through-black between clips; audio crossfaded.
  - `cut`: hard cuts, no transition.
  - `random`: each transition picks randomly from a fixed set of xfade
    types (`fade`, `fadeblack`, `slideleft`, `slideright`, `wipeleft`,
    `wiperight`, `dissolve`); audio is always crossfaded.

Behavior:

- Clip order is shuffled before assembly.
- Original audio is preserved from each clip (no music bed in v1).
- Transition duration is fixed at 0.5s in v1.
- Output path: `<output>/_compilations/<persons>.mp4`, where `<persons>`
  is `_`-joined names (`grandma_dad.mp4`) or `all.mp4` when `--all` is set.
- Output is overwritten on re-run.

### New flags on `waldo run`

- `--compile` (boolean, default `False`) — when set, after `extract`
  finishes, automatically invoke compile with `--all` using the value of
  `--transition` (added to `run` as well, default `crossfade`).

## Design

### Module layout

New module: `src/waldo/assembler.py` (matches existing agentive `-er`
naming used by `cutter`, `cropper`, `embedder`, `sampler`, `snapper`).

Public API:

```python
def discover_clips(output_dir: Path, persons: list[str]) -> list[Path]:
    """Return all .mp4 clips under output_dir/<person>/ for each person.

    Excludes together/ and _compilations/. Skips persons with no folder
    or an empty folder, with a warning.
    """

def assemble(
    clips: list[Path],
    out: Path,
    transition: str,
    transition_duration: float = 0.5,
) -> Path:
    """Stitch clips into out using a single ffmpeg filter_complex call.

    Shuffles clips, probes their durations, builds a filter graph, and
    runs ffmpeg once. Always re-encodes (libx264 + aac).

    For a single clip, copies it to out without invoking ffmpeg.
    """
```

A pure helper `_build_filter_complex(durations, transition, duration) ->
str` is extracted for unit testability.

### Data flow

1. CLI parses `--persons` / `--all` into a `list[str]` of person names.
2. `discover_clips(output_dir, persons)` returns a sorted flat
   `list[Path]` of `.mp4` files (sorted only for deterministic ordering
   prior to shuffling).
3. `assemble()`:
   1. `random.shuffle(clips)`.
   2. `ffprobe` each clip for its actual duration. (Clips were cut with
      `-ss/-to`, but real duration depends on keyframe alignment — we
      need accurate durations for filter offsets.)
   3. Build the `filter_complex` string per the chosen transition (see
      below).
   4. Single ffmpeg invocation, output to a temp path inside the
      destination directory, then atomic-rename to the final path.
4. Return the output path; CLI prints it.

### Filter graph

Let N be the number of clips, d_i be the duration of clip i, D be the
transition duration.

**`crossfade`** (xfade `transition=fade`):

```
[0:v][1:v]xfade=transition=fade:duration=D:offset=(d_0 - D)[v01];
[v01][2:v]xfade=transition=fade:duration=D:offset=(d_0 + d_1 - 2*D)[v02];
...
[0:a][1:a]acrossfade=d=D[a01];
[a01][2:a]acrossfade=d=D[a02];
...
```

The k-th offset is `sum(d_0..d_{k-1}) - k*D`, so each xfade lines up
with the trailing tail of the running output.

**`fade`**: same as crossfade, but `transition=fadeblack`.

**`random`**: same chained shape, each xfade picks a random `transition`
from the set listed above. Audio still uses `acrossfade`.

**`cut`**: single concat filter, no transition math.

```
[0:v][0:a][1:v][1:a]...[N-1:v][N-1:a]concat=n=N:v=1:a=1[v][a]
```

### ffmpeg invocation

Output map: `-map "[v]" -map "[a]"`. Codecs: `-c:v libx264 -preset medium
-crf 20 -c:a aac -b:a 192k -movflags +faststart`. Re-encoding is
unconditional — ensures compilations work even when source clips have
mismatched codecs/resolutions/sample rates from different cameras.

We reuse `cutter.check_ffmpeg()` for the binary check, and
`cutter._safe_path()` for path quoting.

### Discovery rules

`discover_clips`:

- For each `name` in `persons`: include every `.mp4` directly under
  `output_dir / name`. Subdirectories are not recursed (extract writes
  flat folders).
- Skip the special folders `together/` and `_compilations/`.
- Warn and skip persons whose folder is missing or empty.
- Return paths sorted alphabetically (deterministic; assembler shuffles).

`--all` discovery: every immediate subdirectory of `output_dir` whose
name is not `together` and does not start with `_` (which excludes
`_compilations/` and any other underscore-prefixed scratch folder).

### Limitation: `together/` clips

`waldo extract` writes co-appearance clips to `together/` and discards
the per-clip mapping back to which specific people are in each clip. As
a result, v1 cannot include `together/` clips in a per-person reel
without re-running embeddings on those clips. This is documented in the
CLI help text. A future iteration could either:

- Re-embed `together/` clips on demand inside `compile`, or
- Have `extract` emit a sidecar manifest naming the people in each
  `together/` clip.

Out of scope for v1.

## Error handling

| Case | Behavior |
|---|---|
| `--persons NAME` but `<output>/NAME/` doesn't exist | Warn, skip that name. |
| Person folder exists but is empty | Warn, skip that name. |
| All requested persons skipped (no clips found anywhere) | Exit non-zero with a clear message; no output written. |
| Only one clip total after discovery | Copy it to the output path; no ffmpeg call. |
| `ffprobe` fails on a clip (corrupt file) | Warn, drop that clip, continue. |
| `ffmpeg` invocation fails | Surface stderr; let `CalledProcessError` propagate; CLI exits non-zero. |
| `<output>/_compilations/<persons>.mp4` already exists | Overwrite. |
| `ffmpeg` not on PATH | Reuse `cutter.check_ffmpeg()` (exits with the existing error message). |
| `--persons` and `--all` both passed | Typer error before any work runs. |
| Neither `--persons` nor `--all` passed | Typer error before any work runs. |

## Testing

Three layers, mirroring how `cutter`/`snapper`/`timeline` are tested
today.

### Unit tests — `tests/test_assembler.py` (no ffmpeg required)

- `discover_clips()` against a fixture tree with `grandma/`, `dad/`,
  `together/`, `_compilations/` and known `.mp4` files; assert the
  returned list excludes the special folders and includes the
  requested persons.
- `--all` discovery against the same fixture; assert it picks up
  every person folder while excluding the special ones and any
  underscore-prefixed folder.
- `_build_filter_complex(durations, transition, duration)` for each
  transition mode:
  - 2 clips, `crossfade` → expected single `xfade` + single
    `acrossfade` snippet.
  - 3 clips, `crossfade` → chained offsets `d0-D`, `d0+d1-2D`.
  - 1 clip → empty graph (caller takes the `cp` shortcut).
  - 4 clips, `cut` → `concat=n=4:v=1:a=1[v][a]`.
- CLI: `--persons` and `--all` together → typer error; neither →
  typer error.

### Integration tests — `tests/test_assembler_integration.py`

Skip the whole module when `shutil.which("ffmpeg")` is None (matches
existing pattern in the codebase).

- Synthesize 3 short clips with `ffmpeg -f lavfi` (color source + sine
  audio).
- Run `assemble()` for each of `crossfade`, `fade`, `cut`, `random`.
- Assert: output exists and is non-empty; `ffprobe` reports duration
  ≈ `sum(d_i) − (N−1) × D` for crossfade/fade/random, or ≈ `sum(d_i)`
  for cut; output has both video and audio streams.

### End-to-end test — extend `tests/test_workflow.py`

After the existing `extract` step completes, invoke `compile --all
--transition cut` and verify `_compilations/all.mp4` exists. `cut`
is the fastest mode — keeps the workflow test short.

## Out of scope for v1

- Length controls (`--max-clip`, `--max-total`, total duration cap).
  Add later once we've watched real compilations and felt the need.
- Music bed (`--music`). Adds ducking and length-mismatch design
  questions; separate spec.
- Configurable transition duration (`--transition-duration`). Fixed
  at 0.5s.
- Per-clip transition selection (`--transition` is global to the run).
- Including `together/` clips (see Limitation above).
- Reproducible shuffle (`--seed`).

# Using YOLOE to Analyze an Assisted-Lane-Change Cluster Video

A practical, end-to-end guide for turning the Audi *assistierter Spurwechsel*
promo clip into a structured YOLOE test set — download, frame extraction,
region-of-interest (ROI) analysis, carpet (driving-area) detection, and
per-frame pseudo-labels.

> **Background:** [07_YOLOE_CONCEPTS.md](07_YOLOE_CONCEPTS.md) is the conceptual YOLOE learning guide
> (architecture, prompting modes, diagrams, citations). This document is the
> hands-on, example-specific pipeline built on top of it.
> Clip: <https://www.youtube.com/watch?v=0Ol8MA9e8nM>

---

## 1. Goal & use case

The clip is an Audi *Virtual Cockpit* visualization of the **adaptive driving
assistant's assisted lane change**. From the transcript, the relevant states are:

- The assisted lane change is offered only when the route and traffic allow it.
  When available, **two arrows next to a green symbol** appear on the side of
  the cockpit.
- The car **aborts** the lane change if another vehicle blocks the target lane,
  a faster vehicle approaches from behind, or no hands are detected on the wheel.

In the rendered 3D scene this maps to a small set of **visual states** we want to
detect and label automatically:

| # | State | Visual cue |
|---|-------|-----------|
| 1 | Ego centered, no neighbors | white ego car, no carpet |
| 2 | Vehicle left / right | dark render in the side lane |
| 3 | Lane change **available** | **green** carpet / green arrows beside ego |
| 4 | Lane change **blocked/aborted** | **red** carpet on a side |
| 5 | Menu / intro / warning overlay | not the driving scene (to be filtered out) |

The pipeline below produces, per frame: vehicle counts per ROI (left / ego /
right) and the carpet colour per side — i.e. exactly these states as a CSV.

---

## 2. YOLOE in one minute

YOLOE ("Real-Time Seeing Anything") is an open-vocabulary YOLO. Instead of a
fixed 80-class list, the detection head scores **similarity to a prompt
embedding**, so the vocabulary is swappable at inference with zero retraining.
Three prompting modes (see [07_YOLOE_CONCEPTS.md §4](07_YOLOE_CONCEPTS.md#4-the-three-prompting-modes)):

- **Text prompts** — `model.set_classes(["car", ...])`. Most flexible; used here.
- **Visual prompts** — detect by a reference image/box (good for hard-to-name
  things like the green lane-change arrow symbol).
- **Prompt-free** (`*-pf.pt`) — enumerates a built-in 4,585-class vocabulary;
  good for exploratory "what's in this frame" labeling.

For this example we use **text prompts** (`["car"]`) to find the rendered
vehicles, and classic **HSV colour thresholding** for the green/red carpet — the
carpet is a flat synthetic overlay, so colour is far more reliable than a learned
detector.

---

## 3. Environment

This repo already ships a virtual environment with `ultralytics`, `supervision`,
`opencv`, and `pyyaml`. External tools used by the download step:

| Tool | Location / install | Purpose |
|------|--------------------|---------|
| venv Python | `.venv\Scripts\python.exe` | runs the YOLOE scripts |
| `yt-dlp` | `pip install -U yt-dlp` (global) | download the clip |
| `ffmpeg` | `C:\ffmpeg\bin\ffmpeg.exe` | merge HD video+audio streams |

> Always run the YOLOE scripts with the **venv** interpreter — the global Python
> does not have `ultralytics`/`supervision`:
>
> ```powershell
> adascope detect ...
> ```

---

## 4. Pipeline

```text
YouTube clip
  → 4.1 download (yt-dlp + ffmpeg)          → data/raw/cluster_video.mp4
  → 4.2 extract frames (OpenCV)             → data/frames/raw/*.jpg
  → 4.3 ROI + carpet debug (YOLOE)          → outputs/roi_debug/*.jpg
  → 4.4 per-frame states (pseudo-labels)    → outputs/roi_debug/states.csv
```

### 4.1 Download the video (HD)

Without a JS runtime YouTube only offered a single progressive 360p stream.
With **ffmpeg** available, `yt-dlp` can fetch separate HD video + audio streams
and merge them, giving a clean **1920×1080 / 25 fps** file:

```powershell
python -m yt_dlp --ffmpeg-location "C:\ffmpeg\bin" `
  -f "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b" --merge-output-format mp4 `
  --no-playlist -o "data/raw/cluster_video.%(ext)s" `
  "https://www.youtube.com/watch?v=0Ol8MA9e8nM"
```

List available formats first with `python -m yt_dlp --ffmpeg-location "C:\ffmpeg\bin" -F <url>`.

### 4.2 Extract frames

[`adascope extract`](../adascope/cli/extract.py) samples frames at a
target rate using OpenCV only (no ffmpeg dependency). At `--fps 2` the ~2:12 clip
yields **276 frames** — a good size for a first test set (100–300).

```powershell
adascope extract --video data\raw\cluster_video.mp4 --fps 2 --out data\frames\raw
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--video` | — | input video (required) |
| `--out` | `data/frames/raw` | output directory |
| `--fps` | `2.0` | frames sampled per second |
| `--max-frames` | `0` | stop after N frames (0 = no limit) |
| `--quality` | `95` | JPEG quality |

### 4.3 ROI + carpet debug

[`adascope detect`](../adascope/analysis.py) overlays the three ROIs, runs
YOLOE to find vehicles (assigning each detection to a region by its box center),
and detects the green/red carpet per side via HSV thresholds. It writes annotated
frames to `outputs/roi_debug/`.

```powershell
# single frame (prints the detected state)
adascope detect --frame data\frames\raw\frame_00200.jpg

# whole folder
adascope detect --frames data\frames\raw
```

Each annotated frame shows: the **left / ego / right** ROI rectangles with their
vehicle counts, a coloured box per detected vehicle, and a `carpet left/right`
banner.

### 4.4 Per-frame pseudo-labels (CSV)

Add `--csv` to dump one row per frame — this *is* your state table / pseudo-label
set, queryable without any manual labeling:

```powershell
adascope detect `
  --frames data\frames\raw --csv outputs\roi_debug\states.csv
```

Columns: `frame, veh_left, veh_ego, veh_right, state_left, state_ego, state_right`.
`veh_*` count **other** vehicles per lane (the ego car is excluded via `ego_box`);
`state_*` is the semantic drivable state — `available` (green) / `blocked` (red) on
the sides, `drivable` (white path) on ego, else `clear`.

### 4.5 Full debug video (complete detection)

While 4.2–4.4 work on a *sampled* (2 fps) frame set, the `--video` mode runs the
**same analysis on every frame** of the clip and writes an annotated MP4 at native
frame rate — useful for watching the detector behave through the whole sequence.

```powershell
adascope detect `
  --video data\raw\cluster_video.mp4 `
  --out-video outputs\roi_debug\debug.mp4 `
  --csv outputs\roi_debug\states_fullvideo.csv
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--video` | — | input video (mutually exclusive with `--frame`/`--frames`) |
| `--out-video` | `outputs/roi_debug/debug.mp4` | annotated output video |
| `--stride` | `1` | process every Nth frame (output fps is divided by the stride) |
| `--csv` | — | also dump the per-frame state table |

> **Runtime note.** On CPU (`torch.cuda.is_available() == False` here) YOLOE-11-L
> runs at roughly 2 frames/s, so the full ~3,300-frame clip takes ~20–25 min. Use
> `--stride 5` for a quick preview, or run the full job in the background.
> Because the whole clip is processed (not just the driving scene), the intro,
> menu, and map frames appear too — their ROIs are simply empty (`carpet none`).

---

## 5. Configuration (`config.yaml`)

All tunable values live in [configs/detection.yaml](../configs/detection.yaml) — no code edits needed,
and the file itself is fully commented. ROIs are **perspective polygons** (lists
of `[x, y]` points) that follow the lanes; every coordinate is a **fraction** of
the frame (0–1), so it is resolution-independent. Colours are BGR; HSV ranges use
the OpenCV convention (H 0–179, S/V 0–255).

| Key | Meaning |
|-----|---------|
| `model.checkpoint` | YOLOE weights file (e.g. `yoloe-11l-seg.pt`, or a `-pf` / `-26l` variant). |
| `model.classes` | Text prompts for what counts as a vehicle, e.g. `[car, truck]`. |
| `model.conf` | Detection confidence 0–1 (lower = more boxes, more false positives). |
| `rois.<name>` | A lane polygon: list of `[x, y]` fraction points. Top points near the vanishing point, bottom points at the wide near edge. A vehicle is assigned to the first polygon containing its box centre (`left` → `ego` → `right`). |
| `roi_colors.<name>` | BGR outline/box colour for that ROI. |
| `carpet.detect_in` | Which ROIs to test for a carpet colour (default `[left, right]`). |
| `carpet.min_frac` | Min green/red pixel fraction inside a polygon to count as a carpet. |
| `carpet.white_min_frac` | Higher threshold for white (lane markings are also white); green/red win over white when both pass. |
| `carpet.green_hsv` / `red_hsv` / `white_hsv` | HSV gates `[[Hmin,Smin,Vmin],[Hmax,Smax,Vmax]]`. Red needs two ranges (it wraps the hue circle). |

```yaml
rois:                                   # perspective lane polygons
  left:  [[0.41, 0.20], [0.45, 0.20], [0.44, 0.45], [0.27, 0.45]]
  ego:   [[0.455, 0.32], [0.555, 0.32], [0.575, 0.47], [0.435, 0.47]]
  right: [[0.47, 0.20], [0.51, 0.20], [0.71, 0.45], [0.54, 0.45]]

carpet:
  detect_in: [left, right]
  min_frac: 0.04
  white_min_frac: 0.18
  green_hsv: [[38, 70, 70], [90, 255, 255]]
  red_hsv: [[[0, 90, 80], [12, 255, 255]], [[168, 90, 80], [180, 255, 255]]]
  white_hsv: [[0, 0, 95], [179, 45, 170]]
```

**CLI overrides** apply on top of the config for quick experiments:

```powershell
# try a wider prompt set and a lower threshold without editing the file
adascope detect --frames data\frames\raw --classes car vehicle --conf 0.05

# point at an alternative tuning file
adascope detect --frames data\frames\raw --config my_config.yaml
```

---

## 6. Results on this clip

Running over all 276 frames at `conf=0.1`, prompt `["car"]`:

| Metric | Frames |
|--------|-------:|
| Total extracted | 276 |
| Ego vehicle detected | 158 |
| Carpet present (lane-change zone active) | 57 |
| — left green / left red | 17 / 4 |
| — right green / right red | 31 / 7 |
| Menu / intro / empty (nothing detected) | 115 |

**Key finding — UI stability.** The clip is *not* a single stable view: ~115
frames are the Audi logo intro, the central MMI menu, or title/warning overlays.
But the 3D driving scene itself is **very stable**, and on those frames the fixed
ROIs + colour detection work well:

- The white **ego car** is reliably detected and lands in the ego ROI.
- The **green** (available) vs **red** (blocked/aborted) carpet is cleanly
  separated per side — directly encoding states 3 and 4 from §1.

The takeaway from the plan is confirmed: **filter to the driving-scene frames**
(e.g. keep rows where `veh_ego > 0` or any carpet is present → ~160 frames) to get
a clean cluster test set, and discard the menu/intro frames.

---

## 7. Tuning guide

| Symptom | Fix in `config.yaml` |
|---------|----------------------|
| Lane ROI doesn't follow the lane | nudge that polygon's 4 corner points in `rois.<name>` |
| Vehicle assigned to the wrong lane | adjust where the polygons meet near the ego |
| HUD clock/temp counted as a vehicle | pull the `right` polygon's top points left / down |
| Faint carpet missed / flicker | lower `carpet.min_frac`, widen `green_hsv`/`red_hsv` |
| White carpet missed or over-triggers | tune `carpet.white_hsv` + `carpet.white_min_frac` |
| Lead vehicles up the road missed | lower `model.conf`, or add prompts: `classes: [car, vehicle]` |
| Different clip resolution | nothing — polygons are fractions, already resolution-independent |

---

## 8. Limitations & next steps

- **Small / dark lead vehicles.** The `car` prompt at `conf=0.1` misses the
  faint renders far up the road. Try `--classes car vehicle "white car"` and/or
  `--conf 0.05`, or the prompt-free checkpoint
  ([yoloe-11l-seg-pf.pt](../models/yoloe-11l-seg-pf.pt)) for exploratory labeling.
- **The green "lane-change available" arrow symbol** is hard to name in text.
  This is the textbook case for a **visual prompt**: crop one frame where the
  arrows show, mark them with a box, and let SAVPE match them across the clip.
- **Production accuracy.** For a deployed detector on this synthetic UI, a short
  fine-tuning pass on the filtered frames (using these CSV states as a starting
  point) will outperform zero-shot prompting.
- **Temporal consistency.** Carpet/vehicle state can flicker frame-to-frame;
  a simple majority vote over a small window stabilizes the per-event label.

---

## 9. File map

| Path | Role |
|------|------|
| [configs/detection.yaml](../configs/detection.yaml) | all tunable parameters |
| [adascope/cli/extract.py](../adascope/cli/extract.py) | ffmpeg-free frame sampler (`adascope extract`) |
| [adascope/analysis.py](../adascope/analysis.py) | ROI + carpet analysis → frames / video / CSV |
| [adascope/detector.py](../adascope/detector.py) | `Detector` port + `YoloeDetector` adapter |
| `data/raw/cluster_video.mp4` | downloaded 1080p clip (gitignored, regenerated via §4.1) |
| `data/frames/raw/*.jpg` | extracted frames (gitignored, regenerated via §4.2) |
| `outputs/roi_debug/debug.mp4` | full debug video (every frame) |
| `outputs/roi_debug/states.csv` | per-frame state table |

## 10. References

- Ultralytics YOLOE documentation
- YOLOE: Real-Time Seeing Anything (arXiv 2503.07465)
- YOLOE concepts & citations: [07_YOLOE_CONCEPTS.md](07_YOLOE_CONCEPTS.md)
- Clip: <https://www.youtube.com/watch?v=0Ol8MA9e8nM>

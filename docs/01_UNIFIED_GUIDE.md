# adascope — Unified Guide

This is the main documentation entry point for the project. It consolidates the
current repository layout, the intended workflow, the core tools, and where to
go for deeper detail.

## Documentation Order

Read the docs in this sequence:

1. [01_UNIFIED_GUIDE.md](01_UNIFIED_GUIDE.md) - overview, workflow, and project map
2. [02_GETTING_STARTED.md](02_GETTING_STARTED.md) - environment setup and first run
3. [03_USER_GUIDE.md](03_USER_GUIDE.md) - workflows, recipes, and common tasks
4. [04_ARCHITECTURE.md](04_ARCHITECTURE.md) - system design and data flow
5. [05_DESIGN_AND_CONCEPTS.md](05_DESIGN_AND_CONCEPTS.md) - design choices, ROI logic, and detection theory
6. [06_API_REFERENCE.md](06_API_REFERENCE.md) - command-line options, config schema, and outputs
7. [07_YOLOE_CONCEPTS.md](07_YOLOE_CONCEPTS.md) - YOLOE background and open-vocabulary concepts
8. [08_ASSISTED_LANE_CHANGE_CASE_STUDY.md](08_ASSISTED_LANE_CHANGE_CASE_STUDY.md) - Audi lane-change example and findings

## What This Project Does

The project analyzes vehicle and lane-change state from video using YOLOE and
classic image processing:

```text
Video -> frames -> ROI/crop configuration -> YOLOE vehicle detection -> HSV lane-status detection -> CSV + debug video
```

Core outputs:

- `outputs/roi_debug/states.csv` - per-frame vehicle counts and lane status
- `outputs/roi_debug/debug.mp4` - annotated debug video
- `outputs/videos/final_output.mp4` - reassembled output video when requested

## Quick Start

```powershell
.venv\Scripts\Activate.ps1
adascope pipeline
```

For a full automated run:

```powershell
adascope pipeline --full
```

For selected steps:

```powershell
adascope pipeline --steps 2,4,5,7,8 --skip-download
```

## Current Repository Layout

```text
.
|-- config/
|   `-- config.yaml
|-- data/
|   |-- raw/
|   `-- frames/
|       |-- raw/
|       `-- cropped/
|-- docs/
|   |-- 01_UNIFIED_GUIDE.md
|   |-- 02_GETTING_STARTED.md
|   |-- 03_USER_GUIDE.md
|   |-- 04_ARCHITECTURE.md
|   |-- 05_DESIGN_AND_CONCEPTS.md
|   |-- 06_API_REFERENCE.md
|   |-- 07_YOLOE_CONCEPTS.md
|   `-- 08_ASSISTED_LANE_CHANGE_CASE_STUDY.md
|-- models/
|-- outputs/
|   |-- roi_debug/
|   |-- analysis/
|   `-- videos/
|-- src/
|   `-- adascope/             # installable package (pip install -e .)
|       |-- analysis.py carpet.py geometry.py render.py   # pure domain core
|       |-- detector.py         # Detector port + YoloeDetector adapter
|       |-- config.py frames.py states.py paths.py
|       `-- cli/                # `adascope <command>`
|-- tests/                      # unit tests
`-- adascope/                  # das Paket
```

`data/`, `models/`, generated TorchScript files, and `outputs/` are intentionally
ignored by git because they are large or generated artifacts.

## Main Workflow

1. Download or provide a video in `data/raw/`.
2. Extract frames into `data/frames/raw/`.
3. Manually remove unusable frames if needed.
4. Configure lane ROIs with `cli/roi_editor.py`.
5. Configure crop box with `cli/crop_selector.py`.
6. Crop frames with `cli/crop.py`.
7. Run detection with `adascope/analysis.py`.
8. Assemble output video with `cli/assemble.py`.

The orchestrator in `adascope pipeline` runs these steps interactively or by
selected step numbers.

## Core Tools

| Tool | Purpose |
|------|---------|
| `adascope pipeline` | Interactive pipeline orchestrator |
| `cli/download.py` | Download source video with `yt-dlp` |
| `cli/extract.py` | Extract video frames |
| `cli/roi_editor.py` | Edit lane polygons |
| `cli/crop_selector.py` | Edit frame crop box |
| `cli/crop.py` | Batch crop frames using `configs/detection.yaml` |
| `adascope/analysis.py` | Run YOLOE detection and lane-status analysis |
| `cli/assemble.py` | Build a video from frames |

## Configuration

All tunable runtime values live in `configs/detection.yaml`.

Important fields:

- `model.checkpoint` - YOLOE checkpoint filename or path
- `model.classes` - text prompts such as `[car]`
- `model.conf` - detection confidence threshold
- `rois` - normalized lane polygons
- `crop_box` - normalized crop rectangle
- `carpet` - HSV thresholds for green/red/white lane-status detection

Use the GUI tools for ROI and crop changes instead of editing polygons by hand
unless you are intentionally tuning exact values.

## Output Format

CSV output uses one row per frame:

```csv
frame,veh_left,veh_ego,veh_right,state_left,state_ego,state_right
frame_000100.jpg,1,0,1,available,drivable,clear
```

Vehicle counts are assigned by ROI polygon membership. Carpet state is detected
inside configured side-lane polygons using HSV thresholds.

## Empty Folders

Some folders are expected to be empty in a clean checkout:

- `data/raw/` - source videos go here
- `data/frames/raw/` - extracted frames go here
- `data/frames/cropped/` - cropped frames go here
- `outputs/analysis/` - optional analysis exports
- `outputs/videos/` - assembled output videos

The only active Python package under `src/` is `adascope`, which exposes the
YOLOE model loader. The previous `src/utils` placeholder was removed because it
had no implementation.

## Troubleshooting Path

- Setup or dependency issue: read [02_GETTING_STARTED.md](02_GETTING_STARTED.md)
- Workflow question: read [03_USER_GUIDE.md](03_USER_GUIDE.md)
- Config or output schema: read [06_API_REFERENCE.md](06_API_REFERENCE.md)
- Why the system works this way: read [05_DESIGN_AND_CONCEPTS.md](05_DESIGN_AND_CONCEPTS.md)
- YOLOE background: read [07_YOLOE_CONCEPTS.md](07_YOLOE_CONCEPTS.md)

## Recommended Next Step

Run:

```powershell
adascope pipeline --steps 2,4,5,7,8 --skip-download
```

Use this when a video already exists in `data/raw/cluster_video.mp4`. It exercises
the current pipeline without re-downloading the source video.

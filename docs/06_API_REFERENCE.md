# API Reference

**Detailed tool options, configuration parameters, and output schemas**

---

## Table of Contents

- [Tools](#-tools) — Each script and its options
- [Configuration](#-configuration) — config.yaml reference
- [Output Formats](#-output-formats) — CSV, video, etc.
- [Environment Variables](#-environment-variables) — System-level config

---

## 🛠️ Tools

### run_pipeline.py

**Interactive orchestrator for the full pipeline**

```bash
python run_pipeline.py [OPTIONS]
```

**Options:**
```
--full              Run all 8 steps non-interactively
--steps N,N,N       Run specific steps (comma-separated)
--skip-download     Skip download step when it is included
-h, --help          Show this help
```

**Examples:**
```bash
# Interactive menu
adascope pipeline

# Full pipeline (no prompts)
adascope pipeline --full

# Specific steps: download, extract, detect, output
adascope pipeline --steps 1,2,7,8

# Resume from step 4
adascope pipeline --steps 4,5,6,7,8
```

**Exit codes:**
- `0` — Success
- `1` — Invalid step selection or a selected step failed

---

### adascope/analysis.py

**Main detection & analysis tool**

```bash
adascope detect [OPTIONS]
```

**Input options (pick one):**
```
--frame PATH        Single frame (path to .jpg/.png)
--frames DIR        All frames in directory
--video PATH        Video file (.mp4/.mov/.avi)
```

**Output options:**
```
--csv PATH          Export per-frame data to CSV
--out-video PATH    Save annotated video
--out DIR           Output directory for annotated frames
```

**Processing options:**
```
--stride N          Process every Nth video frame (default: 1)
```

**Configuration options:**
```
--config-dir DIR       Config file (default: config/detection.yaml)
--model PATH        YOLOE checkpoint (overrides config)
--classes C1 C2...  Text prompts, space-separated (overrides config)
--conf THRESHOLD    Detection confidence 0..1 (overrides config)
```

**Examples:**

```bash
# Single frame
adascope detect --frame data/frames/raw/frame_000100.jpg

# All frames with CSV export
adascope detect --frames data/frames/raw \
  --csv outputs/states.csv \
  --out-video outputs/debug.mp4

# Video file, every 5th frame (fast preview)
adascope detect --video data/raw/cluster_video.mp4 \
  --stride 5 \
  --out-video outputs/preview.mp4

# Custom model & prompts
adascope detect --frames data/frames/raw \
  --model models/yoloe-26l-seg.pt \
  --classes car vehicle truck bus \
  --conf 0.05 \
  --csv outputs/experimental.csv

# Just analyze, no output
adascope detect --frame data/frames/raw/frame_000100.jpg
```

**Output:**
- Console: Per-frame vehicle counts and carpet status
- `--csv`: CSV file with per-frame data
- `--out-video`: MP4 video with overlays
- `--out`: Annotated frame output directory

---

### cli/roi_editor.py

**Edit lane region polygons interactively**

```bash
adascope roi-editor [OPTIONS]
```

**Options:**
```
--frame PATH        Frame to display (default: frame_001784.jpg)
--frames-dir PATH   Folder used to resolve frame filenames
--config-dir DIR       Config file (default: config/detection.yaml)
--dump PATH         Render current ROIs to an image and exit
```

**Keyboard & mouse controls:**

| Input | Action |
|-------|--------|
| **Left mouse drag** | Move selected vertex |
| **Right mouse click** | Add vertex to current ROI |
| **D** | Delete selected vertex (minimum 3 per polygon) |
| **Tab** | Cycle active ROI: left → ego → right → left |
| **S** | Save changes to config.yaml |
| **R** | Reset to original (discard all changes) |
| **Q or Esc** | Quit without saving |

**GUI elements:**
- **Active ROI:** Highlighted in white
- **Vertices:** Yellow circles (active), blue circles (inactive)
- **Selected vertex:** Large green circle
- **Polygon edges:** Colored lines (left=blue, ego=white, right=orange)

**Example:**
```bash
# Edit frame 100
adascope roi-editor --frame data/frames/raw/frame_000100.jpg

# Use custom config
adascope roi-editor \
  --frame data/frames/raw/frame_000200.jpg \
  --config configs/my_config.yaml
```

**Output:**
- `config/detection.yaml` (if saved with S key)
- No output if quit without saving (Q key)

---

### cli/crop_selector.py

**Edit crop box interactively**

```bash
adascope crop-box [OPTIONS]
```

**Options:**
```
--frame PATH        Frame to display (default: frame_001784.jpg)
--frames-dir PATH   Folder used to resolve frame filenames
--config-dir DIR       Config file (default: config/detection.yaml)
--dump PATH         Render current crop box to an image and exit
--max-width N       Maximum display width
--max-height N      Maximum display height
--step N            Arrow-key nudge size in pixels
--fast-step N       Shift+arrow nudge size in pixels
```

**Keyboard & mouse controls:**

| Input | Action |
|-------|--------|
| **Left mouse drag** | Move or resize crop box from any handle |
| **Right mouse click** | Reset crop to full frame |
| **1 / 2 / 3 / 4** | Select left / right / top / bottom edge |
| **Arrow keys** | Nudge selected edge |
| **Shift + arrow** | Nudge selected edge faster |
| **Tab** | Cycle selected edge |
| **M** | Select move mode |
| **R** | Reset crop to full frame |
| **S** | Save changes to config.yaml |
| **P** | Save preview image (crop_preview.jpg) |
| **C** | Cycle presets (if configured) |
| **Q or Esc** | Quit without saving |

**GUI elements:**
- **Crop box:** Yellow rectangle with corner handles
- **Handles:** Larger circles at corners and edges (25px grab zone)
- **Crosshair:** Green lines showing crop center
- **HUD:** Information panel at top (crop coordinates, dimensions)

**Example:**
```bash
# Edit crop on frame 100
adascope crop-box --frame data/frames/raw/frame_000100.jpg

# Use custom config
adascope crop-box \
  --frame data/frames/raw/frame_000200.jpg \
  --config configs/my_config.yaml
```

**Output:**
- `config/detection.yaml` with updated `crop_box:` field (if saved with S)
- `crop_preview.jpg` (if saved with P)

---

### cli/download.py

**Download video from YouTube**

```bash
adascope download [OPTIONS]
```

**Options:**
```
--url URL           YouTube URL (default: Audi assisted-lane-change)
--out PATH          Output file path (default: data/raw/cluster_video.mp4)
--height N          Maximum video height (default: 1080)
--ffmpeg-location PATH
                    Folder containing ffmpeg (auto-detected if omitted)
--force             Overwrite existing output
```

**Requirements:**
- `yt-dlp` installed (`pip install -U yt-dlp`)
- `ffmpeg` in PATH or specified with `--ffmpeg-location`

**Examples:**
```bash
# Default (Audi demo)
adascope download

# Custom URL
adascope download \
  --url "https://www.youtube.com/watch?v=abc123def456"

# Custom output path
adascope download --out videos/my_video.mp4

# Explicit ffmpeg location
adascope download \
  --ffmpeg-location "C:\ffmpeg\bin"

# Lower maximum height
adascope download --height 720
```

**Output:**
- `data/raw/cluster_video.mp4` (or specified path)
- Resolution: 1920×1080 (or best available)
- Frame rate: 25 fps (or original)
- Format: MP4 with H.264 video + AAC audio

---

### cli/extract.py

**Extract frames from video**

```bash
adascope extract [OPTIONS]
```

**Required options:**
```
--video PATH        Input video file (required)
```

**Output options:**
```
--out PATH          Output directory (default: data/frames/raw)
```

**Sampling options:**
```
--every N           Keep every Nth frame (default: 1 = all)
--fps RATE          Sample at this fps (exclusive with --every)
--max-frames N      Stop after N frames (default: 0 = no limit)
```

**Format options:**
```
--ext FORMAT        Output format: jpg or png (default: jpg)
--quality Q         JPEG quality 1-100 (default: 95)
```

**Examples:**
```bash
# Extract all frames
adascope extract --video data/raw/cluster_video.mp4

# Extract every 2nd frame
adascope extract --video data/raw/cluster_video.mp4 --every 2

# Sample at 2 fps (for long videos)
adascope extract --video data/raw/cluster_video.mp4 --fps 2

# PNG format (lossless, larger files)
adascope extract --video data/raw/cluster_video.mp4 --ext png

# First 100 frames only
adascope extract --video data/raw/cluster_video.mp4 --max-frames 100

# Custom output directory
adascope extract --video data/raw/cluster_video.mp4 --out data/frames_test
```

**Output:**
- `data/frames/raw/frame_000000.jpg`
- `data/frames/raw/frame_000001.jpg`
- ... (one per extracted frame)

**Frame numbering:**
- Zero-padded: `frame_000000.jpg` (6 digits)
- Sorted naturally for video reassembly

---

### cli/crop.py

**Batch crop frames**

```bash
adascope crop [OPTIONS]
```

**Input options (pick one):**
```
--frames DIR        Input frame directory
--video PATH        Input video
--frame PATH        Single frame
```

**Output options:**
```
--out PATH          Output directory (default: data/frames/cropped)
```

**Crop options (pick one):**
```
--box X0 Y0 X1 Y1   Crop coordinates as fractions 0..1
                    (if not provided, reads from config/detection.yaml)
--config-dir DIR       Config file containing crop_box
```

**Format options:**
```
--quality Q         JPEG quality 1-100 (default: 95)
--ext FORMAT        Output format: jpg or png (default: jpg)
```

**Examples:**
```bash
# Use crop_box from config
adascope crop --frames data/frames/raw --out data/frames/cropped

# Override crop box
adascope crop --frames data/frames/raw \
  --out data/frames/cropped \
  --box 0.1 0.2 0.9 0.8

# Lower JPEG quality (smaller files)
adascope crop --frames data/frames/raw \
  --out data/frames/cropped \
  --quality 85

# PNG format
adascope crop --frames data/frames/raw \
  --out data/frames/cropped \
  --ext png
```

**Output:**
- `data/frames/cropped/frame_000000.jpg`
- `data/frames/cropped/frame_000001.jpg`
- ... (same frame numbers as input)

---

### cli/assemble.py

**Build video from frames**

```bash
adascope assemble [OPTIONS]
```

**Required options:**
```
--frames DIR        Input frame directory (required)
--out PATH          Output video path (required)
```

**Playback options:**
```
--fps RATE          Output frame rate (default: 25)
--fourcc CODEC      OpenCV fourcc codec (default: mp4v)
```

**Input options:**
```
--ext FORMAT        Only use files with this extension (jpg, png)
--every N           Use every Nth frame (default: 1 = all)
```

**Examples:**
```bash
# Standard output
adascope assemble \
  --frames data/frames/raw \
  --out outputs/videos/output.mp4

# 30 fps instead of 25
adascope assemble \
  --frames data/frames/raw \
  --out outputs/videos/output.mp4 \
  --fps 30

# From cropped frames
adascope assemble \
  --frames data/frames/cropped \
  --out outputs/videos/cluster.mp4

# PNG files only
adascope assemble \
  --frames data/frames/raw \
  --out output.mp4 \
  --ext png

# Every other frame (50% speed)
adascope assemble \
  --frames data/frames/raw \
  --out output_fast.mp4 \
  --every 2
```

**Output:**
- `output.mp4` (playable with any video player)

---

## ⚙️ Configuration

### config/detection.yaml

Complete reference with annotations:

```yaml
# ============================================================================
# YOLOE Analysis Configuration
# ============================================================================

# ---- Model Selection & Inference Settings ----
model:
  # Model checkpoint file (must exist in models/ directory)
  # Options: yoloe-11l-seg.pt (1.1 GB, accurate)
  #          yoloe-26l-seg.pt (2.3 GB, faster)
  #          yoloe-26m-seg.pt (0.5 GB, small)
  #          yoloe-11l-seg-pf.pt (1.1 GB, no text prompts needed)
  checkpoint: yoloe-11l-seg.pt
  
  # Text prompts (what to detect)
  # One prompt per item; YOLOE matches prompts at inference time
  # Examples:
  #   [car]                           -- Only cars
  #   [car, truck, bus]               -- Multiple vehicle types
  #   [car, vehicle]                  -- Redundant but covers more shapes
  #   [sedan, suv, "white car"]       -- Specific types + color
  classes:
    - car
  
  # Detection confidence threshold (0..1)
  # Lower = more detections (more false positives)
  # Higher = fewer detections (more misses)
  # Typical range: 0.05 (permissive) to 0.3 (strict)
  conf: 0.1

# ---- Lane Regions (Perspective Polygons) ----
# Each ROI is a polygon that follows a lane in 3D
# Coordinates are FRACTIONS (0..1) of frame width/height
# This makes them resolution-independent
# 
# Typical structure:
#   Top points:    near vanishing point (small x-spread, small y)
#   Bottom points: at near edge (large x-spread, larger y)
# 
# Vehicles are assigned to ROIs based on detection box center
# Tested in order: left, ego, right (first match wins)

rois:
  left:
    - [0.4516, 0.2657]  # Top-left
    - [0.4807, 0.2657]  # Top-right
    - [0.4437, 0.4917]  # Bottom-left
    - [0.3339, 0.4917]  # Bottom-right
  
  ego:
    - [0.4734, 0.3111]
    - [0.5177, 0.3111]
    - [0.5448, 0.4667]
    - [0.4474, 0.4676]
  
  right:
    - [0.5104, 0.2639]
    - [0.5385, 0.2648]
    - [0.6557, 0.4880]
    - [0.5490, 0.4880]

# ---- Crop Box (Instrument Cluster Extraction) ----
# Region to extract when using crop.py
# Format: [x_min, y_min, x_max, y_max] as fractions 0..1
# 0 = left/top, 1 = right/bottom
crop_box: [0.135, 0.175, 0.865, 0.655]

# ---- ROI Colors (for visualization) ----
# BGR format (Blue, Green, Red) used by OpenCV
# Range: 0-255 per channel
# Used for drawing polygon outlines and vehicle boxes in debug video
roi_colors:
  left:  [255, 180, 0]      # Blue-tinted
  ego:   [255, 255, 255]    # White
  right: [0, 180, 255]      # Orange-tinted

# ---- Carpet Detection (Driving Area Status) ----
# Detects green (available) / red (blocked) lane markings
# Uses HSV color thresholding on frame regions
# 
# HSV ranges (OpenCV convention):
#   H (Hue):        0-179 (wraps at 180)
#   S (Saturation): 0-255 (0=gray, 255=pure color)
#   V (Value):      0-255 (0=black, 255=bright)

carpet:
  # Which ROIs to check for carpet (usually left+right)
  detect_in: [left, right]
  
  # Minimum pixel fraction (within ROI polygon) to count as carpet
  # Range: 0.01-0.5
  # Typical: 0.04 (4% of polygon pixels must match color)
  min_frac: 0.04
  
  # Higher threshold for white (lane markings are also white)
  # When both white and green/red pass thresholds, colored wins
  white_min_frac: 0.18
  
  # Green carpet (lane available for change)
  # Lower bound [H_min, S_min, V_min]
  # Upper bound [H_max, S_max, V_max]
  green_hsv:
    - [38, 70, 70]        # Lower
    - [90, 255, 255]      # Upper
  
  # Red carpet (lane blocked/aborted)
  # Red wraps hue circle, so two ranges needed
  red_hsv:
    - [[0, 90, 80], [12, 255, 255]]       # Lower red wrap
    - [[168, 90, 80], [180, 255, 255]]    # Upper red wrap
  
  # White (lane markings, no action)
  white_hsv:
    - [0, 0, 95]          # Lower
    - [179, 45, 170]      # Upper
```

---

## 📊 Output Formats

### CSV: states.csv

Per-frame detection results:

```csv
frame,veh_left,veh_ego,veh_right,state_left,state_ego,state_right
0,1,0,1,clear,drivable,clear
1,1,0,1,clear,drivable,clear
2,0,0,1,blocked,drivable,clear
3,0,0,1,blocked,drivable,available
4,0,0,0,clear,drivable,clear
```

**Columns:**

| Name | Type | Range | Meaning |
|------|------|-------|---------|
| `frame` | int/str | index/filename | Frame identifier |
| `veh_left` | int | 0-∞ | **Other** vehicle count in left ROI |
| `veh_ego` | int | 0-∞ | Other vehicle count in ego ROI (ego car excluded via `ego_box`) |
| `veh_right` | int | 0-∞ | Other vehicle count in right ROI |
| `state_left` | str | available/blocked/clear | Side-lane drivable state |
| `state_ego` | str | drivable/blocked/clear | Ego-lane state (white path) |
| `state_right` | str | available/blocked/clear | Side-lane drivable state |

**Analysis example (Python):**
```python
import pandas as pd

df = pd.read_csv('outputs/roi_debug/states.csv')

# Frames where a left lane change is offered
available = df[df['state_left'] == 'available']
print(f"Left change available: {len(available)} frames")

# Frames where the left lane is blocked (not drivable)
blocked = df[df['state_left'] == 'blocked']
print(f"Left blocked: {len(blocked)} frames")

# Statistics
print(df.describe())
```

### Video: debug.mp4

Annotated video with:
- **ROI polygons** (colored outlines)
- **Vehicle boxes** (colored by ROI, labeled with count)
- **Carpet status banner** (green/red indicators)
- **Frame counter**

**Format:**
- Codec: H.264 (MP4)
- Resolution: Same as input frames
- Frame rate: 25 fps (default, customizable)
- Playback: Standard players (VLC, Windows Media Player, etc.)

### Frame: crop_preview.jpg

Single-frame preview generated by `crop_selector.py`:
- Shows crop box overlay on frame
- Useful for preview before batch cropping

---

## 🌍 Environment Variables

```bash
# Use CPU instead of GPU (slower, no CUDA needed)
$Env:CUDA_VISIBLE_DEVICES = -1

# Use specific GPU device (0-based index)
$Env:CUDA_VISIBLE_DEVICES = 0

# Disable YOLOE telemetry
$Env:YOLO_TELEMETRY = 0

# Quiet mode (less console output)
$Env:YOLO_VERBOSE = 0
```

---

## 🔍 Debugging Flags

Enable detailed output:

```bash
# Print debug info (OpenCV, YOLOE)
adascope detect --frame data/frames/raw/frame_000100.jpg 2>&1 | tee debug.log

# Capture full output
python run_pipeline.py > pipeline.log 2>&1
```

---

**Need help?** See [03_USER_GUIDE.md](03_USER_GUIDE.md) for workflows

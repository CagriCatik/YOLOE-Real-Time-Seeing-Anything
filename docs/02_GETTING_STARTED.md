# Getting Started

**Setup, dependencies, and your first run — 15 minutes**

---

## ✅ Prerequisites

Before starting, ensure you have:
- Windows 10+ or Linux/Mac (with appropriate shell)
- Python 3.9+
- Git (optional, for cloning)
- At least 10 GB free space (for video + frames + models)

**External tools** (optional, only needed for YouTube download):
- `ffmpeg` — Video merging. Install from [ffmpeg.org](https://ffmpeg.org/download.html)
- `yt-dlp` — YouTube downloader. Install via: `pip install -U yt-dlp`

---

## 🚀 Installation (5 minutes)

### Step 1: Create Virtual Environment
```powershell
# Navigate to project
cd C:\Users\mccat\Desktop\YOLOE-Real-Time-Seeing-Anything

# Create venv
python -m venv .venv

# Activate it
.venv\Scripts\Activate.ps1
```

If you get an execution policy error:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.venv\Scripts\Activate.ps1
```

### Step 2: Install the Package (editable)
```powershell
pip install --upgrade pip
pip install -e ".[dev]"        # installs deps + the `adascope` command + pytest
```

This installs the `adascope` package (so `import adascope` and the
`adascope` CLI work from anywhere) plus:
- `ultralytics` — YOLOE model
- `supervision` — Detection post-processing
- `opencv-python` — Image processing
- `pyyaml` — Configuration
- `tqdm` — Progress bars

> The editable install replaces the old `sys.path` shim. `add ".[download]"` for
> the yt-dlp download step.

### Step 3: Verify Installation
```powershell
python -c "import adascope; print('✅', adascope.__version__)"
adascope --help
pytest -q                       # 21 unit tests should pass
```

---

## 🎬 First Run (5 minutes)

### Option A: Interactive Menu (Recommended)
```powershell
adascope pipeline
```

You'll see:
```
╔════════════════════════════════════════╗
║  YOLOE Real-Time Analysis Pipeline     ║
╚════════════════════════════════════════╝

Select steps to run:
  1. Download video
  2. Extract frames
  3. Manual cleanup
  4. Configure ROI
  5. Configure crop box
  6. Crop frames
  7. Run detection
  8. Build output video

Enter step numbers (comma-separated) or 'a' for all: 
```

Choose:
- `1` for full pipeline from scratch
- `1,7,8` to skip manual tuning
- `7` to just run detection on existing frames

### Option B: Full Pipeline (One Command)
```powershell
adascope pipeline --full
```

Runs all 8 steps without prompting. Takes ~30-60 minutes depending on video length.

### Option C: Single Frame Test
Quick 30-second test:
```powershell
# Requires existing frame (e.g., frame_000100.jpg in data/frames/raw/)
adascope detect --frame data/frames/raw/frame_000100.jpg
```

---

## 📁 Verify Your Setup

After installation, your project should look like:
```
YOLOE/
├── .venv/                       # Virtual environment (created)
├── models/
│   ├── yoloe-11l-seg.pt        # ✅ Must exist
│   └── yoloe-11l-seg-pf.pt
├── config/
│   └── config.yaml             # ✅ Must exist
├── adascope/             # ✅ Installable package (pip install -e .)
│   └── cli/                    # ✅ `adascope <command>`
├── tests/                      # ✅ Unit tests
├── data/                        # ✅ Created automatically
├── outputs/                     # ✅ Created automatically
└── pyproject.toml              # ✅ Must exist
```

Run this to verify:
```powershell
python -c "
import sys
from pathlib import Path

checks = [
    ('Virtual env', '.venv' in sys.prefix),
    ('Models exist', (Path('models') / 'yoloe-11l-seg.pt').exists()),
    ('Config exists', (Path('config') / 'config.yaml').exists()),
    ('Package installed', __import__('importlib.util', fromlist=['util']).util.find_spec('adascope') is not None),
]

for name, ok in checks:
    print(f'{"✅" if ok else "❌"} {name}')
"
```

---

## ⚙️ Configuration

### Model Selection

Edit `configs/detection.yaml` to choose YOLOE variant:

```yaml
model:
  checkpoint: yoloe-11l-seg.pt   # Full model
  # OR:
  # checkpoint: yoloe-11l-seg-pf.pt  # Prompt-free (explore what's in frame)
  # checkpoint: yoloe-26l-seg.pt      # Larger, slower, more accurate
  
  classes: [car]                  # Text prompts (what to detect)
  conf: 0.1                       # 0.05=more detections, 0.2=fewer
```

### ROI Polygons (Lane Regions)

Default polygons are for the Audi assisted-lane-change demo. To use your own video:

```bash
adascope roi-editor --frame data/frames/raw/frame_000100.jpg
```

**Controls:**
- **Left-drag** — Move vertex
- **Right-click** — Add vertex
- **D** — Delete vertex  
- **Tab** — Cycle ROI (left/ego/right)
- **S** — Save
- **R** — Reset
- **Q** — Quit

### Crop Region (Instrument Cluster)

Adjust the region to extract:
```bash
adascope crop-box --frame data/frames/raw/frame_000100.jpg
```

**Controls:**
- **Left-drag** — Move/resize box
- **W/X/A/D** — Fine adjust edges
- **Right-click** — Reset to full frame
- **S** — Save
- **Q** — Quit

---

## 📊 Understanding Output

After running `adascope detect`:

```
outputs/roi_debug/
├── debug.mp4              # Annotated video
└── states.csv             # Per-frame data
```

**CSV columns:**
- `frame` — Frame index / filename
- `veh_left` / `veh_ego` / `veh_right` — count of **other** vehicles per lane
  (the ego car is excluded via `ego_box`, so `veh_ego` is usually 0)
- `state_left` / `state_right` — side-lane drivable state: `available` (green) /
  `blocked` (red) / `clear`
- `state_ego` — ego-lane state: `drivable` (white path) / `blocked` / `clear`

**Example:**
```
frame,veh_left,veh_ego,veh_right,state_left,state_ego,state_right
0,1,0,1,clear,drivable,clear
1,1,0,1,clear,drivable,clear
2,0,0,1,blocked,drivable,clear
```

---

## 🆘 Troubleshooting

### "ModuleNotFoundError: No module named 'ultralytics'"
```powershell
# Activate venv first
.venv\Scripts\Activate.ps1

# Then reinstall
pip install -r requirements.txt
```

### "ffmpeg not found" (YouTube download fails)
```powershell
# Only needed if downloading from YouTube
# Download ffmpeg from https://ffmpeg.org/download.html
# Or use an existing video and skip step 1

# If ffmpeg is installed elsewhere:
adascope download --ffmpeg-location "C:\Path\to\ffmpeg\bin"
```

### "CUDA out of memory" (GPU inference)
Your GPU is too small for YOLOE-11L. Options:
1. Use smaller model: Change `checkpoint: yoloe-26l-seg.pt` to `yoloe-26m-seg.pt`
2. Use CPU: Set `CUDA_VISIBLE_DEVICES=-1` before running
3. Process fewer frames: Use `--stride 5` in `detect.py`

### "No detections found"
Adjust in `configs/detection.yaml`:
```yaml
model:
  conf: 0.05              # Lower = more detections
  classes: [car, vehicle] # Broader prompts
```

### "Crop box outside image bounds"
Reset in `crop_selector.py` with **right-click**, or manually edit:
```yaml
crop_box: [0.0, 0.0, 1.0, 1.0]  # Full image
```

---

## 📚 Next Steps

1. **Run the full pipeline:** `adascope pipeline`
2. **Read the user guide:** [03_USER_GUIDE.md](03_USER_GUIDE.md)
3. **Understand the architecture:** [04_ARCHITECTURE.md](04_ARCHITECTURE.md)
4. **Deep dive on YOLOE:** [05_DESIGN_AND_CONCEPTS.md](05_DESIGN_AND_CONCEPTS.md)

---

## 🎯 Quick Reference

| Task | Command |
|------|---------|
| Run interactive menu | `adascope pipeline` |
| Run full pipeline | `adascope pipeline --full` |
| Edit lane regions | `adascope roi-editor` |
| Edit crop box | `adascope crop-box` |
| Analyze one frame | `adascope detect --frame data/frames/raw/frame_000100.jpg` |
| Download video | `adascope download` |
| Extract frames | `adascope extract --video data/raw/video.mp4` |

---

**Ready?** Start with: `adascope pipeline`

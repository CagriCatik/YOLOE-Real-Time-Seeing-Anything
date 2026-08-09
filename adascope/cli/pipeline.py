"""In-process workflow orchestrator."""

from . import assemble, crop, detect, download, extract, roi_editor, crop_selector


def configure_parser(parser):
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--steps", help="Comma-separated step numbers (1-7)")
    parser.add_argument("--skip-download", action="store_true")


def run(args):
    if args.full:
        steps = list(range(1, 8))
    elif args.steps:
        try: steps = [int(value) for value in args.steps.split(",")]
        except ValueError: raise SystemExit("--steps must be comma-separated numbers")
    else:
        print("Steps: 1 download, 2 extract, 3 ROI, 4 crop box, 5 crop, 6 detect, 7 assemble")
        value = input("Select steps (e.g. 2,3,4,5,6), or 'a' for all: ").strip().lower()
        steps = list(range(1, 8)) if value == "a" else [int(item) for item in value.split(",")]
    if any(step not in range(1, 8) for step in steps): raise SystemExit("steps must be between 1 and 7")
    if 1 in steps and not args.skip_download:
        url = download.DEFAULT_URL if args.full else input(f"YouTube URL [{download.DEFAULT_URL}]: ").strip() or download.DEFAULT_URL
        download.run(_ns(url_arg=url, url_opt=None, out="data/raw/cluster_video.mp4", output_dir=None,
                         height=1080, ffmpeg_location=None, force=False))
    if 2 in steps: extract.run(_ns(video_arg="data/raw/cluster_video.mp4", video_opt=None, out="data/frames/raw", every=1, fps=None, max_frames=0, ext="jpg", quality=95))
    if 3 in steps: roi_editor.run(_ns(frame=None, frames_dir="data/frames/raw", config="config/detection.yaml", dump=None))
    if 4 in steps: crop_selector.run(_ns(frame=None, frames_dir="data/frames/raw", config="config/detection.yaml", dump=None, max_width=1600, max_height=900, step=1, fast_step=10))
    if 5 in steps: crop.run(_ns(frames_arg="data/frames/raw", frames=None, frame=None, video=None, out="data/frames/cropped", box=None, config="config/detection.yaml", quality=95, ext="jpg"))
    if 6 in steps: detect.run(_ns(input="data/frames/cropped", frame=None, frames=None, video=None, config="config/detection.yaml", model=None, classes=None, conf=None, stride=1, area_mode=None, hud=None, rois=None, csv="outputs/roi_debug/states.csv", out_video="outputs/roi_debug/debug.mp4", out="outputs/roi_debug/frames"))
    if 7 in steps: assemble.run(_ns(frames_arg="outputs/roi_debug/frames", frames=None, out="outputs/roi_debug/final.mp4", fps=25, fourcc="mp4v", ext=None, every=1))
    return 0


def _ns(**values):
    from argparse import Namespace
    return Namespace(**values)

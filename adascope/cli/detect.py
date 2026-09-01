from pathlib import Path
import cv2
from ..vision.analysis import analyse_frame
from ..config import load_detection_config
from ..detection import YoloeDetector
from ..vision.driving_area import DrivingAreaDetector
from ..io import list_frames, read_image
from ..vision.hud import HudDetector
from ..render import draw
from ..io import write_rows


def _on_off(value):
    if value not in {"on", "off"}:
        raise ValueError("expected on or off")
    return value == "on"


def configure_parser(parser):
    parser.add_argument("input", nargs="?", help="Frame directory (script compatibility)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--frame")
    group.add_argument("--frames")
    group.add_argument("--video")
    parser.add_argument("--config", default="configs/detection.yaml")
    parser.add_argument("--model")
    parser.add_argument("--classes", nargs="+")
    parser.add_argument("--conf", type=float)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--area-mode", choices=("yoloe", "hsv"))
    parser.add_argument("--hud", choices=("on", "off"))
    parser.add_argument("--rois", choices=("on", "off"))
    parser.add_argument("--csv")
    parser.add_argument("--out-video")
    parser.add_argument("--out")


def run(args):
    config = load_detection_config(args.config)
    model = args.model or config.model.checkpoint
    detector = YoloeDetector(model, args.classes or config.model.classes)
    area_enabled = config.driving_area.enabled if args.area_mode is None else args.area_mode == "yoloe"
    hud_enabled = config.hud.enabled if args.hud is None else _on_off(args.hud)
    area_detector = DrivingAreaDetector(model, config.driving_area.prompts, config.driving_area.conf) if area_enabled else None
    hud_detector = HudDetector(model, config.hud.prompts, config.hud.conf) if hud_enabled else None
    frames = _inputs(args)
    if not frames:
        raise SystemExit("no input frames found")
    out_dir = Path(args.out) if args.out else None
    if out_dir: out_dir.mkdir(parents=True, exist_ok=True)
    video_writer = None; rows = []; count = 0
    show_rois = _on_off(args.rois) if args.rois else not area_enabled
    for name, image in frames:
        result = analyse_frame(detector, image, config, args.conf, name, area_detector, hud_detector)
        rows.append(result.csv_row())
        annotated = draw(image, result, config, show_rois)
        if out_dir:
            cv2.imwrite(str(out_dir / name), annotated)
        if args.out_video:
            if video_writer is None:
                target = Path(args.out_video); target.parent.mkdir(parents=True, exist_ok=True)
                video_writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), 25,
                                               (annotated.shape[1], annotated.shape[0]))
            video_writer.write(annotated)
        count += 1
        print(f"{name}: {result.counts} {result.lane_states}")
    if video_writer: video_writer.release()
    if args.csv: write_rows(rows, args.csv)
    print(f"Analysed {count} frame(s)")
    return 0


def _inputs(args):
    every = max(1, args.stride)
    if args.video:
        capture = cv2.VideoCapture(args.video); index = 0
        while True:
            ok, image = capture.read()
            if not ok: break
            if index % every == 0: yield f"frame_{index:06d}.jpg", image
            index += 1
        capture.release(); return
    if args.frame:
        image = read_image(args.frame)
        if image is not None: yield Path(args.frame).name, image
        return
    for path in list_frames(args.frames or args.input or "data/frames/cropped", every):
        image = read_image(path)
        if image is not None: yield path.name, image

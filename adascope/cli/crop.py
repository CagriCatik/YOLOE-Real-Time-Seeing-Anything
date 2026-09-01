from pathlib import Path
import cv2
from ..config import load_detection_config
from ..io import list_frames, read_image


def configure_parser(parser):
    parser.add_argument("frames_arg", nargs="?")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--frames")
    group.add_argument("--frame")
    group.add_argument("--video")
    parser.add_argument("--out", default="data/frames/cropped")
    parser.add_argument("--box", nargs=4, type=float)
    parser.add_argument("--config", default="configs/detection.yaml")
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--ext", choices=("jpg", "png"), default="jpg")


def run(args):
    box = tuple(args.box) if args.box else load_detection_config(args.config).crop_box
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    params = [cv2.IMWRITE_JPEG_QUALITY, args.quality] if args.ext == "jpg" else []
    paths = [Path(args.frame)] if args.frame else list_frames(args.frames or args.frames_arg or "data/frames/raw")
    written = 0
    for path in paths:
        image = read_image(path)
        if image is None:
            continue
        h, w = image.shape[:2]; x0, y0, x1, y1 = box
        cropped = image[round(y0*h):round(y1*h), round(x0*w):round(x1*w)]
        cv2.imwrite(str(out / f"{path.stem}.{args.ext}"), cropped, params); written += 1
    print(f"Cropped {written} frame(s) to {out}")
    return 0

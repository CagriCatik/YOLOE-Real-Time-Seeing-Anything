from pathlib import Path
import cv2
from ..io import list_frames, read_image


def configure_parser(parser):
    parser.add_argument("frames_arg", nargs="?")
    parser.add_argument("--frames")
    parser.add_argument("--out", required=True)
    parser.add_argument("--fps", type=float, default=25)
    parser.add_argument("--fourcc", default="mp4v")
    parser.add_argument("--ext")
    parser.add_argument("--every", type=int, default=1)


def run(args):
    paths = list_frames(args.frames or args.frames_arg, args.every, args.ext)
    if not paths:
        raise SystemExit("no input frames found")
    first = read_image(paths[0])
    if first is None:
        raise SystemExit(f"could not read first frame: {paths[0]}")
    target = Path(args.out); target.parent.mkdir(parents=True, exist_ok=True)
    size = (first.shape[1], first.shape[0])
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*args.fourcc), args.fps, size)
    if not writer.isOpened():
        raise SystemExit(f"could not create video: {target}")
    count = 0
    for path in paths:
        image = read_image(path)
        if image is None:
            continue
        if (image.shape[1], image.shape[0]) != size:
            image = cv2.resize(image, size)
        writer.write(image); count += 1
    writer.release(); print(f"Wrote {count} frame(s) to {target}")
    return 0

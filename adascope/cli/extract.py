from pathlib import Path
import cv2


def configure_parser(parser):
    parser.add_argument("video_arg", nargs="?")
    parser.add_argument("--video", dest="video_opt")
    parser.add_argument("--out", default="data/frames/raw")
    parser.add_argument("--stride", "--every", dest="every", type=int, default=1)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--ext", choices=("jpg", "png"), default="jpg")
    parser.add_argument("--quality", type=int, default=95)


def run(args):
    video = args.video_opt or args.video_arg
    if not video:
        raise SystemExit("extract requires a video path")
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise SystemExit(f"could not open video: {video}")
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 25
    every = max(1, round(source_fps / args.fps)) if args.fps else args.every
    if every < 1:
        raise SystemExit("--every/--stride must be at least 1")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    params = [cv2.IMWRITE_JPEG_QUALITY, args.quality] if args.ext == "jpg" else []
    source_index = written = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if source_index % every == 0:
            cv2.imwrite(str(out / f"frame_{written:06d}.{args.ext}"), frame, params)
            written += 1
            if args.max_frames and written >= args.max_frames:
                break
        source_index += 1
    capture.release()
    print(f"Extracted {written} frame(s) to {out}")
    return 0

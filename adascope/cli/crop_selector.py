from pathlib import Path
import cv2
from ..config import load_detection_config, save_crop_box
from ..io import list_frames, read_image


def configure_parser(parser):
    parser.add_argument("--frame")
    parser.add_argument("--frames-dir", default="data/frames/raw")
    parser.add_argument("--config", default="config/detection.yaml")
    parser.add_argument("--dump")
    parser.add_argument("--max-width", type=int, default=1600)
    parser.add_argument("--max-height", type=int, default=900)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--fast-step", type=int, default=10)


def run(args):
    paths = [Path(args.frame)] if args.frame else list_frames(args.frames_dir)
    if not paths: raise SystemExit("no frame available for crop selector")
    image = read_image(paths[0]); box = load_detection_config(args.config).crop_box
    if image is None: raise SystemExit(f"could not read frame: {paths[0]}")
    h, w = image.shape[:2]; x0,y0,x1,y1 = box
    cv2.rectangle(image, (round(x0*w),round(y0*h)), (round(x1*w),round(y1*h)), (0,255,255), 3)
    if args.dump:
        Path(args.dump).parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(args.dump, image); return 0
    print("Drag a crop rectangle, then press Enter/Space to save or C/Esc to cancel.")
    selection = cv2.selectROI("YOLOE crop configuration", image, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    sx, sy, sw, sh = selection
    if sw > 0 and sh > 0:
        normalized = [round(sx/w, 6), round(sy/h, 6), round((sx+sw)/w, 6), round((sy+sh)/h, 6)]
        save_crop_box(args.config, normalized); print(f"Saved crop box to {args.config}: {normalized}")
    else:
        print("Crop selection cancelled; config was not changed.")
    return 0

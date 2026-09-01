from pathlib import Path
import cv2
from ..config import load_detection_config, save_rois
from ..io import list_frames, read_image
from ..vision.geometry import polys_from_rois


def configure_parser(parser):
    parser.add_argument("--frame")
    parser.add_argument("--frames-dir", default="data/frames/raw")
    parser.add_argument("--config", default="configs/detection.yaml")
    parser.add_argument("--dump")


def run(args):
    paths = [Path(args.frame)] if args.frame else list_frames(args.frames_dir)
    if not paths: raise SystemExit("no frame available for ROI editor")
    image = read_image(paths[0]); config = load_detection_config(args.config)
    if image is None: raise SystemExit(f"could not read frame: {paths[0]}")
    preview = _render(image, config.rois, config.roi_colors, None)
    if args.dump:
        Path(args.dump).parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(args.dump, preview); return 0
    names = list(config.rois); original = {k: [p[:] for p in v] for k, v in config.rois.items()}
    state = {"rois": {k: [p[:] for p in v] for k, v in original.items()}, "active": 0,
             "selected": None, "dragging": False}
    h, w = image.shape[:2]; window = "YOLOE ROI configuration"
    def mouse(event, x, y, flags, _param):
        name = names[state["active"]]; points = state["rois"][name]
        if event == cv2.EVENT_LBUTTONDOWN:
            distances = [((px*w-x)**2 + (py*h-y)**2, index) for index, (px, py) in enumerate(points)]
            if distances and min(distances)[0] <= 15**2:
                state["selected"] = min(distances)[1]; state["dragging"] = True
            else:
                points.append([x/w, y/h]); state["selected"] = len(points)-1
        elif event == cv2.EVENT_MOUSEMOVE and state["dragging"] and state["selected"] is not None:
            points[state["selected"]] = [min(1,max(0,x/w)), min(1,max(0,y/h))]
        elif event == cv2.EVENT_LBUTTONUP:
            state["dragging"] = False
        elif event == cv2.EVENT_RBUTTONDOWN:
            points.append([x/w, y/h]); state["selected"] = len(points)-1
    cv2.namedWindow(window); cv2.setMouseCallback(window, mouse)
    print("Drag vertices; right-click adds one. Tab: next lane, D: delete, R: reset, S: save, Esc/Q: quit.")
    while True:
        active = names[state["active"]]
        canvas = _render(image, state["rois"], config.roi_colors, active)
        cv2.putText(canvas, f"Active: {active}", (12, canvas.shape[0]-16), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,255,255), 2)
        cv2.imshow(window, canvas); key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q")): break
        if key == 9: state["active"] = (state["active"] + 1) % len(names); state["selected"] = None
        elif key == ord("r"): state["rois"] = {k: [p[:] for p in v] for k,v in original.items()}
        elif key == ord("d") and state["selected"] is not None:
            points = state["rois"][active]
            if len(points) > 3: points.pop(state["selected"]); state["selected"] = None
        elif key == ord("s"):
            save_rois(args.config, state["rois"]); print(f"Saved ROIs to {args.config}"); break
    cv2.destroyWindow(window)
    return 0


def _render(image, rois, colors, active):
    canvas = image.copy()
    for lane, poly in polys_from_rois(rois, image.shape[1], image.shape[0]).items():
        color = (255,255,255) if lane == active else colors.get(lane, [255,255,255])
        cv2.polylines(canvas, [poly], True, color, 3 if lane == active else 2)
        for x, y in poly: cv2.circle(canvas, (int(x), int(y)), 5, color, -1)
    return canvas

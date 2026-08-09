"""Debug overlays for analysis results."""

import cv2
from ..vision.geometry import polys_from_rois


def draw(frame, result, config, show_rois=True):
    image = frame.copy()
    if show_rois:
        for lane, poly in polys_from_rois(config.rois, image.shape[1], image.shape[0]).items():
            cv2.polylines(image, [poly], True, config.roi_colors.get(lane, [255, 255, 255]), 2)
    for box, lane in result.assignments:
        color = config.roi_colors.get(lane, [128, 128, 128])
        cv2.rectangle(image, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), color, 2)
    y = 24
    for lane in ("left", "ego", "right"):
        text = f"{lane}: {result.counts[lane]} / {result.lane_states[lane]}"
        cv2.putText(image, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2)
        y += 22
    return image

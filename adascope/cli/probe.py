"""YOLOE-Textprompts auf einem Einzelbild ausprobieren.

Der schnelle Weg, Prompts zu finden, bevor sie in `config/detection.yaml`
landen: ein Bild rein, annotiertes PNG und JSON raus. Ersetzt die frueheren
Einzelskripte `scripts/test_yoloe_*.py`, die dafuer je eine eigene Kopie von
Modell-Setup, Zeichnen und JSON-Ausgabe mitbrachten.

    adascope probe --image test_images/test_frame.png
    adascope probe --image test_images/test_frame_arrow.png \
                     --prompts "green lane change arrow" "green road" --conf 0.02
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from ..detection import YoloeDetector
from ..io import read_image
from ..render.primitives import FONT
from ._common import add_config_args, load_settings


def configure_parser(parser: argparse.ArgumentParser) -> None:
    add_config_args(parser)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompts", nargs="+", default=None,
                        help="Standard: model.classes aus detection.yaml")
    parser.add_argument("--checkpoint", default=None, help="ueberschreibt model.checkpoint")
    parser.add_argument("--conf", type=float, default=None, help="ueberschreibt model.conf")
    parser.add_argument("--outdir", type=Path, default=Path("outputs/probe"))


def run(args: argparse.Namespace) -> int:
    settings = load_settings(args)
    detection = settings.require_detection()
    prompts = args.prompts or detection.model.classes
    if not prompts:
        raise SystemExit("keine Prompts -- --prompts setzen oder model.classes fuellen")

    image = read_image(args.image)
    if image is None:
        raise SystemExit(f"Bild nicht lesbar: {args.image}")

    conf = args.conf if args.conf is not None else detection.model.conf
    detector = YoloeDetector(args.checkpoint or detection.model.checkpoint, list(prompts))
    boxes = detector.detect(image, conf)

    args.outdir.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem
    records = [{"label": b.label, "confidence": round(b.confidence, 5),
                "box": [round(v, 1) for v in (b.x1, b.y1, b.x2, b.y2)],
                "box_frac": round(((b.x2 - b.x1) * (b.y2 - b.y1))
                                  / (image.shape[0] * image.shape[1]), 5)}
               for b in sorted(boxes, key=lambda b: -b.confidence)]
    (args.outdir / f"{stem}.json").write_text(
        json.dumps({"image": str(args.image), "prompts": list(prompts),
                    "conf": conf, "detections": records}, indent=2),
        encoding="utf-8")
    cv2.imwrite(str(args.outdir / f"{stem}.png"), _annotate(image, boxes))

    print(f"{len(boxes)} Detektionen bei conf={conf} fuer {list(prompts)}")
    for record in records[:15]:
        print(f"  {record['confidence']:.3f}  {record['label']:<40s} "
              f"Flaeche {record['box_frac']:.3f}")
    print(f"\n  {args.outdir / (stem + '.png')}\n  {args.outdir / (stem + '.json')}")
    return 0


def _annotate(image, boxes):
    out = image.copy()
    for box in boxes:
        p0, p1 = (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2))
        cv2.rectangle(out, p0, p1, (0, 220, 255), 2)
        cv2.putText(out, f"{box.label} {box.confidence:.2f}",
                    (p0[0], max(14, p0[1] - 6)), FONT, 0.45, (0, 220, 255), 1, cv2.LINE_AA)
    return out

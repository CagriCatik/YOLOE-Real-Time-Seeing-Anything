#!/usr/bin/env python3
"""YOLOE auf einem Bild, Video oder Frame-Ordner testen.

**Offenes Vokabular**: statt fester Klassen gibt man Textprompts vor. Das ist
der Unterschied zu yolo11n/yolo11l-seg -- man kann nach Dingen suchen, die in
COCO nicht vorkommen ("grauer Fahrbereich", "Spurwechselpfeil", "Warnsymbol").

    python scripts/test_yoloe.py --source test_images/test_frame.png \
                                 --prompts "car" "truck"
    python scripts/test_yoloe.py --source test_images/test_frame_arrow.png \
                                 --prompts "gray trapezoid on the road" --conf 0.01
    python scripts/test_yoloe.py --source scenarien/acc_plus_1_vid.mp4 \
                                 --from-config --device 0
    python scripts/test_yoloe.py --source ... --prompt-free      # ohne Prompts

WAS BEIM PROMPTEN ZAEHLT -- GEMESSEN
------------------------------------
Auf den synthetischen HMI-Renderings entscheidet das FORMWORT, nicht die Farbe.
Gegen die per Helligkeit ermittelte Zielflaeche auf test_frame_arrow.png:

    gray trapezoid on the road    conf 0.019   IoU 0.82   trifft
    gray road area                conf 0.007   IoU 0.26
    gray overlay on the lane      conf 0.007   IoU 0.07   Ganzbildtreffer
    gray patch/shape/polygon/area, shaded area, driving area   kein Treffer

Zwei Konsequenzen fuer die Bedienung:
  * `--conf` muss klein sein (0.005..0.03). Die Standardschwelle 0.25 verwirft
    auf diesem Material praktisch alles.
  * Ganzbildtreffer sind der haeufigste Muell. `--max-area` filtert sie; in der
    Zusammenfassung sind sie mit "<- Ganzbild?" markiert.

YOLOE trifft diskrete Objekte gut (Schilder, Fahrzeuge) und flaechige
Farbbereiche schlecht. Fuer Letztere ist die HSV-Schwelle unter `carpet:` in
configs/detection.yaml der zuverlaessigere Weg.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from _common import REPO_ROOT, Detection, add_common_args, resolve_weights, run

DEFAULT_WEIGHTS = "yoloe-11l-seg.pt"
PROMPT_FREE_WEIGHTS = "yoloe-11l-seg-pf.pt"


def configure() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--weights", default=None,
                        help=f"Standard: {DEFAULT_WEIGHTS}, mit --prompt-free {PROMPT_FREE_WEIGHTS}")
    parser.add_argument("--prompts", nargs="+", default=None,
                        help="Textprompts; kurze Nominalphrasen treffen am besten")
    parser.add_argument("--from-config", action="store_true",
                        help="Prompts aus configs/detection.yaml nehmen "
                             "(model.classes + driving_area + hud)")
    parser.add_argument("--prompt-free", action="store_true",
                        help="prompt-freie Variante: findet, was sie kennt, ohne Vorgabe")
    parser.add_argument("--max-area", type=float, default=1.0,
                        help="Detektionen ueber diesem Bildanteil verwerfen "
                             "(0.5 filtert Ganzbildtreffer)")
    parser.add_argument("--no-masks", action="store_true")
    return parser


def main(argv=None) -> int:
    args = configure().parse_args(argv)
    if args.conf >= 0.2 and not args.prompt_free:
        print(f"Hinweis: --conf {args.conf} ist fuer YOLOE auf diesem Material hoch. "
              "Auf synthetischen HMI-Grafiken liegen die Treffer bei 0.005..0.03.")

    from ultralytics import YOLOE

    weights = args.weights or (PROMPT_FREE_WEIGHTS if args.prompt_free else DEFAULT_WEIGHTS)
    model = YOLOE(resolve_weights(weights))

    prompts: list[str] = []
    if not args.prompt_free:
        prompts = args.prompts or (config_prompts() if args.from_config else [])
        if not prompts:
            raise SystemExit(
                "Keine Prompts. --prompts \"...\" angeben, --from-config nehmen "
                "oder --prompt-free verwenden.")
        model.set_classes(prompts, model.get_text_pe(prompts))
        print(f"{len(prompts)} Prompt(s): {', '.join(prompts)}\n")

    options = {"conf": args.conf, "iou": args.iou, "imgsz": args.imgsz, "verbose": False}
    if args.device is not None:
        options["device"] = args.device

    def predict(frame):
        result = model.predict(source=frame, **options)[0]
        detections = to_detections(result, frame.shape[:2], not args.no_masks)
        if args.max_area < 1.0:
            height, width = frame.shape[:2]
            detections = [d for d in detections
                          if d.area_frac_of(width, height) <= args.max_area]
        return detections

    return run(args, "yoloe_pf" if args.prompt_free else "yoloe", predict)


def config_prompts() -> list[str]:
    """Alle Prompts aus der Projektkonfiguration einsammeln.

    So testet man genau das, was die Pipeline auch benutzt -- ohne die Liste
    von Hand abzuschreiben und dabei versehentlich zu aendern.
    """
    from adascope.config import Settings

    detection = Settings.load(REPO_ROOT / "config").require_detection()
    collected = list(detection.model.classes)
    for group in (detection.driving_area.prompts, detection.hud.prompts):
        for values in group.values():
            collected.extend(values)
    return list(dict.fromkeys(collected))          # Reihenfolge halten, Dubletten weg


def to_detections(result, shape: tuple[int, int], with_masks: bool) -> list[Detection]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    names = result.names
    masks = None
    if with_masks and getattr(result, "masks", None) is not None:
        height, width = shape
        raw = result.masks.data.cpu().numpy()
        masks = [cv2.resize(m.astype(np.uint8), (width, height),
                            interpolation=cv2.INTER_NEAREST).astype(bool) for m in raw]

    detections = []
    for index, (xyxy, conf, cls) in enumerate(zip(boxes.xyxy.cpu().numpy(),
                                                  boxes.conf.cpu().numpy(),
                                                  boxes.cls.cpu().numpy())):
        label = names.get(int(cls), str(int(cls))) if isinstance(names, dict) else names[int(cls)]
        detections.append(Detection(
            str(label), float(conf), tuple(float(v) for v in xyxy),
            mask=masks[index] if masks is not None and index < len(masks) else None))
    return detections


if __name__ == "__main__":
    raise SystemExit(main())

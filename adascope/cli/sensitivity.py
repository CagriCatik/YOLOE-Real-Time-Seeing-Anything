"""Robustheitsmessungen der BEV-Geometrie auf einem Einzelbild.

Das sind KEINE Tests -- sie quantifizieren die Grenzen der Pipeline und liefern
die Zahlen, die im README stehen. Deshalb geben sie Messwerte aus statt zu
behaupten, etwas sei richtig.

    adascope lane-sensitivity --frame data/test_frame_masked.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from ..io import read_image
from ..lanes import (
    assign_lane, build_lane_mask, corridors_from, detect_lanes, find_lane_boundaries,
    footprint_is_plausible, homography_from_pair, outer_solid_pair, project_footprint,
    restrict_to_driving_area, source_points, warp_lane_mask,
)
from ..lanes.indexing import build_lane_index
from ._common import add_config_args, load_settings

# Platzhalter fuer die Detektorausgabe; in dieser HMI-Ansicht ist das Ego
# selbst im Bild (Chase-Cam).
EGO_BOX = (565, 175, 652, 258)
OTHER_BOXES = {"left_car": (461, 120, 528, 175),
               "truck": (628, 30, 672, 78),
               "far_car": (548, 25, 600, 70)}


def configure_parser(parser: argparse.ArgumentParser) -> None:
    add_config_args(parser)
    parser.add_argument("--frame", type=Path, required=True, help="Referenz-Einzelbild")
    parser.add_argument("--trials", type=int, default=300, help="Durchlaeufe je Rauschstufe")


def run(args: argparse.Namespace) -> int:
    settings = load_settings(args)
    image = read_image(args.frame)
    if image is None:
        raise SystemExit(f"Bild nicht lesbar: {args.frame}")
    lane, bev = settings.lane, settings.bev

    pair = outer_solid_pair(detect_lanes(image, lane))
    if pair is None:
        raise SystemExit("keine zwei durchgezogenen Randlinien im Referenzbild")
    boxes = [EGO_BOX, *OTHER_BOXES.values()]
    mask = build_lane_mask(image, lane, boxes)
    mask = restrict_to_driving_area(mask, source_points(pair, lane))
    H = homography_from_pair(pair, lane, bev)

    _homography_noise(pair, mask, settings, args.trials)
    _boundary_dropout(mask, H, settings)
    _footprint_widths(mask, H, settings)
    return 0


def _homography_noise(pair, mask, settings, trials: int) -> None:
    """Wie stark stoert Rauschen auf den Linienstuetzpunkten die Zuordnung?

    Ergebnis auf dem Referenzbild: bis +-3 px bleiben Ego-Spur-Index und
    Korridoranzahl konstant. Die Geometrie ist NICHT der schwache Punkt.
    """
    lane, bev = settings.lane, settings.bev
    ls, rs = pair
    dst = np.float32([[bev.x_left, bev.y_near], [bev.x_right, bev.y_near],
                      [bev.x_right, bev.y_far], [bev.x_left, bev.y_far]])
    rng = np.random.default_rng(0)

    print("== Homographie-Rauschen ==")
    for noise in (0.0, 1.0, 2.0, 3.0):
        ego_lanes, counts = set(), set()
        for _ in range(trials):
            n = rng.normal(0, noise, 4)
            src = np.float32([
                [ls.x_at(lane.y_bottom) + n[0], lane.y_bottom],
                [rs.x_at(lane.y_bottom) + n[1], lane.y_bottom],
                [rs.x_at(lane.y_top) + n[2], lane.y_top],
                [ls.x_at(lane.y_top) + n[3], lane.y_top],
            ])
            H = cv2.getPerspectiveTransform(src, dst)
            warped = warp_lane_mask(mask, H, bev)
            corridors = corridors_from(find_lane_boundaries(warped, bev))
            counts.add(len(corridors))
            if len(corridors) >= 3:
                ego_lanes.add(assign_lane(project_footprint("ego", EGO_BOX, H), corridors)[0])
        print(f"  +-{noise:.0f} px -> Ego-Spur {sorted(ego_lanes)}  Korridoranzahl {sorted(counts)}")
    print()


def _boundary_dropout(mask, H, settings) -> None:
    """Was passiert, wenn der Fernbereich der Fahrbahn ausfaellt?

    Ursachen im Video: Strichluecke, Verdeckung, Kuppe. Faellt eine Grenze weg,
    verschmelzen zwei Korridore -- der positionsbasierte Index rutscht, die
    ego-relative Nummer aus `lanes.indexing` faengt es auf.
    """
    bev, indexing = settings.bev, settings.indexing
    print("== Spurgrenzen bei fehlendem Fernbereich ==")
    for cut in (0, 60, 120, 180, 240):
        cropped = mask.copy()
        cropped[:cut, :] = 0
        warped = warp_lane_mask(cropped, H, bev)
        boundaries = find_lane_boundaries(warped, bev)
        corridors = corridors_from(boundaries)
        ego_fp = project_footprint("ego", EGO_BOX, H)
        positional, _ = assign_lane(ego_fp, corridors) if corridors else (-1, {})
        try:
            lanes_rel, width = build_lane_index(corridors, ego_fp.x_left, ego_fp.x_right, indexing)
            relative = f"{len(lanes_rel)} Spuren, Breite {width:.0f} px"
        except (ValueError, IndexError) as exc:
            relative = f"-- ({exc})"
        print(f"  oberste {cut:3d} px entfernt -> {len(boundaries)} Grenzen, "
              f"Ego positionsbasiert L{positional} | ego-relativ: {relative}")
    print()


def _footprint_widths(mask, H, settings) -> None:
    """Plausibilitaet der rueckprojizierten Fahrzeugbreiten.

    Ein Fahrzeug kann keine Spurbreite ueberschreiten. Werte deutlich darueber
    sind Projektionsartefakte des Fernfelds und definieren die nutzbare
    Detektionsreichweite.
    """
    bev, indexing = settings.bev, settings.indexing
    warped = warp_lane_mask(mask, H, bev)
    corridors = corridors_from(find_lane_boundaries(warped, bev))
    ego_fp = project_footprint("ego", EGO_BOX, H)
    _, lane_width = build_lane_index(corridors, ego_fp.x_left, ego_fp.x_right, indexing)

    print(f"== Footprint-Plausibilitaet (Spurbreite {lane_width:.0f} px) ==")
    for name, box in {"ego": EGO_BOX, **OTHER_BOXES}.items():
        fp = project_footprint(name, box, H)
        ok = footprint_is_plausible(fp, lane_width, bev)
        print(f"  {name:9s} y={fp.y:5.1f}  Breite={fp.width:5.1f} px "
              f"= {fp.width / lane_width:.2f} Spur  [{'ok' if ok else 'ARTEFAKT'}]")
    print()

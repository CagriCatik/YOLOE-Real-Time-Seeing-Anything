"""Stufe 7 - Korridore, eigene Fahrbahn, ego-relative Spuren.

Aus den Grenzen werden Korridore, daraus die eigene Fahrbahn.

Eine Nicht-Spur-Flaeche TRENNT: alles jenseits davon gehoert zu einer
anderen Fahrbahn (Gegenrichtung, Standstreifen) und wird verworfen.
`n_korridore` gegen `n_spuren` zeigt, wie viel dabei wegfaellt.

Weichen beide stark ab, ist entweder die Spurbreitenschaetzung falsch
oder es werden Flaechen jenseits der eigenen Fahrbahn mitgezaehlt.

Stellschrauben: indexing.lane_width, multiple_tolerance, max_merge, min_corridors

    python scripts/stage_07_corridors.py --source scenarien/<name> --frames 200
"""

from __future__ import annotations

import cv2
import numpy as np

import argparse

from _stage import Stage, parse

from adascope.detection import YoloVehicleTracker
from adascope.lanes import SequencePipeline


def zusatz(ap) -> None:
    ap.add_argument("--detect", action=argparse.BooleanOptionalAction, default=None,
                    help="Fahrzeuge erkennen und Footprints einzeichnen "
                         "(--no-detect laeuft ohne Modell, dann ohne Ego-Bezug)")


def main() -> None:
    args = parse(__doc__, zusatz, stufe="stage_07_corridors")
    st = Stage(args, "stage_07_corridors",
               "indexing.lane_width, multiple_tolerance, max_merge, min_corridors")
    p = SequencePipeline(st.settings)
    # Ohne Detektor gibt es keinen Ego-Footprint -- dann faellt die Auswahl der
    # eigenen Fahrbahn auf einen Ersatzpunkt zurueck und die Korridore sind
    # nicht dieselben, die die Pipeline im Szenariolauf sieht.
    tracker = YoloVehicleTracker(st.settings.tracking) if args.detect else None
    print(f"  Detektion     {'an' if tracker else 'AUS (ohne Ego-Bezug)'}")
    for i, name, img, lane in st.frames():
        fahrzeuge = tracker.update(img) if tracker else []
        fa = p.process(i, name, img, fahrzeuge)
        st.row(frame=i, zustand=fa.h_state, n_fahrzeuge=len(fahrzeuge),
               n_grenzen=len(fa.boundaries), n_korridore=len(fa.corridors),
               n_spuren=len(fa.lanes_rel),
               n_synthetisch=sum(1 for L in fa.lanes_rel if L.synthetic),
               spurbreite=round(fa.lane_width, 1),
               ego_in_spur=round(fa.ego_in_lane, 2),
               ego_breite=round(fa.ego_footprint.width, 1) if fa.ego_footprint else 0.0,
               hinweis=fa.index_note)

        bev = st.settings.bev
        bild = np.zeros((bev.height, bev.width, 3), np.uint8)
        if fa.mask_bev is not None:
            bild = cv2.cvtColor(fa.mask_bev, cv2.COLOR_GRAY2BGR)
        for x in fa.boundaries:                     # alle gefundenen Grenzen
            cv2.line(bild, (int(x), 0), (int(x), bev.height), (90, 90, 90), 1)
        for L in fa.lanes_rel:                      # nur die eigene Fahrbahn
            farbe = (255, 255, 0) if L.rel == 0 else (0, 200, 0)
            if L.synthetic:
                farbe = (0, 170, 255)
            cv2.rectangle(bild, (int(L.x_lo), 4), (int(L.x_hi), bev.height - 4),
                          farbe, 1)
            cv2.putText(bild, f"{L.rel:+d}", (int((L.x_lo + L.x_hi) / 2) - 8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, farbe, 1, cv2.LINE_AA)
        # Footprints: die auf den Boden projizierten Bbox-Unterkanten. NUR sie
        # liegen in der Bodenebene -- Fahrzeuge haben Bauhoehe und verlaufen im
        # BEV radial. Deshalb ein Segment, kein Rechteck.
        y_fuss = bev.height - 18
        if fa.ego_footprint is not None:
            e = fa.ego_footprint
            cv2.line(bild, (int(e.x_left), int(e.y)), (int(e.x_right), int(e.y)),
                     (255, 255, 0), 3)
            cv2.putText(bild, f"EGO {fa.ego_in_lane:.2f}", (int(e.x_left), int(e.y) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
        for occ in fa.occupancies:
            f = occ.footprint
            farbe = (0, 255, 255) if occ.valid else (140, 140, 140)
            cv2.line(bild, (int(f.x_left), int(f.y)), (int(f.x_right), int(f.y)),
                     farbe, 2)
            rel = "?" if occ.rel is None else f"{occ.rel:+d}"
            cv2.putText(bild, f"{occ.track} {rel} {occ.state or ''}",
                        (int(f.x_left), int(f.y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, farbe, 1, cv2.LINE_AA)
        cv2.putText(bild, f"{len(fahrzeuge)} Fahrzeuge", (6, y_fuss),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
        st.show(i, bild, f"f{i}  {len(fa.corridors)} Korridore -> "
                         f"{len(fa.lanes_rel)} Spuren  {fa.index_note}")
    st.finish()


if __name__ == "__main__":
    main()

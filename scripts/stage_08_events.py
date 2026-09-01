"""Stufe 8 - State Machine und Ereignisse.

Die letzte Stufe. Sie braucht die Fahrzeugdetektion -- ohne sie gibt
es keine Fremdfahrzeuge und damit kein cut_in/cut_out.

Die Spalte `zustand_je_track` zeigt outside/encroaching/inside je
verfolgtem Fahrzeug. Ereignisse entstehen nur an Uebergaengen; ein
Fahrzeug, das beim Start schon drin ist, erzeugt keines.

Stellschrauben: events.thr_inside, thr_encroaching, confirm_frames, require_direction

    python scripts/stage_08_events.py --source scenarien/<name> --frames 200
"""

from __future__ import annotations

import cv2
import numpy as np

from _stage import Stage, parse

from adascope.detection import YoloVehicleTracker
from adascope.lanes import SequencePipeline


def main() -> None:
    args = parse(__doc__, stufe="stage_08_events")
    st = Stage(args, "stage_08_events",
               "events.thr_inside, thr_encroaching, confirm_frames, "
               "require_direction")
    p = SequencePipeline(st.settings)
    tracker = YoloVehicleTracker(st.settings.tracking)
    alle = []
    for i, name, img, lane in st.frames():
        fahrzeuge = tracker.update(img)
        fa = p.process(i, name, img, fahrzeuge)
        alle.extend(fa.events)
        st.row(frame=i, n_fahrzeuge=len(fahrzeuge),
               n_belegungen=len(fa.occupancies),
               ego_in_spur=round(fa.ego_in_lane, 2),
               n_ereignisse=len(fa.events),
               zustand_je_track="|".join(
                   f"{o.track}:{o.state or '?'}" for o in fa.occupancies))
        for e in fa.events:
            print(f"    {e}")

        # Bildebene mit Zustandsfarben: hier entstehen die Ereignisse, also
        # muss hier auch nachvollziehbar sein, warum.
        bild = img.copy()
        zustandsfarbe = {"outside": (0, 210, 0), "encroaching": (0, 170, 255),
                         "inside": (0, 0, 255), None: (150, 150, 150)}
        belegung = {o.track: o for o in fa.occupancies}
        for v in fahrzeuge:
            occ = belegung.get(v.track_id if hasattr(v, "track_id") else None)
            farbe = zustandsfarbe.get(occ.state if occ else None, (150, 150, 150))
            x1, y1, x2, y2 = v.bbox
            cv2.rectangle(bild, (int(x1), int(y1)), (int(x2), int(y2)), farbe, 2)
            if occ is not None:
                rel = "?" if occ.rel is None else f"{occ.rel:+d}"
                cv2.putText(bild, f"{occ.track} {rel} {occ.state or ''}",
                            (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, farbe, 1, cv2.LINE_AA)
        for k, e in enumerate(alle[-4:]):          # laufendes Ereignisprotokoll
            cv2.putText(bild, str(e), (8, bild.shape[0] - 52 + k * 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 255), 1,
                        cv2.LINE_AA)
        st.show(i, bild, f"f{i}  {len(fahrzeuge)} Fahrzeuge  "
                         f"Ego in Spur {fa.ego_in_lane:.2f}")
    st.finish()
    print(f"\n  {len(alle)} Ereignisse gesamt")
    for e in alle:
        print(f"    {e}")


if __name__ == "__main__":
    main()

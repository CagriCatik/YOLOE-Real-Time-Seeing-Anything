"""Gemeinsamer Unterbau der Einzelschritt-Skripte der Spurpipeline.

Wozu Einzelschritte
-------------------
`adascope scenarios` laesst die ganze Kette laufen und sagt am Ende, wie gut
es lief. Wenn eine Stufe schlecht arbeitet, sagt es nicht WELCHE. Die Skripte
`stage_01` bis `stage_08` schneiden je eine Stufe heraus: gleiche Quelle,
gleiche Konfiguration, aber Ausgabe genau eines Zwischenstands -- als Bild,
als Video und als CSV.

Damit ist jede Stufe einzeln zu beurteilen und einzeln zu kalibrieren:

    stage_01_mask.py         Weissmaske und ROI          lane.white_l_min, roi_polygon
    stage_02_segments.py     Kanten und Hough-Segmente   lane.hough_*
    stage_03_lines.py        Cluster, Linienfit, Rollen  lane.cluster_*, robust_*
    stage_04_homography.py   Randlinienpaar und H        lane.y_top/y_bottom, bev.*
    stage_05_bev.py          gewarpte Spurmaske          bev.width/height
    stage_06_histogram.py    Spaltenhistogramm, Peaks    bev.peak_*, histogram_blur
    stage_07_corridors.py    Korridore, eigene Fahrbahn  indexing.*
    stage_08_events.py       State Machine               events.*

Jedes Skript nennt in seiner Ausgabe die Konfigurationsschluessel, die es
beeinflussen -- damit klar ist, an welcher Schraube zu drehen ist.

Alle teilen dieselben Argumente:

    --source   Video oder Bildordner (Standard: erstes Szenario)
    --config   Konfigurationsordner (Standard: scripts/configs)
    --frames   Anzahl Frames (0 = alle)
    --out      Ausgabeordner (Standard: results/stages/<stufe>)
    --video / --no-video   Debugvideo schreiben
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adascope.config import Settings           # noqa: E402
from adascope.config.loader import read_yaml   # noqa: E402
from adascope.io import iter_source            # noqa: E402
from adascope.runner import choose_crop, crop_frame   # noqa: E402

SKRIPT_CONFIG = Path(__file__).resolve().parent / "configs" / "scripts.yaml"
EINGEBAUT = {"source": "scenarien/lane_departure_3_lanes",
             "config": "scripts/configs",
             "frames": 150, "fps": 25.0, "video": True,
             "snapshots": 3, "out_root": "results/stages"}


def voreinstellungen(stufe: str) -> dict:
    """defaults + Abschnitt der Stufe aus `scripts/configs/scripts.yaml`.

    Fehlt die Datei, gelten die eingebauten Werte -- die Skripte bleiben ohne
    sie lauffaehig. Ein CLI-Flag gewinnt in jedem Fall.
    """
    werte = dict(EINGEBAUT)
    if not SKRIPT_CONFIG.exists():
        return werte
    roh = read_yaml(SKRIPT_CONFIG) or {}
    werte.update(roh.get("defaults") or {})
    werte.update((roh.get("stages") or {}).get(stufe) or {})
    return werte


def parse(beschreibung: str, extra=None, stufe: str = "") -> argparse.Namespace:
    """Argumente einlesen; nicht gesetzte Flags kommen aus scripts.yaml.

    Die Voreinstellungen stehen absichtlich NICHT im argparse-`default`,
    sondern werden danach eingesetzt: nur so ist unterscheidbar, ob ein Wert
    weggelassen oder bewusst auf den Vorgabewert gesetzt wurde.
    """
    vor = voreinstellungen(stufe)
    ap = argparse.ArgumentParser(description=beschreibung,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=None, help="Video oder Bildordner")
    ap.add_argument("--config", default=None, help="Konfigurationsordner")
    ap.add_argument("--frames", type=int, default=None,
                    help="Anzahl Frames, 0 = alle")
    ap.add_argument("--out", default=None, help="Ausgabeordner")
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--video", action=argparse.BooleanOptionalAction, default=None,
                    help="Debugvideo schreiben")
    ap.add_argument("--snapshots", type=int, default=None,
                    help="Anzahl Frames mit benannten Zwischenbildern")
    if extra:
        extra(ap)
    args = ap.parse_args()
    # Was auf None steht, hat die Kommandozeile nicht gesetzt -- genau das wird
    # aus der Datei gefuellt und danach als solches ausgewiesen.
    aus_datei = []
    for name, wert in vor.items():
        if name == "out_root":
            continue
        if getattr(args, name, None) is None:
            setattr(args, name, wert)
            aus_datei.append(name)
    args.out_root = vor["out_root"]
    args.aus_config = aus_datei
    return args


class Stage:
    """Rahmen einer Stufe: Frames liefern, Ausgabe sammeln, Ergebnis schreiben."""

    def __init__(self, args: argparse.Namespace, name: str,
                 schluessel: str = "") -> None:
        if args.snapshots < 0:
            raise ValueError("--snapshots muss mindestens 0 sein")
        self.args = args
        self.name = name
        self.schluessel = schluessel
        self.settings = Settings.load(args.config)
        self.out = Path(args.out or f"{getattr(args, 'out_root', 'results/stages')}/{name}")
        self.out.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict] = []
        self._writer: cv2.VideoWriter | None = None
        self._crop = None
        print(f"\n=== {name} ===")
        print(f"  Quelle        {args.source}")
        print(f"  Konfiguration {args.config}")
        print(f"  Frames        {args.frames or 'alle'}")
        if schluessel:
            print(f"  Stellschrauben {schluessel}")
        if getattr(args, "aus_config", None):
            print(f"  (aus scripts/configs/scripts.yaml: "
                  f"{', '.join(sorted(args.aus_config))})")

    def frames(self):
        """(index, name, zugeschnittenes Bild, auf die Groesse gezogene LaneConfig)."""
        quelle, _ = iter_source(self.args.source, 1, self.args.fps)
        grenze = self.args.frames or 10**9
        for i, (name, img) in enumerate(quelle):
            if i >= grenze:
                break
            if self._crop is None:
                self._crop = choose_crop(self.settings, img.shape[1],
                                         img.shape[0]) or False
            if self._crop:
                img = crop_frame(img, self._crop)
            lane = self.settings.lane.scaled_to(img.shape[1], img.shape[0])
            yield i, name, img, lane

    def show(self, index: int, bild: np.ndarray, titel: str = "") -> None:
        """Ein Einzelbild ins Video und die ersten ``--snapshots`` als PNG."""
        bild = self._bgr(bild)
        if titel:
            cv2.putText(bild, titel, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 1, cv2.LINE_AA)
        if index < self.args.snapshots:
            cv2.imwrite(str(self.out / f"frame_{index:04d}.png"), bild)
        if not self.args.video:
            return
        if self._writer is None:
            h, w = bild.shape[:2]
            self._writer = cv2.VideoWriter(
                str(self.out / f"{self.name}.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"), self.args.fps, (w, h))
        self._writer.write(bild)

    @staticmethod
    def _bgr(bild: np.ndarray) -> np.ndarray:
        """Debugbilder vereinheitlichen, ohne das Rechenergebnis zu beschriften."""
        if bild.ndim == 2:
            return cv2.cvtColor(bild, cv2.COLOR_GRAY2BGR)
        return bild.copy()

    def snapshot(self, index: int, name: str, bild: np.ndarray,
                 titel: str = "") -> None:
        """Benannten Zwischenstand fuer die ersten ``--snapshots`` Frames sichern.

        ``show`` bleibt genau ein Videoframe je Eingabeframe. Diese Methode ist
        absichtlich nur fuer PNGs: eine Stufe darf so viele erklaerende
        Zwischenbilder liefern, ohne Bildrate oder Videolaenge zu verfaelschen.
        """
        if index >= self.args.snapshots:
            return
        out = self._bgr(bild)
        if titel:
            cv2.rectangle(out, (0, 0), (out.shape[1], 27), (0, 0, 0), -1)
            cv2.putText(out, titel, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 1, cv2.LINE_AA)
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        cv2.imwrite(str(self.out / f"frame_{index:04d}_{safe}.png"), out)

    def row(self, **werte) -> None:
        self.rows.append(werte)

    def finish(self) -> None:
        if self._writer is not None:
            self._writer.release()
        if self.rows:
            felder = list(self.rows[0])
            with open(self.out / f"{self.name}.csv", "w", newline="",
                      encoding="utf-8") as fh:
                schreiber = csv.DictWriter(fh, fieldnames=felder)
                schreiber.writeheader()
                schreiber.writerows(self.rows)
            self._zusammenfassung(felder)
        print(f"\n  Ausgabe       {self.out}")

    def _zusammenfassung(self, felder: list[str]) -> None:
        print(f"\n  {len(self.rows)} Frames")
        for feld in felder:
            werte = [r[feld] for r in self.rows if isinstance(r[feld], (int, float))]
            if len(werte) < 2 or feld == "frame":
                continue
            a = np.array(werte, float)
            print(f"    {feld:<22s} Median {np.median(a):>8.2f}   "
                  f"Spanne {a.min():>8.2f} .. {a.max():>8.2f}")

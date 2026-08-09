"""Gemeinsamer Unterbau der Modell-Testskripte.

Warum das hier steht und nicht dreimal daneben
-----------------------------------------------
Der frühere `scripts/`-Ordner wurde aufgelöst, weil jedes Skript eine eigene
Kopie von Modell-Setup, Zeichnen und Ausgabe mitbrachte (siehe ADR-0002). Diese
Datei ist die Antwort darauf: alles, was die drei Skripte teilen, steht genau
einmal hier. Ein Skript enthält nur noch das, was sein Modell besonders macht.

Abgrenzung zur Pipeline
-----------------------
Diese Skripte sind bewusst **außerhalb** von `adascope`. Sie beantworten
"was sieht dieses Modell auf diesem Bild", nicht "wo ist die Spur". Keine
Kalibrierung, keine Homographie, keine Zustandslogik — nur Modell rein,
Detektionen raus. Wer die Spurpipeline testen will, nimmt `adascope scenarios`.

Geteilt wird nur die I/O-Schicht von `adascope` (Frames, Videos, Tabellen),
damit auch hier nicht ein viertes Mal `imread`-Sonderfälle entstehen.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:            # auch ohne `pip install -e .` lauffähig
    sys.path.insert(0, str(REPO_ROOT))

from adascope.io import IMAGE_EXTENSIONS, VideoWriter, is_video, iter_source, write_named

MODELS_DIR = REPO_ROOT / "models"
DEFAULT_OUTDIR = REPO_ROOT / "outputs" / "model_tests"
FONT = cv2.FONT_HERSHEY_SIMPLEX
CSV_FIELDS = ["frame", "source", "label", "confidence", "track_id",
              "x1", "y1", "x2", "y2", "area_frac", "has_mask"]


# --------------------------------------------------------------------------- #
# Einheitliches Ergebnis über alle drei Modelle                               #
# --------------------------------------------------------------------------- #
@dataclass
class Detection:
    label: str
    confidence: float
    box: tuple[float, float, float, float]
    track_id: int | None = None
    mask: np.ndarray | None = None          # (H, W) bool, nur bei -seg-Modellen

    @property
    def area_frac_of(self):
        def compute(width: int, height: int) -> float:
            x1, y1, x2, y2 = self.box
            return max(0.0, (x2 - x1) * (y2 - y1)) / (width * height)
        return compute


# --------------------------------------------------------------------------- #
# Argumente                                                                   #
# --------------------------------------------------------------------------- #
def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", required=True, type=Path,
                        help="Bild, Video oder Ordner mit Bildern")
    parser.add_argument("--conf", type=float, default=0.25, help="Konfidenzschwelle")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default=None, help="z.B. 0 oder cpu")
    parser.add_argument("--iou", type=float, default=0.50, help="NMS-Schwelle")
    parser.add_argument("--max-frames", type=int, default=0, help="0 = alle")
    parser.add_argument("--stride", type=int, default=1, help="jeden n-ten Frame")
    parser.add_argument("--fps", type=float, default=25.0,
                        help="Ausgabe-Bildrate für Bildordner")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--no-labels", action="store_true",
                        help="Boxen ohne Beschriftung zeichnen")
    parser.add_argument("--top", type=int, default=15,
                        help="wie viele Klassen in der Zusammenfassung")


def resolve_weights(name_or_path: str) -> str:
    """Gewichte finden, sonst nach `models/` laden.

    Ultralytics lädt fehlende Gewichte sonst ins ARBEITSVERZEICHNIS — dann
    liegen 70-MB-Dateien im Projektwurzelverzeichnis statt bei den anderen.
    """
    path = Path(name_or_path)
    if path.exists():
        return str(path)
    target = path if path.parent != Path(".") else MODELS_DIR / path.name
    if target.exists():
        return str(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Gewichte fehlen, lade nach {target} …")
    from ultralytics.utils.downloads import attempt_download_asset
    attempt_download_asset(str(target))
    return str(target)


# --------------------------------------------------------------------------- #
# Quelle                                                                      #
# --------------------------------------------------------------------------- #
def frame_source(source: Path, stride: int, fps: float):
    """(Frame-Iterator, fps, ist_einzelbild).

    `adascope.io.iter_source` kennt Videos und Ordner; ein einzelnes Bild ist
    hier der häufigste Fall und wird deshalb ergänzt.
    """
    source = Path(source)
    if source.is_file() and source.suffix.lower() in IMAGE_EXTENSIONS:
        image = cv2.imread(str(source))
        if image is None:
            raise SystemExit(f"Bild nicht lesbar: {source}")
        return iter([(source.name, image)]), fps, True
    frames, source_fps = iter_source(source, stride, fps)
    return frames, source_fps, False


def limited(frames, stride: int, max_frames: int):
    processed = 0
    for index, (name, frame) in enumerate(frames):
        if index % max(stride, 1):
            continue
        if max_frames and processed >= max_frames:
            return
        yield index, name, frame
        processed += 1


# --------------------------------------------------------------------------- #
# Zeichnen                                                                    #
# --------------------------------------------------------------------------- #
def color_for(label: str) -> tuple[int, int, int]:
    """Feste Farbe je Klassenname -- gleiche Klasse, gleiche Farbe über Läufe."""
    h = abs(hash(label))
    return (60 + h % 180, 60 + (h >> 8) % 180, 60 + (h >> 16) % 180)


def draw(frame: np.ndarray, detections: list[Detection], show_labels: bool = True,
         mask_alpha: float = 0.40) -> np.ndarray:
    out = frame.copy()
    with_masks = [d for d in detections if d.mask is not None]
    if with_masks:
        overlay = out.copy()
        for det in with_masks:
            overlay[det.mask] = color_for(det.label)
        out = cv2.addWeighted(overlay, mask_alpha, out, 1 - mask_alpha, 0)
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det.box)
        color = color_for(det.label)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        if show_labels:
            identity = f"#{det.track_id} " if det.track_id is not None else ""
            text = f"{identity}{det.label} {det.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(text, FONT, 0.45, 1)
            cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
            cv2.putText(out, text, (x1 + 2, max(10, y1 - 4)), FONT, 0.45,
                        (0, 0, 0), 1, cv2.LINE_AA)
    return out


# --------------------------------------------------------------------------- #
# Lauf                                                                        #
# --------------------------------------------------------------------------- #
def run(args: argparse.Namespace, title: str, predict) -> int:
    """Über die Quelle laufen, zeichnen, schreiben, berichten.

    `predict(frame) -> list[Detection]` ist das Einzige, was die drei Skripte
    unterscheidet.
    """
    frames, fps, single = frame_source(args.source, args.stride, args.fps)
    stem = Path(args.source).stem
    args.outdir.mkdir(parents=True, exist_ok=True)
    writer = None if single else VideoWriter(
        args.outdir / f"{stem}_{title}.mp4", fps / max(args.stride, 1))

    rows: list[dict] = []
    per_frame: list[int] = []
    timings: list[float] = []
    try:
        for index, name, frame in limited(frames, args.stride, args.max_frames):
            height, width = frame.shape[:2]
            started = time.perf_counter()
            detections = predict(frame)
            timings.append((time.perf_counter() - started) * 1000)

            annotated = draw(frame, detections, not args.no_labels)
            if single:
                cv2.imwrite(str(args.outdir / f"{stem}_{title}.png"), annotated)
            else:
                writer.write(annotated)

            per_frame.append(len(detections))
            for det in detections:
                x1, y1, x2, y2 = det.box
                rows.append({
                    "frame": index, "source": name, "label": det.label,
                    "confidence": round(det.confidence, 5),
                    "track_id": "" if det.track_id is None else det.track_id,
                    "x1": round(x1, 1), "y1": round(y1, 1),
                    "x2": round(x2, 1), "y2": round(y2, 1),
                    "area_frac": round(det.area_frac_of(width, height), 5),
                    "has_mask": int(det.mask is not None),
                })
            if not single and (len(per_frame) == 1 or len(per_frame) % 50 == 0):
                print(f"  {len(per_frame):5d} Frames | {len(detections)} Detektionen "
                      f"| {timings[-1]:.0f} ms")
    finally:
        if writer is not None:
            writer.close()

    csv_path = args.outdir / f"{stem}_{title}.csv"
    write_named(rows, csv_path, CSV_FIELDS)
    json_path = args.outdir / f"{stem}_{title}.json"
    json_path.write_text(json.dumps(
        {"model": title, "source": str(args.source), "conf": args.conf,
         "frames": len(per_frame), "detections": rows}, indent=2), encoding="utf-8")

    report(title, rows, per_frame, timings, args.top)
    print("\nAusgaben:")
    if single:
        print(f"  {args.outdir / f'{stem}_{title}.png'}")
    else:
        print(f"  {writer.path}  {writer.size}")
    print(f"  {csv_path}\n  {json_path}")
    return 0


def report(title: str, rows: list[dict], per_frame: list[int],
           timings: list[float], top: int) -> None:
    """Die Zahlen, die man beim Modellvergleich tatsächlich braucht."""
    print(f"\n== {title} ==")
    if not per_frame:
        print("Keine Frames verarbeitet.")
        return
    print(f"Frames                {len(per_frame)}")
    print(f"Detektionen           {len(rows)} gesamt, "
          f"{len(rows) / len(per_frame):.1f} je Frame "
          f"(min {min(per_frame)}, max {max(per_frame)})")
    if timings:
        ordered = sorted(timings)
        print(f"Inferenz              {sum(timings) / len(timings):.0f} ms im Mittel, "
              f"p90 {ordered[int(len(ordered) * 0.9)]:.0f} ms "
              f"({1000 / max(sum(timings) / len(timings), 1e-6):.1f} fps)")
    if not rows:
        print("\nNichts erkannt -- --conf senken oder die Prompts prüfen.")
        return

    counts = Counter(r["label"] for r in rows)
    print(f"\n{'Klasse':<40s} {'Anzahl':>7s} {'conf med':>9s} {'Fläche med':>11s}")
    print("-" * 70)
    for label, count in counts.most_common(top):
        confs = sorted(r["confidence"] for r in rows if r["label"] == label)
        areas = sorted(r["area_frac"] for r in rows if r["label"] == label)
        marker = "  <- Ganzbild?" if areas[len(areas) // 2] > 0.5 else ""
        print(f"{label:<40s} {count:>7d} {confs[len(confs) // 2]:>9.3f} "
              f"{areas[len(areas) // 2]:>11.4f}{marker}")
    if len(counts) > top:
        print(f"… und {len(counts) - top} weitere Klassen")

    tracks = {r["track_id"] for r in rows if r["track_id"] != ""}
    if tracks:
        print(f"\nTrack-IDs             {len(tracks)} verschiedene")

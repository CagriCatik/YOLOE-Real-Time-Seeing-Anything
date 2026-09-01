"""Gemeinsame CLI-Bausteine.

Jedes Kommando, das die Spur-Pipeline benutzt, nimmt dieselben drei Argumente
entgegen: woher die Konfiguration kommt, welches Szenario sie ueberlagert und
woher die Frames stammen. Sie hier einmal zu definieren haelt die Kommandos
duenn und die Bedienung ueber alle Kommandos hinweg gleich.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..config import DEFAULT_CONFIG_DIR, Settings

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR,
                        help="Verzeichnis mit den Domaenen-YAMLs (Standard: configs)")
    parser.add_argument("--scenario", default=None,
                        help="Overlay aus <config-dir>/scenarios/<name>.yaml")


def add_source_args(parser: argparse.ArgumentParser, default: Path | None = None) -> None:
    parser.add_argument("--source", type=Path, default=default,
                        help="Videodatei oder Ordner mit Frames")
    parser.add_argument("--fps", type=float, default=25.0,
                        help="Ausgabe-Bildrate fuer Bildordner (Videos bringen ihre eigene mit)")
    parser.add_argument("--stride", type=int, default=1, help="jeden n-ten Frame")
    parser.add_argument("--max-frames", type=int, default=0, help="0 = alle")


def add_crop_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--no-crop", action="store_true",
                        help="unbeschnittene Aufnahmen NICHT automatisch auf "
                             "detection.crop_box zuschneiden")


def add_tracking_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--no-detect", action="store_true",
                        help="ohne YOLO -- nur Geometrie, Ego aus der Kalibrierung")
    parser.add_argument("--weights", default=None, help="ueberschreibt tracking.weights")
    parser.add_argument("--device", default=None, help="z.B. 0 oder cpu")
    parser.add_argument("--conf", type=float, default=None,
                        help="ueberschreibt tracking.confidence")


def load_settings(args: argparse.Namespace) -> Settings:
    return Settings.load(args.config_dir, getattr(args, "scenario", None))


def build_tracker(args: argparse.Namespace, settings: Settings):
    """None bei `--no-detect`; sonst ein YoloVehicleTracker mit CLI-Overrides.

    Der Import liegt absichtlich in der Funktion: ohne Detektion soll kein
    `ultralytics` geladen werden, sonst kostet ein reiner Geometrielauf
    Sekunden und ein GPU-Kontext.
    """
    if args.no_detect:
        return None
    from ..detection import YoloVehicleTracker
    from dataclasses import replace

    config = settings.tracking
    if args.device is not None:
        config = replace(config, device=args.device)
    if args.conf is not None:
        config = replace(config, confidence=args.conf)
    weights = args.weights or settings.weights_path()
    return YoloVehicleTracker(config, weights=weights)


def iter_limited(frames, stride: int, max_frames: int):
    """(index, name, frame) mit Schrittweite und Obergrenze.

    Der Index bleibt der Index der QUELLE, nicht der Zaehler der verarbeiteten
    Frames -- sonst passen die Zeitstempel in CSV und Video nicht mehr zur
    Vorlage, sobald `--stride` gesetzt ist.
    """
    processed = 0
    for index, (name, frame) in enumerate(frames):
        if index % max(stride, 1):
            continue
        if max_frames and processed >= max_frames:
            return
        yield index, name, frame
        processed += 1

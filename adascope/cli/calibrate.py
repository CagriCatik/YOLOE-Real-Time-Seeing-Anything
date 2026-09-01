"""Spurgeometrie automatisch aus dem Material bestimmen (FR-6).

Liest eine Aufnahme, misst die Referenzgroessen und schlaegt Config-Werte vor.
Ohne ``--apply`` wird nichts an der Konfiguration geaendert. Mit ``--apply``
werden nur stabile, automatisch geeignete Werte mit Backup geschrieben.

    adascope calibrate --source scenarien/acc_plus_1_vid.mp4
    adascope calibrate --source scenarien/acc_plus_1 --max-frames 500
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..calibration import apply_calibration, calibrate
from ._common import add_config_args, load_settings


def configure_parser(parser: argparse.ArgumentParser) -> None:
    add_config_args(parser)
    parser.add_argument("--source", required=True, type=Path,
                        help="Video oder Frame-Ordner des Fahrzeugprojekts")
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--detect", action=argparse.BooleanOptionalAction, default=True,
                        help="Fahrzeuge vor der BEV-Messung aus der Maske entfernen")
    parser.add_argument("--report", type=Path, default=None,
                        help="maschinenlesbaren YAML-Bericht schreiben")
    parser.add_argument("--apply", action="store_true",
                        help="stabile Werte mit Backup nach <config-dir> schreiben")


def run(args: argparse.Namespace) -> int:
    settings = load_settings(args)
    tracker = None
    if args.detect:
        from ..detection import YoloVehicleTracker
        tracker = YoloVehicleTracker(settings.tracking, weights=settings.weights_path())
    result = calibrate(args.source, settings, args.max_frames, args.stride, print,
                       tracker=tracker)
    print()
    print(result.as_text())

    fragments = result.as_yaml_fragments()
    automatic = result.as_yaml_fragments(auto_only=True)
    if not fragments:
        print("Keine uebernehmbaren Werte -- zu wenig verwertbares Material.")
        if args.report:
            print(f"Bericht geschrieben: {result.write_report(args.report)}")
        return 1
    print("Messvorschlag (Geometrie und instabile Werte vorher manuell pruefen):\n")
    for domain, values in fragments.items():
        print(f"  # {args.config_dir / f'{domain}.yaml'}")
        if len(fragments) > 1:
            print(f"  {domain}:")
        for key, value in values.items():
            indent = "    " if len(fragments) > 1 else "  "
            print(f"{indent}{key}: {value}")
        print()
    print(f"Automatisch anwendbar: {automatic or 'nichts'}\n")
    if args.report:
        print(f"Bericht geschrieben: {result.write_report(args.report)}")

    if args.apply:
        backups = apply_calibration(result, args.config_dir)
        print(f"Kalibrierung angewandt auf {args.config_dir}")
        for backup in backups:
            print(f"  Backup: {backup}")
    else:
        print("Trockenlauf: mit --apply werden nur stabile Auto-Werte uebernommen.")
    return 0 if result.usable else 1

"""Spurgeometrie automatisch aus dem Material bestimmen (FR-6).

Liest eine Aufnahme, misst die Referenzgroessen und schlaegt Config-Werte vor.
Geschrieben wird NICHTS -- eine Kalibrierung, die sich selbst einspielt,
aendert das Verhalten ohne Spur in der Historie.

    adascope calibrate --source scenarien/acc_plus_1_vid.mp4
    adascope calibrate --source scenarien/acc_plus_1 --max-frames 500
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..calibration import calibrate
from ._common import add_config_args, load_settings


def configure_parser(parser: argparse.ArgumentParser) -> None:
    add_config_args(parser)
    parser.add_argument("--source", required=True, type=Path,
                        help="Video oder Frame-Ordner des Fahrzeugprojekts")
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--stride", type=int, default=3)


def run(args: argparse.Namespace) -> int:
    settings = load_settings(args)
    result = calibrate(args.source, settings, args.max_frames, args.stride, print)
    print()
    print(result.as_text())

    fragments = result.as_yaml_fragments()
    if not fragments:
        print("Keine uebernehmbaren Werte -- zu wenig verwertbares Material.")
        return 1
    print("Vorschlag zum Uebernehmen:\n")
    for domain, values in fragments.items():
        print(f"  # config/{domain}.yaml   (oder config/scenarios/<name>.yaml)")
        if len(fragments) > 1:
            print(f"  {domain}:")
        for key, value in values.items():
            indent = "    " if len(fragments) > 1 else "  "
            print(f"{indent}{key}: {value}")
        print()
    return 0 if result.usable else 1

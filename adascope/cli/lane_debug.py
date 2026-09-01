"""Debug-Videos der Spur-Pipeline fuer EINE Quelle.

Fuer Szenarien aus `scenarien/` ist `adascope scenarios` der bequemere Weg --
dieses Kommando ist der direkte Zugriff auf beliebige Quellen und Ausgabeorte.

    adascope lane-debug --source data/frames/xyz --views all
    adascope lane-debug --source video.mp4 --scenario dreispurig --views dash
    adascope lane-debug --source data/frames/xyz --views bev,hist --no-detect
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..render import available_views
from ..runner import resolve_views, run_debug
from ._common import (
    add_config_args, add_crop_arg, add_source_args, add_tracking_args, build_tracker,
    load_settings,
)


def configure_parser(parser: argparse.ArgumentParser) -> None:
    add_config_args(parser)
    add_source_args(parser)
    add_tracking_args(parser)
    add_crop_arg(parser)
    parser.add_argument("--views", default="dash",
                        help="kommasepariert oder 'all' (siehe --list-views)")
    parser.add_argument("--list-views", action="store_true",
                        help="verfuegbare Ansichten anzeigen und beenden")
    parser.add_argument("--outdir", type=Path, default=Path("results/lane-debug"))


def run(args: argparse.Namespace) -> int:
    settings = load_settings(args)
    if args.list_views:
        print("Verfuegbare Ansichten:", ", ".join(available_views(settings)))
        print("Weitere entstehen durch einen Eintrag unter `cameras:` in configs/debug.yaml.")
        return 0
    if args.source is None:
        raise SystemExit("--source fehlt (Videodatei oder Frame-Ordner)")

    try:
        views = resolve_views(args.views, settings)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None

    summary = run_debug(
        args.source, settings, views, args.outdir,
        tracker=build_tracker(args, settings), stride=args.stride,
        max_frames=args.max_frames, fps=args.fps,
        auto_crop=not args.no_crop, on_progress=print)

    print("\n" + summary.as_text())
    return 0

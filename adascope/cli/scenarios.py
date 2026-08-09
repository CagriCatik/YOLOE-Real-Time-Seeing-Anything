"""Szenarien aus `scenarien/` ausfuehren und nach `results/` schreiben.

Der Einstiegspunkt fuer den Alltag: eine Aufnahme ablegen, ein Kommando, fertig.

    adascope scenarios --list                  # was ist da?
    adascope scenarios                         # alle laufen lassen
    adascope scenarios lane_departure_3_lanes  # nur dieses eine
    adascope scenarios --views dash --quick    # schneller Blick, 100 Frames

Je Szenario entsteht `results/<name>/` mit allen Debug-Videos, den CSVs und
einer `summary.txt`. Darueber liegt `results/index.md` mit einer Vergleichs-
tabelle ueber alle Laeufe -- der Regressionsblick.

Findet sich `config/scenarios/<name>.yaml`, wird sie automatisch als
Kalibrier-Ueberlagerung verwendet; sonst gilt die Basiskalibrierung.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..runner import resolve_views, run_debug
from ..scenarios import (
    DEFAULT_RESULT_DIR, DEFAULT_SCENARIO_DIR, RunSummary, discover, render_table,
    resolve,
)
from ._common import add_config_args, add_crop_arg, add_tracking_args, build_tracker

QUICK_FRAMES = 100


def configure_parser(parser: argparse.ArgumentParser) -> None:
    add_config_args(parser)
    add_tracking_args(parser)
    add_crop_arg(parser)
    parser.add_argument("names", nargs="*",
                        help="Szenarionamen; ohne Angabe laufen alle")
    parser.add_argument("--list", action="store_true", dest="list_only",
                        help="gefundene Szenarien anzeigen und beenden")
    parser.add_argument("--scenario-dir", type=Path, default=DEFAULT_SCENARIO_DIR)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--ground-truth", type=Path, default=Path("ground_truth"),
                        help="Verzeichnis mit den Sollereignissen je Szenario")
    parser.add_argument("--views", default="all",
                        help="kommasepariert oder 'all' (Standard)")
    parser.add_argument("--quick", action="store_true",
                        help=f"nur die ersten {QUICK_FRAMES} Frames je Szenario")
    parser.add_argument("--max-frames", type=int, default=0, help="0 = alle")
    parser.add_argument("--stride", type=int, default=1, help="jeden n-ten Frame")
    parser.add_argument("--fps", type=float, default=25.0,
                        help="Bildrate fuer Frame-Ordner (Videos bringen ihre eigene mit)")


def run(args: argparse.Namespace) -> int:
    found = discover(args.scenario_dir, args.config_dir)
    if args.list_only or not found:
        return _list(found, args.scenario_dir)

    try:
        scenarios = resolve(args.names, args.scenario_dir, args.config_dir)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None

    max_frames = QUICK_FRAMES if args.quick else args.max_frames
    # Der Tracker wird EINMAL gebaut und ueber alle Szenarien wiederverwendet:
    # das Modell zu laden dauert laenger als ein kurzes Szenario zu rechnen.
    tracker = build_tracker(args, scenarios[0].settings(args.config_dir))

    summaries: list[RunSummary] = []
    for number, scenario in enumerate(scenarios, 1):
        settings = scenario.settings(args.config_dir)
        outdir = scenario.result_dir(args.results)
        overlay = f"config/scenarios/{scenario.name}.yaml" if scenario.config_overlay \
            else "Basiskalibrierung"
        print(f"\n[{number}/{len(scenarios)}] {scenario.name}")
        print(f"    Quelle  {scenario.source}")
        print(f"    Config  {overlay}")
        print(f"    Ziel    {outdir}")

        if tracker is not None:
            tracker.reset()          # Track-IDs nicht ins naechste Szenario ziehen
        try:
            summary = run_debug(
                scenario.source, settings, resolve_views(args.views, settings), outdir,
                tracker=tracker, stride=args.stride, max_frames=max_frames,
                fps=args.fps, label=scenario.name,
                auto_crop=not args.no_crop,
                truth=scenario.ground_truth(args.ground_truth), on_progress=print)
        except (ValueError, RuntimeError) as exc:
            # Ein kaputtes Szenario darf den Durchlauf nicht abbrechen -- die
            # uebrigen sind trotzdem auswertbar, der Fehler steht in der Tabelle.
            print(f"    FEHLER: {exc}")
            summary = RunSummary(scenario=scenario.name, error=str(exc))
        summaries.append(summary)
        print("\n" + "\n".join("    " + l for l in summary.as_text().splitlines()[1:]))

    index = Path(args.results) / "index.md"
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text(_index_markdown(summaries, args), encoding="utf-8")

    print("\n" + render_table(summaries))
    print(f"Vergleichstabelle: {index}")
    # Rueckgabewert taugt damit als Testergebnis: 0 nur, wenn nichts
    # gescheitert ist UND jede vorhandene Annotation erfuellt wurde.
    failed = [s for s in summaries if s.error or (s.score and not s.score.perfect)]
    if failed:
        print(f"\n{len(failed)} von {len(summaries)} Szenario(s) nicht in Ordnung: "
              + ", ".join(s.scenario for s in failed))
    else:
        annotated = sum(1 for s in summaries if s.score is not None)
        print(f"\nAlle {len(summaries)} Szenario(s) durchgelaufen, "
              f"{annotated} davon gegen Annotation geprueft.")
    return 1 if failed else 0


def _list(found, scenario_dir: Path) -> int:
    if not found:
        print(f"Keine Szenarien in {scenario_dir}/ gefunden.\n\n"
              "Erwartet wird dort eine Videodatei (.mp4, .mov, .avi, .mkv, .webm)\n"
              "oder ein Ordner mit Bildern. Der Dateiname ist der Szenarioname:\n\n"
              f"    {scenario_dir}/mein_szenario.mp4\n"
              "    config/scenarios/mein_szenario.yaml   (optional: Abweichungen)\n"
              "    results/mein_szenario/                (entsteht beim Lauf)")
        return 1
    print(f"{len(found)} Szenario(s) in {scenario_dir}/:\n")
    for scenario in found:
        print("  " + scenario.describe())
    print("\nAusfuehren:  adascope scenarios            (alle)"
          "\n             adascope scenarios <name>     (eines)")
    return 0


def _index_markdown(summaries: list[RunSummary], args: argparse.Namespace) -> str:
    note = ""
    if args.quick or args.max_frames:
        limit = QUICK_FRAMES if args.quick else args.max_frames
        note = f"\n> Teillauf: nur die ersten {limit} Frames je Szenario.\n"
    return (
        "# Szenario-Ergebnisse\n"
        f"\nErzeugt mit `adascope scenarios`. Ansichten: `{args.views}`."
        f"{note}\n"
        + render_table(summaries)
        + "\n## Spalten\n\n"
        "- **H fresh** — Anteil Frames mit beiden durchgezogenen Randlinien.\n"
        "  Der Rest laeuft auf gehaltener oder gar keiner Homographie und ist die\n"
        "  Obergrenze fuer alles Nachgelagerte.\n"
        "- **Spurliste** — Anteil Frames mit verwertbarer ego-relativer Spurliste.\n"
        "- **Index-Spruenge** — Spruenge des *positionsbasierten* Ego-Index. Die\n"
        "  Groesse, die `lanes.indexing` aufloesen soll; ein hoher Wert heisst,\n"
        "  dass die Rohgeometrie flackert.\n"
        "- **Ego min** — kleinster Anteil des Ego-Footprints in der eigenen Spur.\n"
        "  Unter 1.00 beruehrt oder ueberschreitet das Ego die Spurgrenze.\n"
        "- **Ereignisse** — was die State Machine gemeldet hat.\n"
        "\nDetails je Lauf in `<szenario>/summary.txt`, Rohdaten in\n"
        "`<szenario>/debug_metrics.csv`.\n")

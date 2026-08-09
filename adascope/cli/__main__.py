"""Einheitliches ``adascope``-Kommando -- der Einstiegspunkt des Projekts.

Jedes Subkommando ist ein Modul mit `configure_parser(parser)` und `run(args)`.
Neue Kommandos brauchen nur einen Eintrag in COMMANDS -- die Dispatch-Logik
bleibt unveraendert.
"""

import argparse
from importlib import import_module

EPILOG = """
Schnellstart
  adascope scenarios --list          zeigt, was in scenarien/ liegt
  adascope scenarios                 laesst alle laufen -> results/<name>/
  adascope scenarios <name>          nur eines

Aufruf ueber `adascope <command>` oder `python -m adascope <command>`.
"""

COMMANDS = {
    # Szenarien -- der uebliche Einstieg
    "scenarios": ("scenarios", "Szenarien aus scenarien/ nach results/ ausfuehren"),
    # Datenaufbereitung
    "download": ("download", "Video von YouTube laden"),
    "extract": ("extract", "Frames aus einem Video extrahieren"),
    "crop": ("crop", "Frames zuschneiden"),
    "assemble": ("assemble", "Frames zu einem Video zusammensetzen"),
    # Kalibrierung
    "calibrate": ("calibrate", "Spurgeometrie automatisch aus dem Material bestimmen"),
    "roi-editor": ("roi_editor", "Spurpolygone konfigurieren"),
    "crop-box": ("crop_selector", "Zuschnitt-Rechteck konfigurieren"),
    # YOLOE-Analyse
    "detect": ("detect", "Fahrzeuge detektieren und Spuren auswerten"),
    "pipeline": ("pipeline", "Mehrstufigen Workflow ausfuehren"),
    "probe": ("probe", "YOLOE-Prompts auf einem Einzelbild ausprobieren"),
    # Spur-/BEV-Analyse
    "track": ("track", "Fahrzeugtracking in der Bildebene (Video + CSV)"),
    "lane-debug": ("lane_debug", "Debug-Videos der Spur-Pipeline"),
    "lane-sensitivity": ("sensitivity", "Robustheitsmessungen auf einem Einzelbild"),
    # Werkzeuge
    "tool": ("tool", "PyQt6-Datenwerkzeuge starten (Extra [gui])"),
}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="adascope",
                                     description="adascope -- Analyse von ADAS-HMI-Videos: Spuren, Bodenebene, Fahrzeuge, Cut-In-Ereignisse",
                                     epilog=EPILOG,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    handlers = {}
    for command, (module_name, help_text) in COMMANDS.items():
        module = import_module(f"{__package__}.{module_name}")
        subparser = subparsers.add_parser(command, help=help_text,
                                          description=module.__doc__,
                                          formatter_class=argparse.RawDescriptionHelpFormatter)
        module.configure_parser(subparser)
        handlers[command] = module.run
    args = parser.parse_args(argv)
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())

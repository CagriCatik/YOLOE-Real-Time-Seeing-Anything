"""Startet die PyQt6-Datenwerkzeuge (Dataset Forge).

PyQt6 ist eine optionale Abhaengigkeit: `pip install -e .[gui]`. Der Import
liegt deshalb in `run()` -- ohne ihn wuerde die gesamte CLI eine GUI-Bibliothek
brauchen, nur damit `adascope --help` funktioniert.
"""

from __future__ import annotations

import argparse
import sys


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-config", default=None,
                        help="Pfad zur Dataset-Konfiguration des Werkzeugs")


def run(args: argparse.Namespace) -> int:
    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError as exc:
        raise SystemExit("PyQt6 fehlt -- installieren mit: pip install -e .[gui]") from exc
    from ..tool.main_window import MainWindow

    app = QApplication(sys.argv[:1])
    app.setApplicationName("Dataset Forge")
    window = MainWindow()
    window.show()
    return app.exec()

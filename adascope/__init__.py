"""YOLOE-Lane: Spur-, Fahrbahn- und Cut-In-Analyse fuer HMI-Clustervideos.

Schichten, von innen nach aussen -- jede kennt nur die inneren:

    config      typisierte Kalibrierung, eine YAML je Domaene
    io          Frames, Videos, Tabellen; kein Domaenenwissen
    detection   Modell-Adapter; einziger Ort mit `ultralytics`
    vision      YOLOE-Domaene: ROIs, Carpet, Driving Area, HUD
    lanes       klassische CV-Domaene: Spuren, Bodenebene, Ereignisse
    render      Overlays und Debug-Ansichten
    cli         duenne Adapter, ein Subkommando je Aufgabe
    tool        optionale PyQt6-Datenwerkzeuge

Domaenencode importiert weder `ultralytics` noch `argparse` und schreibt keine
Dateien -- deshalb ist er mit Fakes testbar, ohne ein Modell zu laden.
"""

from .config import Settings
from .lanes import FrameAnalysis, SequencePipeline
from .vision import FrameResult, analyse_frame

__all__ = ["FrameAnalysis", "FrameResult", "SequencePipeline", "Settings", "analyse_frame"]
__version__ = "1.0.0"

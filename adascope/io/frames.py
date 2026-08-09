"""Einziger Besitzer von Frame- und Video-I/O.

Vor der Zusammenfuehrung war dieselbe Logik dreimal implementiert: als
`frames.list_frames`, als `track_video.list_images/video_frames/video_fps` und
noch einmal im Datenwerkzeug. Jede Variante hatte eine andere Menge erlaubter
Endungen und ein anderes Verhalten bei nicht lesbaren Dateien.

Alle Leser liefern `(name, frame)`-Paare, damit Bildordner und Videodatei
gegeneinander austauschbar sind -- `iter_source()` entscheidet anhand des Pfads,
welcher Leser zustaendig ist.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

import cv2
import numpy as np

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm"})
DEFAULT_FPS = 25.0

Frame = tuple[str, np.ndarray]


def wanted_exts(ext: str | None = None) -> frozenset[str]:
    """Erlaubte Bildendungen; `ext` schraenkt auf genau eine ein."""
    if ext is None:
        return IMAGE_EXTENSIONS
    return frozenset({"." + str(ext).lower().lstrip(".")})


def list_frames(folder: str | Path, every: int = 1, ext: str | None = None) -> list[Path]:
    """Sortierte Bildpfade eines Ordners, jeder `every`-te."""
    if every < 1:
        raise ValueError("every muss mindestens 1 sein")
    allowed = wanted_exts(ext)
    paths = sorted(p for p in Path(folder).iterdir()
                   if p.is_file() and p.suffix.lower() in allowed)
    return paths[::every]


def read_image(path: str | Path) -> np.ndarray | None:
    """Liest ein Bild; None, wenn OpenCV es nicht dekodieren kann."""
    return cv2.imread(str(path))


def image_frames(paths: Iterable[str | Path]) -> Iterator[Frame]:
    """Nicht lesbare Dateien werden uebersprungen, nicht als None geliefert."""
    for path in paths:
        frame = read_image(path)
        if frame is not None:
            yield Path(path).name, frame


def video_frames(path: str | Path) -> Iterator[Frame]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Video kann nicht geoeffnet werden: {path}")
    try:
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                return
            yield f"frame_{index:06d}", frame
            index += 1
    finally:
        capture.release()


def video_fps(path: str | Path, default: float = DEFAULT_FPS) -> float:
    capture = cv2.VideoCapture(str(path))
    try:
        return float(capture.get(cv2.CAP_PROP_FPS) or default)
    finally:
        capture.release()


def is_video(path: str | Path) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def iter_source(path: str | Path, every: int = 1,
                fps: float = DEFAULT_FPS) -> tuple[Iterator[Frame], float]:
    """Video *oder* Bildordner -> (Frame-Iterator, fps).

    Damit nehmen alle Kommandos beide Quellen entgegen, ohne den Unterschied
    selbst zu kennen. `fps` gilt nur fuer Bildordner; ein Video bringt seine
    eigene Bildrate mit.
    """
    source = Path(path)
    if is_video(source):
        return video_frames(source), video_fps(source, fps)
    if not source.is_dir():
        raise ValueError(f"Weder Video noch Ordner: {source}")
    paths = list_frames(source, every)
    if not paths:
        raise ValueError(f"Keine Bilder gefunden in: {source}")
    return image_frames(paths), fps


class VideoWriter:
    """VideoWriter, der seine Groesse aus dem ersten Frame nimmt.

    mp4v verlangt gerade Kantenlaengen. Bei ungerader Groesse schneidet der
    Encoder sonst stumm eine Zeile oder Spalte ab -- die Ausgabe passt danach
    nicht mehr zu den Bildkoordinaten, ohne dass ein Fehler auftritt. Deshalb
    wird hier explizit auf gerade zugeschnitten.
    """

    def __init__(self, path: str | Path, fps: float, fourcc: str = "mp4v"):
        self.path = Path(path)
        self.fps, self.fourcc = fps, fourcc
        self._writer: cv2.VideoWriter | None = None
        self.size: tuple[int, int] | None = None

    def write(self, frame: np.ndarray) -> None:
        if self._writer is None:
            height, width = frame.shape[:2]
            self.size = (width - width % 2, height - height % 2)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = cv2.VideoWriter(
                str(self.path), cv2.VideoWriter_fourcc(*self.fourcc), self.fps, self.size)
            if not self._writer.isOpened():
                raise RuntimeError(f"Ausgabevideo kann nicht erstellt werden: {self.path}")
        width, height = self.size
        if frame.shape[:2] != (height, width):
            frame = (frame[:height, :width]
                     if frame.shape[0] >= height and frame.shape[1] >= width
                     else cv2.resize(frame, self.size))
        self._writer.write(frame)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

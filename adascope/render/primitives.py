"""Zeichen-Grundbausteine der Debug-Ansichten. Kein Domaenenwissen."""

from __future__ import annotations

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
Color = tuple[int, int, int]


def hud(canvas: np.ndarray, lines: list[tuple[str, Color]],
        x: int = 10, y: int = 20, scale: float = 0.48) -> None:
    """Textzeilen mit abgedunkeltem Kasten dahinter -- lesbar auf jedem Grund."""
    if not lines:
        return
    step = max(int(20 * scale / 0.48), 10)
    width = int(max(len(text) for text, _ in lines) * 8.2 * scale / 0.48)
    box = canvas[max(y - step, 0):y + step * len(lines), max(x - 5, 0):x + width + 8]
    if box.size:
        box[:] = (box * 0.35).astype(np.uint8)
    for i, (text, color) in enumerate(lines):
        cv2.putText(canvas, text, (x, y + i * step), FONT, scale, color, 1, cv2.LINE_AA)


def dashed_line(canvas: np.ndarray, p0, p1, color: Color, thickness: int = 1,
                dash: int = 9) -> None:
    p0, p1 = np.array(p0, float), np.array(p1, float)
    length = float(np.hypot(*(p1 - p0)))
    if length < 1:
        return
    for start in range(0, int(length), dash * 2):
        a = p0 + (p1 - p0) * (start / length)
        b = p0 + (p1 - p0) * (min(start + dash, length) / length)
        cv2.line(canvas, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                 color, thickness, cv2.LINE_AA)


def dashed_polyline(canvas: np.ndarray, points: np.ndarray, color: Color,
                    thickness: int = 1) -> None:
    """Jedes zweite Segment zeichnen -- fuer virtuelle Grenzen in Schraegsicht."""
    for a, b in zip(points[::2], points[1::2]):
        cv2.line(canvas, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                 color, thickness, cv2.LINE_AA)


def fit(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """Skaliert seitenverhaeltnistreu und legt das Ergebnis auf schwarzen Grund."""
    scale = min(width / img.shape[1], height / img.shape[0])
    resized = cv2.resize(img, (max(int(img.shape[1] * scale), 1),
                               max(int(img.shape[0] * scale), 1)))
    canvas = np.zeros((height, width, 3), np.uint8)
    y0, x0 = (height - resized.shape[0]) // 2, (width - resized.shape[1]) // 2
    canvas[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
    return canvas


def placeholder(width: int, height: int, text: str, color: Color) -> np.ndarray:
    """Ersatzbild fuer eine Stufe, die in diesem Frame nichts geliefert hat.

    Bewusst ein Bild und kein uebersprungener Frame: im Video muss sichtbar
    bleiben, dass die Stufe ausgefallen ist, sonst laeuft die Zeitachse falsch.
    """
    canvas = np.zeros((height, width, 3), np.uint8)
    size = cv2.getTextSize(text, FONT, 0.6, 1)[0]
    cv2.putText(canvas, text, (max((width - size[0]) // 2, 4), height // 2), FONT,
                0.6, color, 1, cv2.LINE_AA)
    return canvas

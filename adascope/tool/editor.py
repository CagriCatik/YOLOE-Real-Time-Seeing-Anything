"""Interaktiver Editor fuer Zuschnitt, Polygone und Boxen.

Drei Modi teilen sich dieselbe Zeichenflaeche:

    crop     ein Rechteck aufziehen (Zuschnitt)
    roi      Polygon Punkt fuer Punkt setzen
    box      Rechteck aufziehen -- gespeichert als Vierpunkt-Polygon

Eine Box ist bewusst kein eigener Datentyp. Sie ist ein Polygon mit vier
Ecken, damit alles Nachgelagerte unveraendert bleibt. Gemerkt wird nur die
HERKUNFT (`shapes`), und zwar aus einem Grund: zieht man an einer Ecke,
sollen die Nachbarecken mitgehen, damit die Box rechteckig bleibt. Ohne
diese Notiz waere sie nach dem ersten Ziehen ein beliebiges Viereck.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import QWidget

from .core import box_polygon

# Vorschlaege fuer neue Masken -- untereinander und auf Strassenaufnahmen
# gut unterscheidbar. Wer eine andere Farbe will, waehlt sie im Werkzeug.
PALETTE = ["#ff414d", "#48a7ff", "#5ddc7a", "#ffd166", "#c77dff",
           "#ff8fab", "#22d3ee", "#f97316"]


class RegionEditor(QWidget):
    changed = pyqtSignal()

    COLORS = {
        "left": QColor("#48a7ff"),
        "ego": QColor("#f4f7fb"),
        "right": QColor("#ff9f43"),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 390)
        self.setMouseTracking(True)
        self.image = QImage()
        self.image_path: Path | None = None
        self.mode = "crop"
        self.active_roi = "left"
        self.crop_box = [0.1, 0.1, 0.9, 0.9]
        self.rois = {"left": [], "ego": [], "right": []}
        self.show_crop = True
        self.default_polygon_color = QColor("#ffffff")
        self.polygon_fill_alpha = 35
        # AUSDRUECKLICH gewaehlte Fuellfarbe je Maske. Was hier fehlt, wird
        # beim Speichern zur globalen Fuellfarbe (schwarz) -- maskieren
        # heisst schwaerzen. Zum ANZEIGEN dient `display_color()`, damit
        # auch eine schwarze Maske sichtbar bleibt.
        self.colors: dict[str, QColor] = {}
        # "box" oder "polygon" je Maske -- siehe Modulkopf.
        self.shapes: dict[str, str] = {}
        self._original = None
        self._drag_start: QPointF | None = None
        self._drag_vertex: int | None = None

    def load_image(self, path: str | Path) -> None:
        image = QImage(str(path))
        if image.isNull():
            raise ValueError(f"Bild kann nicht geladen werden: {path}")
        self.image = image
        self.image_path = Path(path)
        self.update()

    def set_regions(self, crop_box, rois) -> None:
        self.crop_box = list(crop_box)
        merged = {"left": [], "ego": [], "right": []}
        merged.update({name: [list(point) for point in points] for name, points in rois.items()})
        self.rois = merged
        self._original = (list(self.crop_box), deepcopy(self.rois))
        self.update()

    def set_polygons(self, polygons, colors=None, shapes=None) -> None:
        """Beliebige benannte Polygonsammlung laden (Maskeneditor)."""
        self.rois = {name: [list(point) for point in points] for name, points in polygons.items()}
        if colors is not None:
            self.colors = {name: QColor(value) for name, value in colors.items()}
        if shapes is not None:
            self.shapes = dict(shapes)
        self._original = (list(self.crop_box), deepcopy(self.rois))
        self.update()

    def display_color(self, name: str) -> QColor:
        """Farbe zum Zeichnen -- die gewaehlte, sonst eine aus der Palette.

        Eine schwarze Fuellfarbe waere im Editor und im Debugbild unsichtbar.
        Deshalb wird sie hier aufgehellt statt uebernommen.
        """
        chosen = self.colors.get(name)
        # Zu dunkel zum Erkennen -> Palettenfarbe. Aufhellen hilft hier nicht:
        # aus #080808 wird auch verdreifacht nur #151515. Dieselbe Schwelle
        # benutzt `core.mask_display_colors` fuer das Debugbild.
        if chosen is not None and max(chosen.red(), chosen.green(), chosen.blue()) >= 60:
            return chosen
        fallback = self.COLORS.get(name)
        if fallback is not None:
            return fallback
        index = list(self.rois).index(name) if name in self.rois else 0
        return self.suggest_color(index)

    def fill_color(self, name: str) -> QColor | None:
        """Die gewaehlte Fuellfarbe, oder None fuer den Standard (schwarz)."""
        return self.colors.get(name)

    def color_of(self, name: str) -> QColor:
        """Alias fuer die Anzeigefarbe -- was man sieht, ist die Anzeige."""
        return self.display_color(name)

    def set_color(self, name: str, color: QColor) -> None:
        self.colors[name] = QColor(color)
        self.changed.emit()
        self.update()

    def suggest_color(self, index: int) -> QColor:
        """Naechste Palettenfarbe -- damit zwei neue Masken nie gleich aussehen."""
        return QColor(PALETTE[index % len(PALETTE)])

    def is_box(self, name: str) -> bool:
        return self.shapes.get(name) == "box"

    def reset_regions(self) -> None:
        if self._original:
            self.crop_box, self.rois = list(self._original[0]), deepcopy(self._original[1])
        else:
            self.crop_box = [0.1, 0.1, 0.9, 0.9]
            self.rois = {"left": [], "ego": [], "right": []}
        self.changed.emit(); self.update()

    def clear_active_roi(self) -> None:
        self.rois.setdefault(self.active_roi, []).clear()
        self.shapes.pop(self.active_roi, None)
        self.changed.emit(); self.update()

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.update()

    def set_active_roi(self, name: str) -> None:
        self.active_roi = name
        self.update()

    def image_size(self) -> tuple[int, int]:
        return self.image.width(), self.image.height()

    def _image_rect(self) -> QRectF:
        if self.image.isNull():
            return QRectF()
        scale = min(self.width() / self.image.width(), self.height() / self.image.height())
        width, height = self.image.width() * scale, self.image.height() * scale
        return QRectF((self.width() - width) / 2, (self.height() - height) / 2, width, height)

    def _to_normalized(self, point: QPointF) -> QPointF | None:
        rect = self._image_rect()
        if not rect.contains(point):
            return None
        return QPointF((point.x() - rect.left()) / rect.width(), (point.y() - rect.top()) / rect.height())

    def _to_widget(self, point) -> QPointF:
        rect = self._image_rect()
        return QPointF(rect.left() + point[0] * rect.width(), rect.top() + point[1] * rect.height())

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#12171f"))
        if self.image.isNull():
            painter.setPen(QColor("#8290a3"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Ein Bild auswählen")
            return
        rect = self._image_rect()
        painter.drawImage(rect, self.image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self.show_crop:
            crop = QRectF(self._to_widget(self.crop_box[:2]), self._to_widget(self.crop_box[2:])).normalized()
            painter.setPen(QPen(QColor("#ffd166"), 3 if self.mode == "crop" else 1.5))
            painter.setBrush(QColor(255, 209, 102, 24))
            painter.drawRect(crop)

        for name, points in self.rois.items():
            if not points:
                continue
            color = self.color_of(name)
            active = name == self.active_roi and self.mode in ("roi", "box")
            pen = QPen(color, 3 if active else 1.5)
            if self.is_box(name):
                # Boxen gestrichelt, solange sie nicht aktiv sind -- so ist
                # ohne Beschriftung erkennbar, was Box und was Polygon ist.
                pen.setStyle(Qt.PenStyle.SolidLine if active else Qt.PenStyle.DashLine)
            painter.setPen(pen)
            polygon = QPolygonF([self._to_widget(point) for point in points])
            painter.setBrush(QColor(color.red(), color.green(), color.blue(), self.polygon_fill_alpha))
            if len(points) >= 3:
                painter.drawPolygon(polygon)
            else:
                painter.drawPolyline(polygon)
            painter.setBrush(color)
            for point in polygon:
                painter.drawEllipse(point, 5, 5)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        normalized = self._to_normalized(event.position())
        if normalized is None:
            return
        if self.mode == "crop" and event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = normalized
            self.crop_box = [normalized.x(), normalized.y(), normalized.x(), normalized.y()]
        elif self.mode == "box" and event.button() == Qt.MouseButton.LeftButton:
            points = self.rois.setdefault(self.active_roi, [])
            nearest = self._nearest_vertex(event.position(), points)
            if nearest is not None:
                self._drag_vertex = nearest          # bestehende Ecke ziehen
            else:
                # Neue Box aufziehen. Sie ERSETZT die bisherige: eine Maske
                # traegt genau eine Box, sonst waere die Ecklogik mehrdeutig.
                self._drag_start = normalized
                self.rois[self.active_roi] = box_polygon(
                    normalized.x(), normalized.y(), normalized.x(), normalized.y())
                self.shapes[self.active_roi] = "box"
                self.changed.emit()
                self.update()
        elif self.mode == "roi":
            points = self.rois.setdefault(self.active_roi, [])
            nearest = self._nearest_vertex(event.position(), points)
            if event.button() == Qt.MouseButton.RightButton and nearest is not None:
                points.pop(nearest); self.changed.emit(); self.update()
            elif event.button() == Qt.MouseButton.LeftButton:
                if nearest is None:
                    points.append([normalized.x(), normalized.y()]); self._drag_vertex = len(points) - 1
                else:
                    self._drag_vertex = nearest
                self.changed.emit(); self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        normalized = self._to_normalized(event.position())
        if normalized is None:
            return
        if self.mode == "crop" and self._drag_start is not None:
            self.crop_box = [min(self._drag_start.x(), normalized.x()), min(self._drag_start.y(), normalized.y()),
                             max(self._drag_start.x(), normalized.x()), max(self._drag_start.y(), normalized.y())]
            self.update()
        elif self.mode == "box" and self._drag_start is not None:
            self.rois[self.active_roi] = box_polygon(
                self._drag_start.x(), self._drag_start.y(), normalized.x(), normalized.y())
            self.update()
        elif self._drag_vertex is not None:
            points = self.rois[self.active_roi]
            if self.is_box(self.active_roi) and len(points) == 4:
                # Gegenecke festhalten und die Box neu aufspannen -- so bleibt
                # sie rechteckig, statt zum beliebigen Viereck zu werden.
                opposite = points[(self._drag_vertex + 2) % 4]
                self.rois[self.active_roi] = box_polygon(
                    opposite[0], opposite[1], normalized.x(), normalized.y())
            else:
                points[self._drag_vertex] = [normalized.x(), normalized.y()]
            self.update()

    def mouseReleaseEvent(self, _event: QMouseEvent) -> None:
        if self.mode == "box" and self._drag_start is not None:
            # Ein versehentlicher Klick ohne Ziehen soll keine unsichtbare
            # Null-Box hinterlassen.
            points = self.rois.get(self.active_roi) or []
            if len(points) == 4 and (points[1][0] - points[0][0] < .005
                                     or points[2][1] - points[1][1] < .005):
                self.rois[self.active_roi] = []
                self.shapes.pop(self.active_roi, None)
            self.changed.emit()
        elif self.mode == "crop" and self._drag_start is not None:
            if self.crop_box[2] - self.crop_box[0] < .005 or self.crop_box[3] - self.crop_box[1] < .005:
                self.crop_box = [0.1, 0.1, 0.9, 0.9]
            self.changed.emit()
        self._drag_start = None
        self._drag_vertex = None
        self.update()

    def _nearest_vertex(self, position: QPointF, points) -> int | None:
        distances = [((self._to_widget(point).x() - position.x()) ** 2 +
                      (self._to_widget(point).y() - position.y()) ** 2, index)
                     for index, point in enumerate(points)]
        if not distances:
            return None
        distance, index = min(distances)
        return index if distance <= 12 ** 2 else None

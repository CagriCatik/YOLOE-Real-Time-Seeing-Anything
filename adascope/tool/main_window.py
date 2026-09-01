"""PyQt6 user interface for the independent dataset tool."""

from __future__ import annotations

from pathlib import Path
from threading import Event
import traceback

from PyQt6.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QDesktopServices
from PyQt6.QtCore import QUrl
from PyQt6.QtWidgets import (
    QButtonGroup, QComboBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel,
    QInputDialog, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar,
    QPushButton, QSpinBox, QDoubleSpinBox, QTabWidget, QVBoxLayout, QWidget,
)

from .core import (DEFAULT_FILL_BGR, create_mask_outputs, crop_frames, download_video,
                   extract_frames,
                   find_ffmpeg, frames_to_video, load_mask_config, load_regions,
                   probe_video, save_mask_config, save_regions)
from PyQt6.QtWidgets import QColorDialog
from .editor import RegionEditor

WORKSPACE = Path(__file__).resolve().parents[2]

BOX_HINT = ("Ziehen legt eine Box an und ersetzt die bisherige dieser Maske. Ecke ziehen passt sie an -- sie bleibt rechteckig.")
POLYGON_HINT = ("Linksklick setzt einen Punkt, Ziehen verschiebt ihn, Rechtsklick loescht ihn. Ab drei Punkten wird gefuellt.")


class WorkerSignals(QObject):
    progress = pyqtSignal(int, str)
    result = pyqtSignal(object)
    error = pyqtSignal(str)
    finished = pyqtSignal()


class Worker(QRunnable):
    def __init__(self, operation, *args):
        super().__init__()
        self.operation = operation
        self.args = args
        self.cancel = Event()
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        try:
            result = self.operation(*self.args, self.signals.progress.emit, self.cancel)
            self.signals.result.emit(result)
        except InterruptedError as exc:
            self.signals.error.emit(str(exc))
        except Exception:
            self.signals.error.emit(traceback.format_exc())
        finally:
            self.signals.finished.emit()


class PathField(QWidget):
    def __init__(self, mode="file", file_filter="Alle Dateien (*)", parent=None):
        super().__init__(parent)
        self.mode, self.file_filter = mode, file_filter
        self.edit = QLineEdit()
        button = QPushButton("…")
        button.setFixedWidth(38)
        button.clicked.connect(self.browse)
        layout = QHBoxLayout(self); layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.edit); layout.addWidget(button)

    def text(self): return self.edit.text().strip()
    def setText(self, value): self.edit.setText(str(value))

    def browse(self):
        current = self.text() or str(Path.cwd())
        if self.mode == "directory":
            value = QFileDialog.getExistingDirectory(self, "Ordner auswählen", current)
        elif self.mode == "save":
            value, _ = QFileDialog.getSaveFileName(self, "Datei speichern", current, self.file_filter)
        else:
            value, _ = QFileDialog.getOpenFileName(self, "Datei auswählen", current, self.file_filter)
        if value: self.setText(value)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dataset Forge — Video Data Preparation")
        self.resize(1180, 820)
        self.thread_pool = QThreadPool.globalInstance()
        self.worker: Worker | None = None
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.tabs.addTab(self._download_page(), "1  Download")
        self.tabs.addTab(self._extract_page(), "2  Frames")
        self.tabs.addTab(self._configure_page(), "3  Regionen")
        self.tabs.addTab(self._crop_page(), "4  Batch Crop")
        self.tabs.addTab(self._video_page(), "5  Video")
        self.tabs.addTab(self._mask_page(), "6  Maskierung")
        self._build_status()
        self._apply_style()

    def _page(self, title, subtitle):
        page = QWidget(); layout = QVBoxLayout(page); layout.setContentsMargins(28, 24, 28, 24); layout.setSpacing(14)
        heading = QLabel(title); heading.setObjectName("heading")
        description = QLabel(subtitle); description.setObjectName("muted"); description.setWordWrap(True)
        layout.addWidget(heading); layout.addWidget(description)
        return page, layout

    def _download_page(self):
        page, layout = self._page("Video herunterladen", "Ein einzelnes Online-Video lokal für die Datenerzeugung speichern.")
        form = QFormLayout(); form.setSpacing(12)
        self.download_url = QLineEdit(); self.download_url.setPlaceholderText("https://www.youtube.com/watch?v=…")
        self.download_dir = PathField("directory"); self.download_dir.setText(WORKSPACE / "data" / "raw")
        self.download_name = QLineEdit("video.mp4")
        self.download_height = QComboBox(); self.download_height.addItems(["2160", "1440", "1080", "720", "480"]); self.download_height.setCurrentText("1080")
        self.download_ffmpeg = PathField("file", "FFmpeg (ffmpeg.exe)")
        detected_ffmpeg = find_ffmpeg()
        if detected_ffmpeg: self.download_ffmpeg.setText(detected_ffmpeg)
        form.addRow("Video-URL", self.download_url); form.addRow("Zielordner", self.download_dir)
        form.addRow("Dateiname", self.download_name); form.addRow("Maximale Höhe", self.download_height)
        form.addRow("FFmpeg", self.download_ffmpeg)
        layout.addLayout(form)
        self.download_button = QPushButton("Download starten"); self.download_button.setObjectName("primary")
        self.download_button.clicked.connect(self._start_download); layout.addWidget(self.download_button)
        self.download_log = QPlainTextEdit(); self.download_log.setReadOnly(True); self.download_log.setPlaceholderText("Download-Status erscheint hier …")
        layout.addWidget(self.download_log, 1)
        return page

    def _extract_page(self):
        page, layout = self._page("Frames extrahieren", "Frames aus einem lokalen Video gewinnen. Wahlweise jeden n-ten Frame oder eine Ziel-FPS verwenden.")
        form = QFormLayout(); form.setSpacing(12)
        self.extract_video = PathField("file", "Videos (*.mp4 *.mov *.avi *.mkv *.webm)")
        self.extract_dir = PathField("directory"); self.extract_dir.setText(WORKSPACE / "data" / "frames" / "raw")
        self.extract_every = QSpinBox(); self.extract_every.setRange(1, 10000); self.extract_every.setValue(1)
        self.extract_fps = QDoubleSpinBox(); self.extract_fps.setRange(0, 240); self.extract_fps.setDecimals(2); self.extract_fps.setSpecialValueText("Aus")
        self.extract_format = QComboBox(); self.extract_format.addItems(["jpg", "png"])
        self.extract_quality = QSpinBox(); self.extract_quality.setRange(1, 100); self.extract_quality.setValue(95)
        form.addRow("Eingabevideo", self.extract_video); form.addRow("Ausgabeordner", self.extract_dir)
        form.addRow("Jeden n-ten Frame", self.extract_every); form.addRow("Ziel-FPS (optional)", self.extract_fps)
        form.addRow("Bildformat", self.extract_format); form.addRow("JPEG-Qualität", self.extract_quality)
        layout.addLayout(form)
        buttons = QHBoxLayout()
        inspect = QPushButton("Video prüfen"); inspect.clicked.connect(self._inspect_video)
        self.extract_button = QPushButton("Extraktion starten"); self.extract_button.setObjectName("primary"); self.extract_button.clicked.connect(self._start_extract)
        buttons.addWidget(inspect); buttons.addStretch(); buttons.addWidget(self.extract_button); layout.addLayout(buttons); layout.addStretch()
        return page

    def _configure_page(self):
        page, layout = self._page("ROI und Crop konfigurieren", "Crop-Rechteck aufziehen. Im ROI-Modus Punkte setzen oder ziehen; Rechtsklick entfernt einen Punkt. Alle Koordinaten werden normalisiert gespeichert.")
        top = QHBoxLayout()
        self.editor_image = PathField("file", "Bilder (*.jpg *.jpeg *.png *.bmp *.webp)")
        load_image = QPushButton("Bild laden"); load_image.clicked.connect(self._load_editor_image)
        top.addWidget(QLabel("Referenzbild")); top.addWidget(self.editor_image, 1); top.addWidget(load_image); layout.addLayout(top)
        controls = QHBoxLayout()
        crop_mode = QPushButton("Crop-Rechteck"); crop_mode.setCheckable(True); crop_mode.setChecked(True)
        roi_mode = QPushButton("ROI-Polygon"); roi_mode.setCheckable(True)
        modes = QButtonGroup(self); modes.setExclusive(True); modes.addButton(crop_mode); modes.addButton(roi_mode)
        crop_mode.clicked.connect(lambda: self.editor.set_mode("crop")); roi_mode.clicked.connect(lambda: self.editor.set_mode("roi"))
        self.roi_name = QComboBox(); self.roi_name.addItems(["left", "ego", "right"]); self.roi_name.currentTextChanged.connect(self.editor_set_roi)
        clear = QPushButton("Aktive ROI leeren"); clear.clicked.connect(lambda: self.editor.clear_active_roi())
        reset = QPushButton("Zurücksetzen"); reset.clicked.connect(lambda: self.editor.reset_regions())
        controls.addWidget(crop_mode); controls.addWidget(roi_mode); controls.addWidget(QLabel("ROI:")); controls.addWidget(self.roi_name); controls.addWidget(clear); controls.addStretch(); controls.addWidget(reset)
        layout.addLayout(controls)
        self.editor = RegionEditor(); layout.addWidget(self.editor, 1)
        bottom = QHBoxLayout()
        self.region_config = PathField("save", "JSON (*.json)"); self.region_config.setText(WORKSPACE / "configs" / "dataset_config.json")
        load_config = QPushButton("Konfiguration laden"); load_config.clicked.connect(self._load_regions)
        save_config = QPushButton("Konfiguration speichern"); save_config.setObjectName("primary"); save_config.clicked.connect(self._save_regions)
        bottom.addWidget(QLabel("Konfiguration")); bottom.addWidget(self.region_config, 1); bottom.addWidget(load_config); bottom.addWidget(save_config); layout.addLayout(bottom)
        return page

    def editor_set_roi(self, name):
        if hasattr(self, "editor"): self.editor.set_active_roi(name)

    def _crop_page(self):
        page, layout = self._page("Frames zuschneiden", "Das gespeicherte Crop-Rechteck auf einen kompletten Bilderordner anwenden. ROI-Polygone verändern die Bilder nicht.")
        form = QFormLayout(); form.setSpacing(12)
        self.crop_input = PathField("directory"); self.crop_input.setText(WORKSPACE / "data" / "frames" / "raw")
        self.crop_output = PathField("directory"); self.crop_output.setText(WORKSPACE / "data" / "frames" / "cropped")
        self.crop_config = PathField("file", "JSON (*.json)"); self.crop_config.setText(WORKSPACE / "configs" / "dataset_config.json")
        self.crop_format = QComboBox(); self.crop_format.addItem("Originalformat", "same"); self.crop_format.addItem("JPEG", "jpg"); self.crop_format.addItem("PNG", "png")
        self.crop_quality = QSpinBox(); self.crop_quality.setRange(1, 100); self.crop_quality.setValue(95)
        form.addRow("Frame-Ordner", self.crop_input); form.addRow("Ausgabeordner", self.crop_output)
        form.addRow("Konfiguration", self.crop_config); form.addRow("Ausgabeformat", self.crop_format); form.addRow("JPEG-Qualität", self.crop_quality)
        layout.addLayout(form)
        self.crop_button = QPushButton("Batch-Cropping starten"); self.crop_button.setObjectName("primary"); self.crop_button.clicked.connect(self._start_crop)
        layout.addWidget(self.crop_button); layout.addStretch()
        return page

    def _video_page(self):
        page, layout = self._page(
            "Video aus Frames erstellen",
            "Die zugeschnittenen Bilder nach Dateinamen sortieren und als stummes MP4-Video zusammensetzen.",
        )
        form = QFormLayout(); form.setSpacing(12)
        self.video_input = PathField("directory"); self.video_input.setText(WORKSPACE / "data" / "frames" / "cropped")
        self.video_output = PathField("save", "MP4-Video (*.mp4)"); self.video_output.setText(WORKSPACE / "outputs" / "cropped_video.mp4")
        self.video_fps = QDoubleSpinBox(); self.video_fps.setRange(0.1, 240); self.video_fps.setDecimals(3); self.video_fps.setValue(25)
        self.video_every = QSpinBox(); self.video_every.setRange(1, 10000); self.video_every.setValue(1)
        self.video_codec = QComboBox()
        self.video_codec.addItem("MP4V — hohe Kompatibilität", "mp4v")
        self.video_codec.addItem("AVC1 / H.264 — falls OpenCV unterstützt", "avc1")
        form.addRow("Frame-Ordner", self.video_input)
        form.addRow("Ausgabevideo", self.video_output)
        form.addRow("Frames pro Sekunde", self.video_fps)
        form.addRow("Jeden n-ten Frame", self.video_every)
        form.addRow("Codec", self.video_codec)
        layout.addLayout(form)
        note = QLabel("Hinweis: Der Ausgang enthält keine Audiospur. Bilder mit abweichender Größe werden an den ersten Frame angepasst.")
        note.setObjectName("muted"); note.setWordWrap(True); layout.addWidget(note)
        self.video_button = QPushButton("Video erstellen"); self.video_button.setObjectName("primary")
        self.video_button.clicked.connect(self._start_video); layout.addWidget(self.video_button); layout.addStretch()
        return page

    def _mask_page(self):
        page, layout = self._page(
            "Bildbereiche maskieren",
            "Bereiche als Box aufziehen oder als Polygon Punkt fuer Punkt setzen und als "
            "normalisierte JSON-Konfiguration speichern. Jede Maske hat ihre eigene Farbe; "
            "das Ergebnisbild wird damit gefuellt, das Debug-Bild zeigt sie halbtransparent.",
        )
        image_row = QHBoxLayout()
        self.mask_image = PathField("file", "Bilder (*.jpg *.jpeg *.png *.bmp *.webp)")
        load_image = QPushButton("Bild laden"); load_image.clicked.connect(self._load_mask_image)
        image_row.addWidget(QLabel("Referenzbild")); image_row.addWidget(self.mask_image, 1); image_row.addWidget(load_image)
        layout.addLayout(image_row)

        mode_row = QHBoxLayout()
        self.mask_mode = QComboBox()
        self.mask_mode.addItem("Box aufziehen", "box")
        self.mask_mode.addItem("Polygon punktweise", "roi")
        self.mask_mode.currentIndexChanged.connect(self._select_mask_mode)
        self.mask_hint = QLabel(BOX_HINT); self.mask_hint.setWordWrap(True)
        self.mask_hint.setStyleSheet("color:#8290a3")
        mode_row.addWidget(QLabel("Werkzeug:")); mode_row.addWidget(self.mask_mode)
        mode_row.addWidget(self.mask_hint, 1); layout.addLayout(mode_row)

        controls = QHBoxLayout()
        self.mask_name = QComboBox(); self.mask_name.addItem("mask_1")
        self.mask_name.currentTextChanged.connect(self._select_mask)
        self.mask_color_button = QPushButton("Füllfarbe wählen")
        self.mask_color_button.clicked.connect(self._choose_mask_color)
        add_mask = QPushButton("Neue Maske"); add_mask.clicked.connect(self._add_mask)
        delete_mask = QPushButton("Maske löschen"); delete_mask.clicked.connect(self._delete_mask)
        clear_mask = QPushButton("Leeren"); clear_mask.clicked.connect(lambda: self.mask_editor.clear_active_roi())
        reset = QPushButton("Zurücksetzen"); reset.clicked.connect(self._reset_masks)
        controls.addWidget(QLabel("Aktive Maske:")); controls.addWidget(self.mask_name)
        controls.addWidget(self.mask_color_button)
        controls.addWidget(add_mask); controls.addWidget(delete_mask); controls.addWidget(clear_mask)
        controls.addStretch(); controls.addWidget(reset); layout.addLayout(controls)

        self.mask_editor = RegionEditor()
        self.mask_editor.mode = "box"          # Box ist der haeufigere Fall
        self.mask_editor.show_crop = False
        self.mask_editor.polygon_fill_alpha = 75
        self.mask_editor.set_polygons({"mask_1": []})
        self.mask_editor.set_active_roi("mask_1")
        layout.addWidget(self.mask_editor, 1)

        config_row = QHBoxLayout()
        self.mask_config = PathField("save", "JSON (*.json)"); self.mask_config.setText(WORKSPACE / "configs" / "mask_config.json")
        load_config_button = QPushButton("Konfiguration laden"); load_config_button.clicked.connect(self._load_mask_config)
        config_row.addWidget(QLabel("Masken-Config")); config_row.addWidget(self.mask_config, 1); config_row.addWidget(load_config_button)
        layout.addLayout(config_row)

        output_form = QFormLayout()
        self.masked_output = PathField("save", "PNG (*.png)"); self.masked_output.setText(WORKSPACE / "outputs" / "masking" / "masked.png")
        self.mask_debug_output = PathField("save", "PNG (*.png)"); self.mask_debug_output.setText(WORKSPACE / "outputs" / "masking" / "mask_debug.png")
        output_form.addRow("Maskiertes Bild", self.masked_output); output_form.addRow("Debug-Bild", self.mask_debug_output)
        layout.addLayout(output_form)
        save_button = QPushButton("Konfiguration und Debug-Bilder erzeugen"); save_button.setObjectName("primary")
        save_button.clicked.connect(self._save_masks_and_debug); layout.addWidget(save_button)
        return page

    def _build_status(self):
        self.progress = QProgressBar(); self.progress.setFixedWidth(260); self.progress.setVisible(False)
        self.cancel_button = QPushButton("Abbrechen"); self.cancel_button.setVisible(False); self.cancel_button.clicked.connect(self._cancel)
        self.statusBar().addPermanentWidget(self.progress); self.statusBar().addPermanentWidget(self.cancel_button)
        self.statusBar().showMessage("Bereit")

    def _start_download(self):
        if not self.download_url.text().strip() or not self.download_dir.text():
            return self._warning("Bitte URL und Zielordner angeben.")
        self._run(download_video, self.download_url.text().strip(), self.download_dir.text(),
                  self.download_name.text(), int(self.download_height.currentText()), self.download_ffmpeg.text(),
                  on_result=self._download_done)

    def _download_done(self, path):
        self.download_log.appendPlainText(f"Fertig: {path}")
        self.extract_video.setText(path)

    def _inspect_video(self):
        try:
            info = probe_video(self.extract_video.text())
            QMessageBox.information(self, "Video-Information",
                f"Auflösung: {info.width} × {info.height}\nFPS: {info.fps:.3f}\nFrames: {info.frames}\nDauer: {info.duration:.1f} s")
        except Exception as exc: self._warning(str(exc))

    def _start_extract(self):
        if not self.extract_video.text() or not self.extract_dir.text(): return self._warning("Bitte Video und Ausgabeordner angeben.")
        self._run(extract_frames, self.extract_video.text(), self.extract_dir.text(), self.extract_every.value(),
                  self.extract_fps.value(), self.extract_format.currentText(), self.extract_quality.value(),
                  on_result=lambda count: self._message(f"{count} Frames wurden extrahiert."))

    def _load_editor_image(self):
        try: self.editor.load_image(self.editor_image.text())
        except Exception as exc: self._warning(str(exc))

    def _save_regions(self):
        if self.editor.image.isNull(): return self._warning("Bitte zuerst ein Referenzbild laden.")
        try:
            save_regions(self.region_config.text(), self.editor.image_size(), self.editor.crop_box, self.editor.rois)
            self.crop_config.setText(self.region_config.text()); self._message("Konfiguration wurde gespeichert.")
        except Exception as exc: self._warning(str(exc))

    def _load_regions(self):
        try:
            data = load_regions(self.region_config.text()); self.editor.set_regions(data["crop_box"], data.get("rois", {}))
        except Exception as exc: self._warning(str(exc))

    def _start_crop(self):
        if not all((self.crop_input.text(), self.crop_output.text(), self.crop_config.text())):
            return self._warning("Bitte Eingabe, Ausgabe und Konfiguration angeben.")
        self._run(crop_frames, self.crop_input.text(), self.crop_output.text(), self.crop_config.text(),
                  self.crop_format.currentData(), self.crop_quality.value(),
                  on_result=self._crop_done)

    def _crop_done(self, count):
        self.video_input.setText(self.crop_output.text())
        self._message(f"{count} Bilder wurden zugeschnitten.\nDer Ausgabeordner wurde für Schritt 5 übernommen.")

    def _start_video(self):
        if not self.video_input.text() or not self.video_output.text():
            return self._warning("Bitte Frame-Ordner und Ausgabevideo angeben.")
        self._run(
            frames_to_video,
            self.video_input.text(), self.video_output.text(), self.video_fps.value(),
            self.video_every.value(), self.video_codec.currentData(),
            on_result=self._video_done,
        )

    def _video_done(self, path):
        self._message(f"Video wurde erstellt:\n{path}")

    def _load_mask_image(self):
        try:
            self.mask_editor.load_image(self.mask_image.text())
        except Exception as exc:
            self._warning(str(exc))

    def _select_mask_mode(self):
        """Zwischen Box und Polygon umschalten."""
        mode = self.mask_mode.currentData()
        self.mask_editor.set_mode(mode)
        self.mask_hint.setText(BOX_HINT if mode == "box" else POLYGON_HINT)

    def _choose_mask_color(self):
        """Fuellfarbe der aktiven Maske waehlen."""
        name = self.mask_name.currentText()
        chosen = QColorDialog.getColor(
            self.mask_editor.color_of(name), self,
            f"Füllfarbe für Maske {name} (Standard: schwarz)")
        if chosen.isValid():
            self.mask_editor.set_color(name, chosen)
            self._refresh_mask_color_button()

    def _refresh_mask_color_button(self):
        """Der Knopf traegt die Farbe, die er setzt -- kein Ratespiel."""
        color = self.mask_editor.color_of(self.mask_name.currentText())
        text = "#000000" if color.lightness() > 140 else "#ffffff"
        self.mask_color_button.setStyleSheet(
            f"background:{color.name()}; color:{text}; font-weight:600")

    def _select_mask(self, name):
        if name and hasattr(self, "mask_editor"):
            self.mask_editor.set_active_roi(name)
            self._refresh_mask_color_button()

    def _add_mask(self):
        name, accepted = QInputDialog.getText(self, "Neue Maske", "Maskenname:")
        name = name.strip()
        if not accepted or not name:
            return
        if name in self.mask_editor.rois:
            return self._warning(f"Die Maske {name!r} existiert bereits.")
        self.mask_editor.rois[name] = []
        # Bewusst KEINE Fuellfarbe setzen: ohne ausdrueckliche Wahl wird
        # schwarz maskiert. Die Anzeigefarbe kommt automatisch aus der
        # Palette, damit die Maske im Editor trotzdem unterscheidbar ist.
        self.mask_name.addItem(name); self.mask_name.setCurrentText(name); self.mask_editor.update()

    def _delete_mask(self):
        name = self.mask_name.currentText()
        if not name:
            return
        self.mask_editor.rois.pop(name, None)
        self.mask_editor.colors.pop(name, None)
        self.mask_editor.shapes.pop(name, None)
        index = self.mask_name.currentIndex(); self.mask_name.removeItem(index)
        if self.mask_name.count() == 0:
            self.mask_editor.rois["mask_1"] = []; self.mask_name.addItem("mask_1")
        self.mask_editor.set_active_roi(self.mask_name.currentText()); self.mask_editor.update()

    def _reset_masks(self):
        self.mask_editor.reset_regions()
        self.mask_name.blockSignals(True); self.mask_name.clear(); self.mask_name.addItems(self.mask_editor.rois.keys()); self.mask_name.blockSignals(False)
        self.mask_editor.set_active_roi(self.mask_name.currentText())

    def _load_mask_config(self):
        try:
            config = load_mask_config(self.mask_config.text())
            # BGR aus der Datei -> QColor (RGB). Masken ohne eigene Farbe
            # bekommen eine Palettenfarbe, damit sie unterscheidbar bleiben.
            colors = {}
            for index, name in enumerate(config["masks"]):
                stored = config["mask_colors"].get(name)
                colors[name] = (QColor(int(stored[2]), int(stored[1]), int(stored[0]))
                                if stored else self.mask_editor.suggest_color(index))
            self.mask_editor.set_polygons(config["masks"], colors, config["mask_shapes"])
            self.mask_name.blockSignals(True); self.mask_name.clear(); self.mask_name.addItems(config["masks"].keys()); self.mask_name.blockSignals(False)
            self.mask_editor.set_active_roi(self.mask_name.currentText())
            self._refresh_mask_color_button()
        except Exception as exc:
            self._warning(str(exc))

    def _save_masks_and_debug(self):
        if self.mask_editor.image.isNull():
            return self._warning("Bitte zuerst ein Referenzbild laden.")
        if not all((self.mask_config.text(), self.masked_output.text(), self.mask_debug_output.text())):
            return self._warning("Bitte Config- und Ausgabepfade vollständig angeben.")
        try:
            # QColor ist RGB, die Konfiguration BGR -- hier umdrehen, nicht
            # spaeter beim Zeichnen, sonst wandert die Umrechnung durchs Projekt.
            colors = {name: (color.blue(), color.green(), color.red())
                      for name, color in self.mask_editor.colors.items()}
            save_mask_config(self.mask_config.text(), self.mask_editor.image_size(),
                             self.mask_editor.rois, DEFAULT_FILL_BGR,
                             colors, self.mask_editor.shapes)
            masked, debug = create_mask_outputs(
                self.mask_image.text(), self.mask_config.text(), self.masked_output.text(), self.mask_debug_output.text()
            )
            self._message(f"Maskierung gespeichert.\n\nMaskiertes Bild: {masked}\nDebug-Bild: {debug}")
        except Exception as exc:
            self._warning(str(exc))

    def _run(self, operation, *args, on_result=None):
        if self.worker is not None: return self._warning("Es läuft bereits ein Vorgang.")
        worker = Worker(operation, *args); self.worker = worker
        worker.signals.progress.connect(self._progress)
        if on_result: worker.signals.result.connect(on_result)
        worker.signals.error.connect(self._job_error); worker.signals.finished.connect(self._job_finished)
        self.progress.setValue(0); self.progress.setVisible(True); self.cancel_button.setVisible(True)
        self.statusBar().showMessage("Vorgang läuft …"); self.thread_pool.start(worker)

    def _progress(self, value, text):
        self.progress.setValue(value); self.statusBar().showMessage(text)
        if hasattr(self, "download_log") and self.tabs.currentIndex() == 0: self.download_log.appendPlainText(text)

    def _cancel(self):
        if self.worker: self.worker.cancel.set(); self.statusBar().showMessage("Abbruch angefordert …")

    def _job_error(self, details):
        if "abgebrochen" in details.lower(): self.statusBar().showMessage(details)
        else: QMessageBox.critical(self, "Vorgang fehlgeschlagen", details)

    def _job_finished(self):
        self.worker = None; self.progress.setVisible(False); self.cancel_button.setVisible(False)
        if "fehlgeschlagen" not in self.statusBar().currentMessage().lower(): self.statusBar().showMessage("Bereit")

    def _warning(self, text): QMessageBox.warning(self, "Hinweis", text)
    def _message(self, text): QMessageBox.information(self, "Fertig", text)

    def closeEvent(self, event):
        if self.worker:
            answer = QMessageBox.question(self, "Vorgang läuft", "Vorgang abbrechen und Tool schließen?")
            if answer != QMessageBox.StandardButton.Yes: event.ignore(); return
            self.worker.cancel.set()
        event.accept()

    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #171d26; color: #e8edf4; font-size: 13px; }
            QTabWidget::pane { border: 1px solid #303b4b; }
            QTabBar::tab { background: #202936; padding: 11px 22px; color: #9cabbc; }
            QTabBar::tab:selected { background: #273446; color: #ffffff; border-bottom: 3px solid #45a3ff; }
            QLabel#heading { font-size: 24px; font-weight: 700; color: #ffffff; }
            QLabel#muted { color: #91a0b3; margin-bottom: 8px; }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit {
                background: #111720; border: 1px solid #344154; border-radius: 5px; padding: 7px; selection-background-color: #247ac5;
            }
            QPushButton { background: #293647; border: 1px solid #3b4a5e; border-radius: 5px; padding: 8px 14px; }
            QPushButton:hover { background: #34455b; }
            QPushButton:checked { background: #245f91; border-color: #55afff; }
            QPushButton#primary { background: #1676c4; border-color: #3298e9; font-weight: 600; }
            QPushButton#primary:hover { background: #2188d8; }
            QProgressBar { border: 1px solid #344154; border-radius: 4px; text-align: center; background: #111720; }
            QProgressBar::chunk { background: #2388d8; }
            QStatusBar { background: #111720; color: #9cabbc; }
        """)

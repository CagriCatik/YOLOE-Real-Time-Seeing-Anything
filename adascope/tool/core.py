"""Framework-independent data preparation operations.

This module intentionally does not import anything from ``adascope``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from threading import Event
from typing import Callable

import cv2
import numpy as np

Progress = Callable[[int, str], None]


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frames: int
    duration: float


def probe_video(path: str | Path) -> VideoInfo:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Video kann nicht geöffnet werden: {path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()
    return VideoInfo(width, height, fps, frames, frames / fps if fps else 0)


def find_ffmpeg(explicit: str | Path | None = None) -> Path | None:
    """Resolve FFmpeg from a user value, PATH, or common Windows locations."""
    candidates = []
    if explicit:
        value = Path(explicit).expanduser()
        candidates.extend([value, value / "ffmpeg.exe"] if value.is_dir() else [value])
    discovered = shutil.which("ffmpeg")
    if discovered:
        candidates.append(Path(discovered))
    candidates.extend([
        Path(r"C:\ffmpeg\bin\ffmpeg.exe"),
        Path(r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"),
    ])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def download_video(url: str, output_dir: str | Path, filename: str, max_height: int,
                   ffmpeg_location: str | Path | None,
                   progress: Progress, cancel: Event) -> Path:
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError("yt-dlp fehlt. Bitte tool/requirements.txt installieren.") from exc

    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    stem = Path(filename.strip() or "video").stem
    template = str(folder / f"{stem}.%(ext)s")

    def hook(status):
        if cancel.is_set():
            raise InterruptedError("Download abgebrochen")
        if status.get("status") == "downloading":
            total = status.get("total_bytes") or status.get("total_bytes_estimate") or 0
            downloaded = status.get("downloaded_bytes") or 0
            percent = int(downloaded * 100 / total) if total else 0
            progress(percent, f"Download: {percent}%")
        elif status.get("status") == "finished":
            progress(96, "Download abgeschlossen, Video wird zusammengeführt …")

    ffmpeg = find_ffmpeg(ffmpeg_location)
    if ffmpeg is None:
        raise RuntimeError(
            "FFmpeg wurde nicht gefunden. Bitte im Download-Bereich den Pfad "
            "zur ffmpeg.exe angeben."
        )
    progress(0, f"FFmpeg: {ffmpeg}")
    options = {
        "format": f"bestvideo[height<={max_height}]+bestaudio/best[height<={max_height}]",
        "outtmpl": template,
        "merge_output_format": "mp4",
        "progress_hooks": [hook],
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "ffmpeg_location": str(ffmpeg),
    }
    with yt_dlp.YoutubeDL(options) as client:
        info = client.extract_info(url, download=True)
        prepared = Path(client.prepare_filename(info))
    mp4 = folder / f"{stem}.mp4"
    result = mp4 if mp4.exists() else prepared
    progress(100, f"Gespeichert: {result}")
    return result


def extract_frames(video: str | Path, output_dir: str | Path, every: int = 1,
                   target_fps: float = 0, extension: str = "jpg", quality: int = 95,
                   progress: Progress = lambda *_: None, cancel: Event | None = None) -> int:
    cancel = cancel or Event()
    if every < 1:
        raise ValueError("Frame-Abstand muss mindestens 1 sein")
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Video kann nicht geöffnet werden: {video}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, round(source_fps / target_fps)) if target_fps > 0 and source_fps > 0 else every
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    ext = extension.lower().lstrip(".")
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext in {"jpg", "jpeg"} else []
    source_index = written = 0
    while not cancel.is_set():
        ok, frame = capture.read()
        if not ok:
            break
        if source_index % step == 0:
            target = folder / f"frame_{source_index:06d}.{ext}"
            if not cv2.imwrite(str(target), frame, params):
                raise OSError(f"Frame konnte nicht geschrieben werden: {target}")
            written += 1
        source_index += 1
        if source_index % 25 == 0:
            percent = int(source_index * 100 / total) if total else 0
            progress(min(percent, 99), f"{written} Frames extrahiert")
    capture.release()
    if cancel.is_set():
        raise InterruptedError("Extraktion abgebrochen")
    progress(100, f"Fertig: {written} Frames")
    return written


def list_images(folder: str | Path) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    path = Path(folder)
    if not path.is_dir():
        return []
    return sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() in extensions)


def save_regions(path: str | Path, image_size: tuple[int, int], crop_box,
                 rois: dict[str, list[list[float]]]) -> None:
    data = {
        "version": 1,
        "coordinate_system": "normalized",
        "source_size": {"width": image_size[0], "height": image_size[1]},
        "crop_box": [round(float(value), 6) for value in crop_box],
        "rois": {
            name: [[round(float(x), 6), round(float(y), 6)] for x, y in points]
            for name, points in rois.items()
        },
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_regions(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    crop = data.get("crop_box")
    if not isinstance(crop, list) or len(crop) != 4:
        raise ValueError("Konfiguration enthält keine gültige crop_box")
    x0, y0, x1, y1 = map(float, crop)
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError("crop_box muss normalisiert sein: 0 ≤ x0 < x1 ≤ 1")
    data["crop_box"] = [x0, y0, x1, y1]
    data.setdefault("rois", {})
    return data


def crop_frames(input_dir: str | Path, output_dir: str | Path, config_path: str | Path,
                extension: str = "same", quality: int = 95,
                progress: Progress = lambda *_: None, cancel: Event | None = None) -> int:
    cancel = cancel or Event()
    config = load_regions(config_path)
    x0, y0, x1, y1 = config["crop_box"]
    paths = list_images(input_dir)
    if not paths:
        raise ValueError("Keine Bilder im Eingabeordner gefunden")
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    written = 0
    for index, path in enumerate(paths):
        if cancel.is_set():
            raise InterruptedError("Batch-Cropping abgebrochen")
        image = cv2.imread(str(path))
        if image is None:
            continue
        height, width = image.shape[:2]
        left, top = round(x0 * width), round(y0 * height)
        right, bottom = round(x1 * width), round(y1 * height)
        cropped = image[top:bottom, left:right]
        ext = path.suffix.lower().lstrip(".") if extension == "same" else extension.lower().lstrip(".")
        target = folder / f"{path.stem}.{ext}"
        params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext in {"jpg", "jpeg"} else []
        if not cv2.imwrite(str(target), cropped, params):
            raise OSError(f"Bild konnte nicht geschrieben werden: {target}")
        written += 1
        progress(int((index + 1) * 100 / len(paths)), f"{written}/{len(paths)} Bilder zugeschnitten")
    return written


def frames_to_video(input_dir: str | Path, output_file: str | Path, fps: float = 25,
                    every: int = 1, codec: str = "mp4v",
                    progress: Progress = lambda *_: None,
                    cancel: Event | None = None) -> Path:
    """Assemble a naturally sorted image sequence into a silent video."""
    cancel = cancel or Event()
    if fps <= 0:
        raise ValueError("FPS muss größer als 0 sein")
    if every < 1:
        raise ValueError("Frame-Abstand muss mindestens 1 sein")
    if len(codec) != 4:
        raise ValueError("Der Video-Codec muss ein FourCC-Code mit vier Zeichen sein")
    paths = list_images(input_dir)[::every]
    if not paths:
        raise ValueError("Keine Bilder im Eingabeordner gefunden")
    first = cv2.imread(str(paths[0]))
    if first is None:
        raise ValueError(f"Erstes Bild kann nicht gelesen werden: {paths[0]}")
    height, width = first.shape[:2]
    target = Path(output_file)
    if not target.suffix:
        target = target.with_suffix(".mp4")
    target.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(target), cv2.VideoWriter_fourcc(*codec), float(fps), (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(
            f"Video kann nicht erstellt werden: {target}. "
            f"Der Codec {codec!r} wird möglicherweise nicht unterstützt."
        )
    written = 0
    try:
        for index, path in enumerate(paths):
            if cancel.is_set():
                raise InterruptedError("Videoerstellung abgebrochen")
            image = cv2.imread(str(path))
            if image is None:
                continue
            if image.shape[:2] != (height, width):
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(image)
            written += 1
            progress(int((index + 1) * 100 / len(paths)), f"{written}/{len(paths)} Frames geschrieben")
    finally:
        writer.release()
    if written == 0:
        target.unlink(missing_ok=True)
        raise ValueError("Kein lesbares Bild gefunden")
    progress(100, f"Video gespeichert: {target}")
    return target


MASK_CONFIG_VERSION = 2
DEFAULT_FILL_BGR = (0, 0, 0)


def box_polygon(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
    """Rechteck als Vierpunkt-Polygon, im Uhrzeigersinn ab oben links.

    Eine Box ist damit KEIN eigener Datentyp -- sie ist ein Polygon mit vier
    Ecken. Alles Nachgelagerte (Fuellen, Debugbild, Laden aelterer Dateien)
    bleibt unveraendert; nur das Zeichnen im Editor kennt den Unterschied.
    """
    left, right = sorted((float(x0), float(x1)))
    top, bottom = sorted((float(y0), float(y1)))
    return [[left, top], [right, top], [right, bottom], [left, bottom]]


def save_mask_config(path: str | Path, image_size: tuple[int, int],
                     masks: dict[str, list[list[float]]], fill_color=DEFAULT_FILL_BGR,
                     mask_colors: dict[str, tuple[int, int, int]] | None = None,
                     mask_shapes: dict[str, str] | None = None) -> None:
    """Benannte normalisierte Maskenpolygone als eigenstaendige JSON speichern.

    `mask_colors` gibt je Maske eine eigene Fuellfarbe (BGR). Was hier fehlt,
    faellt auf `fill_color` zurueck -- so bleiben Dateien ohne Farbzuordnung
    gueltig.

    `mask_shapes` merkt sich, welche Maske als Box gezeichnet wurde. Nur der
    Editor wertet das aus: eine Box soll beim Ziehen einer Ecke rechteckig
    bleiben statt zu einem beliebigen Viereck zu werden.
    """
    valid_masks = {}
    for name, points in masks.items():
        if not name.strip():
            raise ValueError("Maskenname darf nicht leer sein")
        if points and len(points) < 3:
            raise ValueError(f"Maske {name!r} benötigt mindestens drei Punkte")
        normalized = []
        for x, y in points:
            if not 0 <= float(x) <= 1 or not 0 <= float(y) <= 1:
                raise ValueError(f"Punkt in Maske {name!r} liegt außerhalb von 0..1")
            normalized.append([round(float(x), 6), round(float(y), 6)])
        if normalized:
            valid_masks[name] = normalized
    if not valid_masks:
        raise ValueError("Mindestens eine Maske mit drei Punkten ist erforderlich")

    colors = {}
    for name, value in (mask_colors or {}).items():
        if name not in valid_masks:
            continue                        # Farbe einer geloeschten Maske
        channels = [int(v) for v in value]
        if len(channels) != 3 or any(not 0 <= v <= 255 for v in channels):
            raise ValueError(f"Farbe von Maske {name!r} muss BGR in 0..255 sein")
        colors[name] = channels
    shapes = {name: str(shape) for name, shape in (mask_shapes or {}).items()
              if name in valid_masks and shape in ("box", "polygon")}

    data = {
        "version": MASK_CONFIG_VERSION,
        "coordinate_system": "normalized",
        "source_size": {"width": image_size[0], "height": image_size[1]},
        "fill_color_bgr": [int(value) for value in fill_color],
        "masks": valid_masks,
        "mask_colors": colors,
        "mask_shapes": shapes,
    }
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_mask_config(path: str | Path) -> dict:
    """Maskenkonfiguration lesen. Dateien der Version 1 bleiben gueltig."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    masks = data.get("masks")
    if not isinstance(masks, dict) or not masks:
        raise ValueError("Konfiguration enthält keine Masken")
    for name, points in masks.items():
        if not isinstance(points, list) or len(points) < 3:
            raise ValueError(f"Maske {name!r} benötigt mindestens drei Punkte")
        if any(len(point) != 2 or any(not 0 <= float(value) <= 1 for value in point) for point in points):
            raise ValueError(f"Maske {name!r} enthält ungültige normalisierte Punkte")
    data.setdefault("fill_color_bgr", list(DEFAULT_FILL_BGR))
    data.setdefault("mask_colors", {})
    data.setdefault("mask_shapes", {})
    unknown = set(data["mask_colors"]) - set(masks)
    if unknown:
        raise ValueError(f"Farben ohne zugehoerige Maske: {sorted(unknown)}")
    return data


def mask_fill_colors(config: dict) -> dict[str, tuple[int, int, int]]:
    """Fuellfarbe je Maske -- die eigene, sonst die globale (Standard schwarz)."""
    fallback = tuple(int(v) for v in config["fill_color_bgr"])
    return {name: tuple(int(v) for v in config["mask_colors"].get(name, fallback))
            for name in config["masks"]}


# Palette fuer das Debugbild, sobald eine Maske keine eigene Farbe hat oder
# ihre Fuellfarbe zu dunkel zum Erkennen ist. BGR.
DEBUG_PALETTE = [(77, 65, 255), (255, 167, 72), (122, 220, 93), (102, 209, 255),
                 (255, 125, 199), (238, 210, 34), (22, 115, 249), (171, 143, 255)]


def mask_display_colors(config: dict) -> dict[str, tuple[int, int, int]]:
    """Farbe zum ANZEIGEN je Maske -- immer erkennbar.

    Das Debugbild soll zeigen, WO maskiert wird. Eine schwarze Fuellfarbe
    waere dort nicht auffindbar, deshalb tritt hier eine Palettenfarbe an
    ihre Stelle. Die Fuellfarbe des Ergebnisbildes bleibt davon unberuehrt.
    """
    fills = mask_fill_colors(config)
    display = {}
    for index, name in enumerate(config["masks"]):
        blue, green, red = fills[name]
        too_dark = max(blue, green, red) < 60
        chosen = name in config["mask_colors"]
        display[name] = (DEBUG_PALETTE[index % len(DEBUG_PALETTE)]
                         if too_dark or not chosen else fills[name])
    return display


def create_mask_outputs(image_path: str | Path, config_path: str | Path,
                        masked_output: str | Path, debug_output: str | Path) -> tuple[Path, Path]:
    """Apply configured masks and create a red-overlay debug visualization."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Bild kann nicht gelesen werden: {image_path}")
    config = load_mask_config(config_path)
    height, width = image.shape[:2]
    colors = mask_fill_colors(config)
    display = mask_display_colors(config)
    pixel_polygons = {
        name: np.asarray([[round(x * width), round(y * height)] for x, y in points],
                         dtype=np.int32)
        for name, points in config["masks"].items()
    }

    # Je Maske einzeln fuellen statt ueber eine gemeinsame Binaermaske: nur so
    # bekommt jede ihre eigene Farbe. Bei Ueberlappung gewinnt die spaetere --
    # dieselbe Reihenfolge wie in der Maskenliste des Editors.
    masked = image.copy()
    overlay = image.copy()
    for name, polygon in pixel_polygons.items():
        cv2.fillPoly(masked, [polygon], colors[name])       # Ergebnisbild
        cv2.fillPoly(overlay, [polygon], display[name])     # Debugbild
    debug = cv2.addWeighted(overlay, .45, image, .55, 0)

    for name, polygon in pixel_polygons.items():
        outline = display[name]
        cv2.polylines(debug, [polygon], True, outline, 3)
        for x, y in polygon:
            cv2.circle(debug, (int(x), int(y)), 5, (255, 255, 255), -1)
            cv2.circle(debug, (int(x), int(y)), 5, outline, 2)
        x, y = polygon[0]
        kind = "Box" if config["mask_shapes"].get(name) == "box" else "Polygon"
        label = f"{name} ({kind})"
        cv2.putText(debug, label, (int(x) + 7, max(20, int(y) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(debug, label, (int(x) + 7, max(20, int(y) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, outline, 1, cv2.LINE_AA)

    masked_target = Path(masked_output); debug_target = Path(debug_output)
    masked_target.parent.mkdir(parents=True, exist_ok=True)
    debug_target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(masked_target), masked):
        raise OSError(f"Maskiertes Bild konnte nicht gespeichert werden: {masked_target}")
    if not cv2.imwrite(str(debug_target), debug):
        raise OSError(f"Debug-Bild konnte nicht gespeichert werden: {debug_target}")
    return masked_target, debug_target

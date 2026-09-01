"""Orchestrierung eines Debug-Laufs: Quelle rein, Videos und Kennzahlen raus.

Der einzige Ort, an dem Pipeline, Renderer und Dateiausgabe zusammenkommen.
`cli.lane_debug` und `cli.scenarios` rufen beide hierher -- so laufen ein
einzelner Aufruf und ein Szenario-Durchlauf garantiert durch denselben Code
und liefern vergleichbare Zahlen.
"""

from __future__ import annotations

from collections import Counter
from itertools import chain
from pathlib import Path
from typing import Callable

from .config import Settings
from .io import VideoWriter, iter_source, write_named, write_rows
from .lanes import SequencePipeline
from .render import available_views, make_view
from .ground_truth import GroundTruth, score_events
from .perception_ground_truth import PERCEPTION_FIELDS, score_perception
from .scenarios import RunSummary

METRIC_FIELDS = [
    "frame", "source", "h_state", "held_frames", "n_lines", "n_boundaries",
    "n_corridors", "ego_lane_pos", "n_lanes_rel", "n_synthetic", "lane_width",
    "ego_in_lane", "index_note", "n_vehicles", "n_valid", "worst_state",
    "n_events", "n_boundary_ids", "ego_motion", "ego_shift_lanes",
    "ego_shift_spread", "rel_indices", "boundaries",
]
# FR-5.1 verlangt {fahrzeug, richtung, grenze_id, frames} im Ereignis --
# also muessen sie auch in der Datei stehen, nicht nur im Speicher.
EVENT_FIELDS = ["frame", "frame_start", "frame_end", "kind", "track", "direction",
                "boundary_id", "confidence", "certain", "detail"]
# FR-1.4: Per-Frame-Zustand je Fahrzeug, EGO und CO im selben Satz.
STATE_FIELDS = ["frame_id", "fahrzeug", "lateral_pos", "aktive_grenze_id",
                "confidence"]


def resolve_views(names: str, settings: Settings) -> list[str]:
    """Kommaseparierte Auswahl oder 'all' zu einer geprueften Liste."""
    known = available_views(settings)
    if names == "all":
        return known
    chosen = [n.strip() for n in names.split(",") if n.strip()]
    unknown = [n for n in chosen if n not in known]
    if unknown:
        raise ValueError(f"Unbekannte Ansicht(en): {unknown}. Verfuegbar: {known}")
    return chosen


def crop_frame(frame, box: tuple[float, float, float, float]):
    """Normierten Ausschnitt [x0,y0,x1,y1] aus einem Frame schneiden.

    Der Weg, unbeschnittene Aufnahmen auf den kalibrierten Bildausschnitt zu
    bringen: `lane.yaml` ist in Pixeln dieses Zuschnitts notiert, ein
    Vollbild zeigt daneben noch Fahrzeuginnenraum und Bildschirmrand.
    """
    height, width = frame.shape[:2]
    x0, y0, x1, y1 = box
    return frame[round(y0 * height):round(y1 * height),
                 round(x0 * width):round(x1 * width)]


def choose_crop(settings: Settings, width: int, height: int
                ) -> tuple[float, float, float, float] | None:
    """Entscheidet je QUELLE, ob zugeschnitten werden muss.

    Der Zuschnitt haengt an der Aufnahme, nicht am Aufruf: dieselbe Sammlung
    enthaelt fertig zugeschnittene Ausschnitte und unbeschnittene Vollbilder.
    Ein globaler Schalter wuerde die einen richtig und die anderen doppelt
    beschneiden -- deshalb wird hier gefragt, nicht dort entschieden.

        passt schon                      -> nicht schneiden
        passt nach detection.crop_box    -> schneiden
        passt so oder so nicht           -> None, `scaled_to()` erklaert es
    """
    if settings.lane.matches(width, height):
        return None
    if settings.detection is None:
        return None
    box = settings.detection.crop_box
    x0, y0, x1, y1 = box
    cropped = (round((x1 - x0) * width), round((y1 - y0) * height))
    return box if settings.lane.matches(*cropped) else None


def run_debug(source: Path, settings: Settings, views: list[str], outdir: Path,
              tracker=None, stride: int = 1, max_frames: int = 0, fps: float = 25.0,
              label: str = "", crop: tuple[float, float, float, float] | None = None,
              auto_crop: bool = True, truth: GroundTruth | None = None,
              on_progress: Callable[[str], None] | None = None) -> RunSummary:
    """Fuehrt die Pipeline ueber eine Quelle aus und schreibt alle Ausgaben.

    Rueckgabe ist die Zusammenfassung -- Kennzahlen statt Dateipfade, damit der
    Aufrufer mehrere Laeufe vergleichen kann, ohne CSVs nachzulesen.
    """
    outdir = Path(outdir)
    frames, source_fps = iter_source(source, stride, fps)

    # Am ersten Frame entscheiden, ob diese Quelle zugeschnitten werden muss,
    # und ihn danach wieder vorne einhaengen.
    if crop is None and auto_crop:
        first = next(iter(frames), None)
        if first is None:
            return RunSummary(scenario=label or Path(source).stem,
                              error="Quelle enthaelt keine lesbaren Frames")
        height, width = first[1].shape[:2]
        crop = choose_crop(settings, width, height)
        if crop is not None and on_progress:
            x0, y0, x1, y1 = crop
            on_progress(f"    Zuschnitt {width}x{height} -> "
                        f"{round((x1 - x0) * width)}x{round((y1 - y0) * height)} "
                        "(detection.crop_box)")
        frames = chain([first], frames)
    pipeline = SequencePipeline(settings)
    renderers = {name: make_view(name, settings) for name in views}
    writers = {name: VideoWriter(outdir / f"debug_{name}.mp4", source_fps / max(stride, 1))
               for name in views}

    rows: list[dict] = []
    states: list[dict] = []
    perception_analyses = {}
    perception_frames = ({item.frame for item in truth.perception}
                         if truth is not None else set())
    try:
        for index, name, frame in _limited(frames, stride, max_frames):
            if crop is not None:
                frame = crop_frame(frame, crop)
            vehicles = tracker.update(frame) if tracker else []
            analysis = pipeline.process(index, name, frame, vehicles)
            for view, render in renderers.items():
                writers[view].write(render(analysis))
            rows.append(_metrics_row(analysis))
            states.extend(analysis.states())
            if index in perception_frames:
                perception_analyses[index] = analysis
            if on_progress:
                for event in analysis.events:
                    on_progress(f"    {event}")
                if len(rows) == 1 or len(rows) % 100 == 0:
                    on_progress(f"    {len(rows):5d} Frames | H={analysis.h_state:5s} "
                                f"Korridore={len(analysis.corridors)} "
                                f"Spuren={len(analysis.lanes_rel)}")
    finally:
        for writer in writers.values():
            writer.close()

    write_named(rows, outdir / "debug_metrics.csv", METRIC_FIELDS)
    write_named(states, outdir / "debug_states.csv", STATE_FIELDS)
    write_named([_event_row(e) for e in pipeline.log],
                outdir / "debug_events.csv", EVENT_FIELDS)

    summary = summarise(label or Path(source).stem, rows, pipeline.log)
    perception_path = None
    if truth is not None:
        summary.score = score_events(truth, pipeline.log)
        if truth.perception:
            summary.perception_score = score_perception(
                truth.perception, perception_analyses, truth.acceptance)
            perception_path = outdir / "debug_perception.csv"
            write_named([row.as_row() for row in summary.perception_score.measurements],
                        perception_path, PERCEPTION_FIELDS)
    summary.outputs = [writers[name].path for name in views] + [
        outdir / "debug_metrics.csv", outdir / "debug_states.csv",
        outdir / "debug_events.csv"]
    if perception_path is not None:
        summary.outputs.append(perception_path)
    (outdir / "summary.txt").write_text(summary.as_text(), encoding="utf-8")
    return summary


def summarise(label: str, rows: list[dict], events: list) -> RunSummary:
    """Aggregiert die Per-Frame-Zeilen zu den Kennzahlen eines Laufs."""
    summary = RunSummary(scenario=label, frames=len(rows))
    if not rows:
        summary.error = "keine Frames verarbeitet"
        return summary

    summary.homography = dict(Counter(r["h_state"] for r in rows))
    usable = [r for r in rows if r["n_lanes_rel"] > 0]
    summary.usable_lanes = len(usable)

    # Nur zwischen Frames vergleichen, die beide einen gueltigen Index haben --
    # sonst zaehlt jeder Stufenausfall als Sprung.
    positional = [r for r in rows if r["n_corridors"] >= 2 and r["ego_lane_pos"] >= 0]
    summary.index_transitions = max(len(positional) - 1, 0)
    summary.index_jumps = sum(1 for a, b in zip(positional, positional[1:])
                              if a["ego_lane_pos"] != b["ego_lane_pos"])

    if usable:
        widths = sorted(r["lane_width"] for r in usable)
        summary.lane_width_median = widths[len(widths) // 2]
        ego = [r["ego_in_lane"] for r in usable]
        summary.ego_in_lane_min = min(ego)
        summary.ego_departing_frames = sum(1 for v in ego if v < 1.0)

    summary.events = dict(Counter(e.kind for e in events))
    return summary


def _limited(frames, stride: int, max_frames: int):
    """(index, name, frame); der Index bleibt der Index der QUELLE.

    Nicht der Zaehler der verarbeiteten Frames -- sonst passen die Zeitstempel
    in CSV und Video nicht mehr zur Vorlage, sobald `--stride` gesetzt ist.
    """
    processed = 0
    for index, (name, frame) in enumerate(frames):
        if index % max(stride, 1):
            continue
        if max_frames and processed >= max_frames:
            return
        yield index, name, frame
        processed += 1


def _event_row(event) -> dict:
    """Ereignis als CSV-Zeile -- mit allen Feldern aus FR-5.1."""
    start, end = event.frames
    return {
        "frame": event.frame, "frame_start": start, "frame_end": end,
        "kind": event.kind, "track": event.track or "EGO",
        "direction": event.direction, "boundary_id": event.boundary_id,
        "confidence": round(event.confidence, 3),
        "certain": int(event.certain), "detail": event.detail,
    }


def _metrics_row(fa) -> dict:
    return {
        "frame": fa.index, "source": fa.name, "h_state": fa.h_state,
        "held_frames": fa.held_frames, "n_lines": len(fa.lanes.lines),
        "n_boundaries": len(fa.boundaries), "n_corridors": len(fa.corridors),
        "ego_lane_pos": fa.ego_lane_pos, "n_lanes_rel": len(fa.lanes_rel),
        "n_synthetic": sum(L.synthetic for L in fa.lanes_rel),
        "lane_width": round(fa.lane_width, 1), "ego_in_lane": round(fa.ego_in_lane, 3),
        "index_note": fa.index_note, "n_vehicles": len(fa.vehicles),
        "n_valid": sum(o.valid for o in fa.occupancies), "worst_state": fa.worst_state,
        "n_events": len(fa.events),
        "n_boundary_ids": len(fa.boundary_ids),
        "ego_motion": fa.ego_motion.verdict,
        "ego_shift_lanes": round(fa.ego_motion.shift_lanes, 3),
        "ego_shift_spread": round(fa.ego_motion.spread, 3),
        "rel_indices": " ".join(f"{o.track}:{o.rel}" for o in fa.occupancies
                                if o.valid and o.rel is not None),
        "boundaries": " ".join(f"{b:.0f}" for b in fa.boundaries),
    }

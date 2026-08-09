"""Debug-Ansichten der Spur-Pipeline, je eine Perspektive pro Renderer.

Jeder Renderer nimmt eine `FrameAnalysis` und liefert ein BGR-Bild fester
Groesse (VideoWriter braucht konstante Frames). Alle sind tolerant gegenueber
`H is None`, leeren Korridoren und fehlgeschlagener Indizierung -- im
Videobetrieb ist das der Normalfall, nicht die Ausnahme.

    front     Bildebene: Hough-Linien, Rollen, Tracks, Bbox-Unterkante
    mask      Stufe 1/2: Weissmaske, Canny, Hough-Segmente nach Cluster
    bev       Bodenebene von oben: Rohkorridore gegen ego-relative Spuren
    hist      Stufe 3: BEV-Maske + Spaltenhistogramm mit Peaks
    smear     BEV der Farbbilder: warum Fahrzeuge dort nicht segmentiert
              werden duerfen (Bauhoehe zerlaeuft radial)
    dash      Komposit + Zeitverlauf + Ereignislog
    <kamera>  jede in config/debug.yaml definierte virtuelle Kamera

Farben und Kameras kommen aus `DebugConfig`, nicht aus Konstanten hier -- eine
neue Perspektive entsteht durch einen YAML-Eintrag.
"""

from __future__ import annotations

from collections import deque
from functools import partial
from typing import Callable

import cv2
import numpy as np

from ..config import Settings
from ..lanes import FrameAnalysis, Lane
from ..lanes.detection import build_masked_edges, cluster_segments, extract_segments
from .camera import VirtualCamera
from .primitives import FONT, dashed_line, dashed_polyline, fit, hud, placeholder

Renderer = Callable[[FrameAnalysis], np.ndarray]

# Zyklisch fuer die Hough-Cluster in der Maskenansicht; rein visuell, ohne
# Bedeutung -- deshalb hier und nicht in der Config.
_CLUSTER_COLORS = [(0, 255, 255), (255, 120, 0), (0, 255, 120), (255, 0, 255),
                   (120, 180, 255), (255, 255, 0), (0, 140, 255), (180, 100, 255)]


# --------------------------------------------------------------------------- #
# Gemeinsame Bausteine                                                        #
# --------------------------------------------------------------------------- #
def state_color(settings: Settings, occ) -> tuple[int, int, int]:
    if not occ.valid:
        return settings.debug.color("state_invalid")
    return settings.debug.color(f"state_{occ.state}" if occ.state else "state_unknown")


def ego_color(settings: Settings, fa: FrameAnalysis) -> tuple[int, int, int]:
    """Gelb solange das Ego vollstaendig in seiner Spur liegt, sonst orange."""
    key = "ego" if fa.ego_in_lane >= 1.0 else "ego_departing"
    return settings.debug.color(key)


def role_color(settings: Settings, role: str) -> tuple[int, int, int]:
    if role.endswith("solid"):
        return settings.debug.color("role_solid")
    if role.endswith("dashed"):
        return settings.debug.color("role_dashed")
    return settings.debug.color("role_unknown")


def ego_lane_of(fa: FrameAnalysis) -> Lane | None:
    return next((L for L in fa.lanes_rel if L.rel == 0), None)


def state_banner(canvas: np.ndarray, fa: FrameAnalysis, settings: Settings) -> None:
    debug = settings.debug
    held = f" ({fa.held_frames})" if fa.h_state == "held" else ""
    lanes = f"Spuren: {len(fa.lanes_rel)}" if fa.lanes_rel else f"Spuren: -- {fa.index_note}"
    hud(canvas, [
        (f"frame {fa.index}  {fa.name}", (255, 255, 255)),
        (f"H: {fa.h_state}{held}", debug.color(f"homography_{fa.h_state}")),
        (f"Korridore: {len(fa.corridors)}   {lanes}",
         (255, 255, 255) if fa.lanes_rel else debug.color("homography_held")),
        (f"Ego in Spur: {fa.ego_in_lane:.2f}", ego_color(settings, fa)),
        (f"Zustand: {fa.worst_state}", debug.color(f"state_{fa.worst_state}")),
    ])


def event_log(canvas: np.ndarray, events: list, settings: Settings,
              x: int, y: int) -> None:
    """Die letzten Ereignisse; sie sind das eigentliche Produkt der Pipeline."""
    recent = events[-settings.debug.event_log_size:]
    if not recent:
        return
    lines = [("Ereignisse:", settings.debug.color("caption"))]
    lines += [(str(e), settings.debug.color(f"event_{e.kind}")) for e in recent]
    hud(canvas, lines, x=x, y=y, scale=0.42)


def no_homography(settings: Settings, width: int, height: int) -> np.ndarray:
    return placeholder(width, height, "keine Homographie",
                       settings.debug.color("homography_none"))


# --------------------------------------------------------------------------- #
# Bildebene                                                                   #
# --------------------------------------------------------------------------- #
def view_front(fa: FrameAnalysis, settings: Settings) -> np.ndarray:
    lane, debug = settings.lane, settings.debug
    out = fa.image.copy()
    cv2.polylines(out, [np.array(lane.roi_polygon, np.int32)], True, debug.color("roi"), 1)

    if fa.lanes.ego_left and fa.lanes.ego_right:
        fill = out.copy()
        cv2.fillPoly(fill, np.array([[
            (fa.lanes.ego_left.x_at(lane.y_bottom), lane.y_bottom),
            (fa.lanes.ego_left.x_at(lane.y_top), lane.y_top),
            (fa.lanes.ego_right.x_at(lane.y_top), lane.y_top),
            (fa.lanes.ego_right.x_at(lane.y_bottom), lane.y_bottom),
        ]], np.int32), (0, 120, 0))
        out = cv2.addWeighted(fill, 0.28, out, 0.72, 0)

    # Die beiden Linien, aus denen H gerechnet wurde, dicker zeichnen.
    solids = [L for L in fa.lanes.lines if L.role.endswith("solid")]
    used = ({min(L.x_bottom for L in solids), max(L.x_bottom for L in solids)}
            if solids and fa.h_state == "fresh" else set())
    for L in fa.lanes.lines:
        color = role_color(settings, L.role)
        cv2.line(out, (L.x_at(lane.y_bottom), lane.y_bottom),
                 (L.x_at(lane.y_top), lane.y_top), color, 4 if L.x_bottom in used else 2)
        cv2.putText(out, f"{L.role} s{L.support}",
                    (L.x_at(lane.y_bottom) - 30, lane.y_bottom + 16),
                    FONT, 0.38, color, 1, cv2.LINE_AA)

    by_track = {o.track: o for o in fa.occupancies}
    for vehicle in fa.vehicles:
        x1, y1, x2, y2 = vehicle.bbox
        occ = by_track.get(f"ID{vehicle.track_id}")
        if vehicle.role == "ego":
            color, label = ego_color(settings, fa), f"EGO {vehicle.track_id} {fa.ego_in_lane:.2f}"
        elif occ is not None:
            color, label = state_color(settings, occ), occ.label
        else:
            color, label = debug.color("untracked"), f"ID{vehicle.track_id}"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 1)
        # Nur die Unterkante ist die Bruecke in die Bodenebene -> hervorheben.
        cv2.line(out, (x1, y2), (x2, y2), color, 3)
        cv2.putText(out, label, (x1, max(12, y1 - 5)), FONT, 0.38, color, 1, cv2.LINE_AA)

    state_banner(out, fa, settings)
    return out


def view_mask(fa: FrameAnalysis, settings: Settings) -> np.ndarray:
    lane, debug = settings.lane, settings.debug
    out = cv2.cvtColor((fa.mask * 0.45).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    edges = build_masked_edges(fa.image, lane)
    out[edges > 0] = (70, 70, 70)

    clusters = cluster_segments(extract_segments(edges, lane), lane)
    for i, cluster in enumerate(clusters):
        color = _CLUSTER_COLORS[i % len(_CLUSTER_COLORS)]
        weak = len(cluster) < lane.min_cluster_support
        for _, _, (x1, y1, x2, y2) in cluster:
            cv2.line(out, (x1, y1), (x2, y2), (60, 60, 60) if weak else color, 2)
        if weak:                       # von fit_lanes() verworfen -> markieren
            _, _, (x1, y1, _, _) = cluster[0]
            cv2.putText(out, "verworfen", (x1 - 20, y1 - 4), FONT, 0.35,
                        debug.color("state_inside"), 1, cv2.LINE_AA)

    cv2.polylines(out, [np.array(lane.roi_polygon, np.int32)], True, debug.color("roi"), 1)
    for vehicle in fa.vehicles:        # aus der Maske ausgestanzte Bereiche
        x1, y1, x2, y2 = vehicle.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), (40, 40, 130), 1)

    hud(out, [(f"frame {fa.index}", (255, 255, 255)),
              (f"Segmente: {fa.lanes.debug['n_segments']}", (255, 255, 255)),
              (f"Cluster: {len(clusters)} -> Linien: {len(fa.lanes.lines)}", (255, 255, 255)),
              (f"grau = unter min_cluster_support ({lane.min_cluster_support})",
               debug.color("state_invalid"))])
    return out


# --------------------------------------------------------------------------- #
# Bodenebene von oben                                                         #
# --------------------------------------------------------------------------- #
def _draw_bev_geometry(bev: np.ndarray, fa: FrameAnalysis, settings: Settings) -> None:
    """Rohkorridore und rekonstruierte Spuren uebereinander.

    Die Gegenueberstellung ist der Zweck der Ansicht: grau ist, was das
    Spaltenhistogramm liefert, weiss/orange ist, was `lanes.indexing` daraus
    macht. Orange gestrichelt sind virtuelle Grenzen -- dort war ein Korridor
    ein Vielfaches der Spurbreite breit und wurde wieder aufgeteilt.
    """
    debug = settings.debug
    height = bev.shape[0]
    # Rohgrenzen als KURVE zeichnen, nicht als senkrechte Linie: bei
    # `method: windows` sind es Polynome, und eine Gerade daraus zu machen
    # waere genau die Vereinfachung, die das Verfahren aufheben soll.
    for index in range(len(fa.boundary_fit)):
        points = fa.boundary_fit.polyline(index, 0, height).astype(np.int32)
        cv2.polylines(bev, [points], False, debug.color("boundary_raw"), 1)

    ego = ego_lane_of(fa)
    if ego is not None:
        fill = bev.copy()
        cv2.rectangle(fill, (int(ego.x_lo), 0), (int(ego.x_hi), height), (0, 110, 0), -1)
        bev[:] = cv2.addWeighted(fill, 0.38, bev, 0.62, 0)

    for L in fa.lanes_rel:
        for x in (L.x_lo, L.x_hi):
            if L.synthetic:
                dashed_line(bev, (int(x), 0), (int(x), height),
                            debug.color("boundary_synthetic"))
            else:
                cv2.line(bev, (int(x), 0), (int(x), height),
                         debug.color("boundary_lane"), 1)
        cv2.putText(bev, L.label, (int((L.x_lo + L.x_hi) / 2) - 20, height - 8), FONT,
                    0.42, (0, 255, 255) if L.rel == 0 else (170, 170, 170), 1, cv2.LINE_AA)

    if fa.ego_footprint:
        fp, color = fa.ego_footprint, ego_color(settings, fa)
        cv2.line(bev, (int(fp.x_left), int(fp.y)), (int(fp.x_right), int(fp.y)), color, 5)
        cv2.putText(bev, f"EGO {fa.ego_in_lane:.2f}", (int(fp.x_left), int(fp.y) + 18),
                    FONT, 0.45, color, 1, cv2.LINE_AA)

    for occ in fa.occupancies:
        fp, color = occ.footprint, state_color(settings, occ)
        y = int(fp.y)
        cv2.line(bev, (int(fp.x_left), y), (int(fp.x_right), y), color, 5)
        cv2.putText(bev, occ.label, (int(fp.x_left) - 20, y - 8), FONT, 0.38,
                    color, 1, cv2.LINE_AA)


def view_bev(fa: FrameAnalysis, settings: Settings) -> np.ndarray:
    cfg = settings.bev
    if fa.H is None:
        return no_homography(settings, cfg.width, cfg.height)
    bev = (cv2.warpPerspective(fa.image, fa.H, (cfg.width, cfg.height)) * 0.5).astype(np.uint8)
    _draw_bev_geometry(bev, fa, settings)
    state_banner(bev, fa, settings)
    return bev


def view_smear(fa: FrameAnalysis, settings: Settings) -> np.ndarray:
    """Warum im BEV nur der Boden gilt: dieselbe Warp-Operation, ungedimmt.

    Fahrzeuge haben Bauhoehe und zerlaufen radial vom Kamerapunkt weg. Die
    magentafarbene Flaeche ist die *Bildflaeche* der Bbox nach BEV -- sie
    ueberdeckt mehrere Spuren. Die gelbe Linie ist die projizierte Unterkante,
    die einzige Groesse, die tatsaechlich in der Bodenebene liegt.
    """
    cfg, debug = settings.bev, settings.debug
    if fa.H is None:
        return no_homography(settings, cfg.width, cfg.height)
    bev = cv2.warpPerspective(fa.image, fa.H, (cfg.width, cfg.height))

    for vehicle in fa.vehicles:
        x1, y1, x2, y2 = vehicle.bbox
        corners = np.float32([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]])
        warped = cv2.perspectiveTransform(corners, fa.H).reshape(-1, 2).astype(np.int32)
        overlay = bev.copy()
        cv2.fillPoly(overlay, [warped], debug.color("smear"))
        bev = cv2.addWeighted(overlay, 0.30, bev, 0.70, 0)
        cv2.polylines(bev, [warped], True, debug.color("smear"), 1)

    for L in fa.lanes_rel:
        cv2.line(bev, (int(L.x_lo), 0), (int(L.x_lo), cfg.height),
                 debug.color("boundary_lane"), 1)
    for fp in [fa.ego_footprint, *(o.footprint for o in fa.occupancies)]:
        if fp is not None:
            cv2.line(bev, (int(fp.x_left), int(fp.y)), (int(fp.x_right), int(fp.y)),
                     debug.color("footprint"), 4)

    hud(bev, [(f"frame {fa.index}", (255, 255, 255)),
              ("magenta = Bbox-Flaeche nach BEV (ungueltig)", (255, 120, 255)),
              ("gelb = projizierte Unterkante (gueltig)", debug.color("footprint"))])
    return bev


def view_hist(fa: FrameAnalysis, settings: Settings) -> np.ndarray:
    cfg, debug = settings.bev, settings.debug
    plot_height = debug.hist_plot_height
    total = cfg.height + plot_height
    if fa.mask_bev is None or fa.histogram is None:
        return no_homography(settings, cfg.width, total)

    out = np.zeros((total, cfg.width, 3), np.uint8)
    out[:cfg.height] = cv2.cvtColor(fa.mask_bev, cv2.COLOR_GRAY2BGR)
    for x in fa.boundaries:
        cv2.line(out, (round(x), 0), (round(x), cfg.height), (0, 200, 255), 1)
    for L in fa.lanes_rel:                  # was daraus als Spur uebrig blieb
        if L.synthetic:
            dashed_line(out, (int(L.x_lo), 0), (int(L.x_lo), cfg.height),
                        debug.color("boundary_synthetic"))

    hist, base = fa.histogram, total - 12
    span = max(float(hist.max()), float(cfg.peak_min_pixels) * 2, 1.0)
    scale = (plot_height - 24) / span
    cv2.polylines(out, [np.array([[x, base - h * scale] for x, h in enumerate(hist)],
                                 np.int32)], False, (255, 255, 255), 1)

    y_thr = int(base - cfg.peak_min_pixels * scale)
    cv2.line(out, (0, y_thr), (cfg.width, y_thr), (0, 120, 255), 1)
    cv2.putText(out, f"peak_min_pixels={cfg.peak_min_pixels}", (6, y_thr - 4),
                FONT, 0.36, (0, 120, 255), 1, cv2.LINE_AA)

    for i, x in enumerate(fa.boundaries):
        # Die Grenzen sind Polynomwerte, also Fliesskomma -- zum Indizieren in
        # das Histogramm runden und auf das Bild begrenzen.
        column = min(max(round(x), 0), len(hist) - 1)
        y = int(base - hist[column] * scale)
        cv2.drawMarker(out, (column, y), (0, 200, 255), cv2.MARKER_TRIANGLE_DOWN, 9, 1)
        cv2.putText(out, f"b{i}", (column - 6, y - 8), FONT, 0.34, (0, 200, 255), 1)

    cv2.line(out, (0, cfg.height), (cfg.width, cfg.height), (60, 60, 60), 1)
    hud(out, [(f"frame {fa.index}  H:{fa.h_state}", debug.color(f"homography_{fa.h_state}")),
              (f"Grenzen: {len(fa.boundary_fit)} ({fa.boundary_fit.method})"
               f" -> Korridore: {len(fa.corridors)}"
               f" -> Spuren: {len(fa.lanes_rel)}", (255, 255, 255)),
              (f"Korridorbreiten: {[round(b - a) for a, b in fa.corridors]}", (200, 200, 200)),
              (f"Spurbreite: {fa.lane_width:.0f} px  (peak_min_distance="
               f"{cfg.peak_min_distance})", (200, 200, 200))])
    return out


# --------------------------------------------------------------------------- #
# Virtuelle Schraegkameras                                                    #
# --------------------------------------------------------------------------- #
def view_camera(fa: FrameAnalysis, settings: Settings, name: str) -> np.ndarray:
    debug = settings.debug
    cam = VirtualCamera(debug.camera(name), settings.bev)
    if fa.H is None:
        return no_homography(settings, cam.width, cam.height)

    out = (cv2.warpPerspective(fa.image, cam.homography() @ fa.H,
                               (cam.width, cam.height)) * 0.55).astype(np.uint8)

    ego = ego_lane_of(fa)
    if ego is not None:
        quad = cam.project_int([[ego.x_lo, 0], [ego.x_hi, 0],
                                [ego.x_hi, settings.bev.height], [ego.x_lo, settings.bev.height]])
        fill = out.copy()
        cv2.fillPoly(fill, [quad], (0, 110, 0))
        out = cv2.addWeighted(fill, 0.35, out, 0.65, 0)

    ys = list(range(0, settings.bev.height + 1, 20))
    for L in fa.lanes_rel:                             # Spurgrenzen als Bodenlinien
        for x in (L.x_lo, L.x_hi):
            pts = cam.project_int([[x, y] for y in ys])
            if L.synthetic:
                dashed_polyline(out, pts, debug.color("boundary_synthetic"))
            else:
                cv2.polylines(out, [pts], False, debug.color("boundary_lane"), 1)

    lane_px = fa.lane_width or debug.fallback_lane_width_px
    height_px = lane_px / debug.lane_width_m * debug.vehicle_height_m

    def draw_box(fp, color, label: str) -> None:
        ground = cam.project_int([[fp.x_left, fp.y], [fp.x_right, fp.y]])
        top = cam.project_int([[fp.x_left, fp.y], [fp.x_right, fp.y]], height_px)
        cv2.line(out, tuple(int(v) for v in ground[0]), tuple(int(v) for v in ground[1]),
                 color, 3)
        cv2.polylines(out, [np.array([ground[0], ground[1], top[1], top[0]])], True, color, 1)
        cv2.putText(out, label, (int(top[0][0]), int(top[0][1]) - 6), FONT, 0.4,
                    color, 1, cv2.LINE_AA)

    if fa.ego_footprint:
        draw_box(fa.ego_footprint, ego_color(settings, fa), f"EGO {fa.ego_in_lane:.2f}")
    for occ in fa.occupancies:
        draw_box(occ.footprint, state_color(settings, occ), occ.label)

    if 0 < cam.horizon_y < cam.height:
        cv2.line(out, (0, cam.horizon_y), (cam.width, cam.horizon_y), (60, 60, 60), 1)

    state_banner(out, fa, settings)
    cfg = debug.camera(name)
    cv2.putText(out, f"virtuelle Kamera '{name}'  pitch={cfg.pitch_deg:.0f} yaw={cfg.yaw_deg:.0f}",
                (10, cam.height - 12), FONT, 0.42, debug.color("caption"), 1, cv2.LINE_AA)
    return out


# --------------------------------------------------------------------------- #
# Komposit mit Zeitverlauf                                                    #
# --------------------------------------------------------------------------- #
class Dashboard:
    """Komposit-Renderer; haelt als einziger View eine Historie.

    Der Zeitstreifen ist die Messung aus README-Schritt 1 und zugleich der
    Wirksamkeitsnachweis von `lanes.indexing`: weiss ist die Rohkorridor-Anzahl,
    cyan die Anzahl rekonstruierter Spuren. Springt Weiss, ohne dass Cyan
    springt, hat die Indizierung einen Ausfall aufgefangen.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.layout = settings.debug.dashboard
        # Ein Sample pro Streifenspalte: der Verlauf waechst nach rechts und
        # scrollt erst, wenn er die Breite erreicht hat.
        self.history: deque = deque(maxlen=self.layout.width)
        self.events: list = []

    def __call__(self, fa: FrameAnalysis) -> np.ndarray:
        settings, layout = self.settings, self.layout
        self.history.append((fa.h_state, len(fa.corridors), len(fa.lanes_rel),
                             fa.ego_lane_pos, fa.ego_in_lane, fa.events))
        self.events.extend(fa.events)

        width, height = layout.width, layout.height
        canvas = np.zeros((height, width, 3), np.uint8)

        front = view_front(fa, settings)
        event_log(front, self.events, settings, x=10, y=front.shape[0] - 100)
        canvas[0:layout.front_height] = fit(front, width, layout.front_height)

        row, bottom = layout.front_height, layout.front_height + layout.panel_height
        low, high = layout.panel_splits
        canvas[row:bottom, 0:low] = fit(view_bev(fa, settings), low, layout.panel_height)
        canvas[row:bottom, low:high] = fit(view_camera(fa, settings, "oblique"),
                                           high - low, layout.panel_height)
        canvas[row:bottom, high:width] = fit(view_hist(fa, settings),
                                             width - high, layout.panel_height)
        canvas[bottom:height] = self._timeline(width, layout.timeline_height)

        for x in (low, high):
            cv2.line(canvas, (x, row), (x, bottom), (50, 50, 50), 1)
        cv2.line(canvas, (0, row), (width, row), (50, 50, 50), 1)
        return canvas

    def _timeline(self, width: int, height: int) -> np.ndarray:
        debug, layout = self.settings.debug, self.layout
        strip = np.full((height, width, 3), 18, np.uint8)
        samples = list(self.history)
        if not samples:
            return strip

        # Drei Baender uebereinander: H-Zustand, Ego-Spurhaltung, Index-Spruenge.
        # Darunter der Zaehlplot, ganz unten die Ereignisse.
        ego_top, ego_bottom = 24, 44
        top, bottom = 64, height - 26
        span = max(6, max(max(c, l) for _, c, l, _, _, _ in samples))
        floor = layout.ego_axis_min

        def y_of(value: float) -> int:
            return int(bottom - value / span * (bottom - top))

        def y_ego(value: float) -> int:
            clamped = max(min(value, 1.0), floor)
            return int(ego_bottom - (clamped - floor) / (1 - floor) * (ego_bottom - ego_top))

        for level in range(0, span + 1, 2):
            y = y_of(level)
            cv2.line(strip, (34, y), (width - 6, y), debug.color("grid"), 1)
            cv2.putText(strip, str(level), (8, y + 4), FONT, 0.33,
                        debug.color("axis_label"), 1)
        cv2.line(strip, (34, ego_top), (width - 6, ego_top), debug.color("grid"), 1)

        for x, (h_state, corridors, rel_lanes, _, ego_in, events) in enumerate(samples):
            cv2.line(strip, (x, 6), (x, 18), debug.color(f"homography_{h_state}"), 1)
            cv2.circle(strip, (x, y_ego(ego_in)), 1,
                       debug.color("ego" if ego_in >= 1.0 else "ego_departing"), -1)
            cv2.circle(strip, (x, y_of(corridors)), 1, debug.color("boundary_lane"), -1)
            cv2.circle(strip, (x, y_of(rel_lanes)), 1, debug.color("ego"), -1)
            for event in events:
                cv2.line(strip, (x, bottom + 4), (x, bottom + 12),
                         debug.color(f"event_{event.kind}"), 1)

        # Spruenge des POSITIONSBASIERTEN Ego-Index: der Fehlermodus, den die
        # ego-relative Nummerierung aufloest. Kurze Marke am Plotrand.
        for x in range(1, len(samples)):
            before, now = samples[x - 1][3], samples[x][3]
            if before >= 0 and now >= 0 and before != now:
                cv2.line(strip, (x, top - 14), (x, top - 4), (0, 140, 255), 1)

        caption = debug.color("caption")
        cv2.putText(strip, "Ego in Spur 1.00", (width - 130, ego_bottom + 10), FONT,
                    0.32, debug.color("axis_label"), 1, cv2.LINE_AA)
        cv2.putText(strip, "H-Zustand (gruen fresh / orange held / rot none)",
                    (6, height - 10), FONT, 0.36, caption, 1, cv2.LINE_AA)
        cv2.putText(strip, "weiss Korridore   cyan Spuren (rel)   orange Index-Sprung"
                    "   unten Ereignisse", (width - 500, height - 10), FONT, 0.36,
                    caption, 1, cv2.LINE_AA)
        return strip


# --------------------------------------------------------------------------- #
# Registry                                                                    #
# --------------------------------------------------------------------------- #
STATIC_VIEWS = {
    "front": view_front,
    "mask": view_mask,
    "bev": view_bev,
    "smear": view_smear,
    "hist": view_hist,
}


def available_views(settings: Settings) -> list[str]:
    """Feste Ansichten plus jede in `config/debug.yaml` definierte Kamera."""
    return sorted({*STATIC_VIEWS, *settings.debug.cameras, "dash"})


def make_view(name: str, settings: Settings) -> Renderer:
    """Erzeugt den Renderer einer Ansicht.

    `dash` ist der einzige zustandsbehaftete View (Historie), deshalb eine
    Fabrik statt einer festen Funktionstabelle.
    """
    if name == "dash":
        return Dashboard(settings)
    if name in STATIC_VIEWS:
        return partial(STATIC_VIEWS[name], settings=settings)
    if name in settings.debug.cameras:
        return partial(view_camera, settings=settings, name=name)
    raise ValueError(f"unbekannte Ansicht {name!r}; verfuegbar: {available_views(settings)}")

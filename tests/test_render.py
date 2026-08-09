"""Rendering: virtuelle Kamera und die View-Registry.

Die Debug-Ansichten werden nicht pixelweise geprueft -- das waere ein Test des
Zeichencodes, nicht der Aussage. Geprueft wird, was zusagt ist: jede Ansicht
liefert fuer JEDEN Frame ein Bild konstanter Groesse, auch wenn Stufen
ausgefallen sind, und neue Kameras entstehen aus der Config.
"""

from __future__ import annotations

import numpy as np
import pytest

from adascope.config import BevConfig, DebugConfig, VirtualCamConfig
from adascope.lanes import SequencePipeline
from adascope.render import VirtualCamera, available_views, make_view


@pytest.fixture
def failed_frame(settings):
    """Ein Frame ohne jede Geometrie -- der haerteste Fall fuer die Renderer."""
    return SequencePipeline(settings).process(
        0, "leer", np.zeros((457, 1209, 3), np.uint8), [])


@pytest.fixture(params=["histogram", "windows"])
def good_frame(request, settings):
    """Ein Frame MIT vollstaendiger Geometrie, in beiden Suchverfahren.

    Der leere Frame allein genuegt nicht: er laesst jeden Zeichenpfad aus, der
    Grenzen, Korridore, Spuren oder Footprints anfasst. Genau dort sass ein
    Fehler, der so durch die Suite kam -- die Grenzen wurden zu Fliesskomma und
    ein Histogramm-Zugriff `hist[x]` brach beim ersten echten Frame.
    """
    from adascope.config import IndexConfig, WindowConfig
    from adascope.synthetic import SyntheticRoad

    road = SyntheticRoad(lane=settings.lane, bev=settings.bev, lanes=3)
    pipeline = SequencePipeline(settings,
                                indexing=IndexConfig(lane_width=road.lane_width),
                                windows=WindowConfig(method=request.param))
    analysis = pipeline.process(0, "syn", road.frame(),
                                [road.ego(), road.vehicle(1, road.lane_center(0), 450)])
    assert analysis.lanes_rel, "Fixture kaputt: die Szene liefert keine Spuren"
    return analysis


# --------------------------------------------------------------------------- #
# Virtuelle Kamera                                                            #
# --------------------------------------------------------------------------- #
def test_ground_plane_maps_to_a_homography():
    """Weil beide Ansichten Bilder DERSELBEN Ebene sind, muss die direkte
    Projektion mit der Homographie uebereinstimmen -- sonst waere die Ansicht
    eine eigene Messung statt einer Umrechnung."""
    import cv2

    bev = BevConfig()
    cam = VirtualCamera(VirtualCamConfig(), bev)
    points = np.float32([[100, 200], [300, 650], [420, 50]])
    direct = cam.project(points)
    via_h = cv2.perspectiveTransform(points.reshape(-1, 1, 2), cam.homography()).reshape(-1, 2)
    assert np.allclose(direct, via_h, atol=0.5)


def test_elevation_lifts_points_upward_in_the_image():
    cam = VirtualCamera(VirtualCamConfig(), BevConfig())
    ground = cam.project([[250, 600]])[0]
    roof = cam.project([[250, 600]], elevation=40)[0]
    assert roof[1] < ground[1]          # kleineres y = weiter oben


def test_points_behind_the_camera_do_not_blow_up():
    cam = VirtualCamera(VirtualCamConfig(behind=1.0), BevConfig())
    assert np.all(np.isfinite(cam.project([[250, -100000]])))


def test_pitch_moves_the_horizon():
    bev = BevConfig()
    flach = VirtualCamera(VirtualCamConfig(pitch_deg=5.0), bev).horizon_y
    steil = VirtualCamera(VirtualCamConfig(pitch_deg=40.0), bev).horizon_y
    assert steil < flach


# --------------------------------------------------------------------------- #
# View-Registry                                                               #
# --------------------------------------------------------------------------- #
def test_available_views_include_every_configured_camera(settings):
    assert {"front", "mask", "bev", "hist", "smear", "dash"} <= set(available_views(settings))
    assert {"oblique", "shoulder"} <= set(available_views(settings))


def test_a_camera_added_in_config_becomes_a_view(settings):
    """Der Kern der Config-Idee: eine neue Perspektive ohne Codeaenderung."""
    from dataclasses import replace

    debug = DebugConfig.from_dict({"cameras": {"heli": {"pitch_deg": 60.0,
                                                        "elevation": 800.0}}})
    erweitert = replace(settings, debug=debug)
    assert "heli" in available_views(erweitert)
    assert make_view("heli", erweitert) is not None


def test_unknown_view_lists_the_alternatives(settings):
    with pytest.raises(ValueError, match="verfuegbar"):
        make_view("gibt-es-nicht", settings)


@pytest.mark.parametrize("name", ["front", "mask", "bev", "hist", "smear",
                                  "oblique", "shoulder", "dash"])
def test_every_view_renders_a_complete_frame(name, settings, good_frame):
    """Jede Ansicht mit vollstaendiger Geometrie -- in beiden Suchverfahren.

    Deckt die Zeichenpfade ab, die der leere Frame nie erreicht: Grenzen,
    Korridore, Spurbeschriftung, Footprints, Histogramm-Marker.
    """
    image = make_view(name, settings)(good_frame)
    assert image.ndim == 3 and image.shape[2] == 3
    assert image.any(), "Bild ist komplett schwarz -- nichts gezeichnet"


@pytest.mark.parametrize("name", ["front", "mask", "bev", "hist", "smear",
                                  "oblique", "shoulder", "dash"])
def test_every_view_renders_a_failed_frame(name, settings, failed_frame):
    """Ein ausgefallener Frame darf keinen Renderer umbringen und keinen Frame
    ueberspringen -- sonst laeuft die Zeitachse des Videos falsch."""
    image = make_view(name, settings)(failed_frame)
    assert image.ndim == 3 and image.shape[2] == 3
    assert image.shape[0] > 0 and image.shape[1] > 0


def test_dashboard_size_matches_the_configured_layout(settings, failed_frame):
    image = make_view("dash", settings)(failed_frame)
    layout = settings.debug.dashboard
    assert image.shape[:2] == (layout.height, layout.width)


def test_dashboard_keeps_a_constant_size_across_frames(settings, failed_frame):
    dash = make_view("dash", settings)
    sizes = {dash(failed_frame).shape for _ in range(3)}
    assert len(sizes) == 1

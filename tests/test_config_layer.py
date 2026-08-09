"""Die Config-Schicht: Defaults, Validierung, Merge, Szenario-Overlay.

Der Vertrag, den diese Tests festschreiben:

1. Jede Domaenen-Config ist OHNE Datei konstruierbar.
2. Eine YAML ueberschreibt nur, was sie nennt.
3. Ungueltige Werte scheitern beim LADEN, nicht irgendwo in der Pipeline.
4. Ein Tippfehler im Schluessel ist ein Fehler, kein stilles Ignorieren.
"""

from __future__ import annotations

import pytest

from adascope.config import (
    BevConfig, DebugConfig, EventConfig, IndexConfig, LaneConfig, PipelineConfig,
    Settings, TrackingConfig,
)
from adascope.config.loader import deep_merge, load_section, read_yaml

ALL_CONFIGS = [BevConfig, DebugConfig, EventConfig, IndexConfig, LaneConfig,
               PipelineConfig, TrackingConfig]


# --------------------------------------------------------------------------- #
# 1. Defaults im Code                                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cls", ALL_CONFIGS)
def test_constructs_without_any_file(cls):
    assert cls() is not None


@pytest.mark.parametrize("cls", ALL_CONFIGS)
def test_empty_mapping_equals_defaults(cls):
    assert cls.from_dict({}) == cls()


# --------------------------------------------------------------------------- #
# 2. Nur Genanntes wird ueberschrieben                                        #
# --------------------------------------------------------------------------- #
def test_partial_override_keeps_other_defaults():
    cfg = BevConfig.from_dict({"peak_min_distance": 40})
    assert cfg.peak_min_distance == 40
    assert cfg.peak_min_pixels == BevConfig().peak_min_pixels
    assert cfg.width == BevConfig().width


def test_deep_merge_replaces_lists_rather_than_appending():
    # Verkettung waere bequem, macht das Entfernen eines geerbten Eintrags aber
    # unmoeglich. Ersetzen ist die vorhersagbare Regel.
    merged = deep_merge({"a": [1, 2], "b": {"c": 1, "d": 2}}, {"a": [9], "b": {"c": 5}})
    assert merged == {"a": [9], "b": {"c": 5, "d": 2}}


def test_missing_file_yields_empty_mapping(tmp_path):
    assert read_yaml(tmp_path / "gibt-es-nicht.yaml") == {}


# --------------------------------------------------------------------------- #
# 3. Validierung beim Laden                                                   #
# --------------------------------------------------------------------------- #
def test_bev_rejects_inverted_x_range():
    with pytest.raises(ValueError, match="x_left"):
        BevConfig.from_dict({"x_left": 400, "x_right": 100})


def test_bev_rejects_encroaching_above_in_lane():
    with pytest.raises(ValueError, match="thr_encroaching"):
        BevConfig.from_dict({"thr_encroaching": 0.9, "thr_in_lane": 0.5})


def test_lane_rejects_top_below_bottom():
    with pytest.raises(ValueError, match="y_top"):
        LaneConfig.from_dict({"y_top": 400, "y_bottom": 100})


def test_lane_rejects_polygon_with_two_points():
    with pytest.raises(ValueError, match="drei Punkte"):
        LaneConfig.from_dict({"roi_polygon": [[0, 0], [1, 1]]})


def test_tracking_rejects_unknown_tracker():
    with pytest.raises(ValueError, match="Tracker"):
        TrackingConfig.from_dict({"tracker": "deepsort.yaml"})


def test_tracking_rejects_inverted_ego_zone():
    with pytest.raises(ValueError, match="ego_zone"):
        TrackingConfig.from_dict({"ego_zone": [0.8, 0.3, 0.4, 0.7]})


def test_events_rejects_negative_confirm_frames():
    with pytest.raises(ValueError, match="confirm_frames"):
        EventConfig.from_dict({"confirm_frames": -1})


def test_dashboard_rejects_odd_edges():
    # mp4v schneidet bei ungerader Kante stumm eine Zeile ab -- das muss beim
    # Laden auffallen, nicht erst an einem um ein Pixel verschobenen Video.
    with pytest.raises(ValueError, match="gerade Kanten"):
        DebugConfig.from_dict({"dashboard": {"width": 1221}})


def test_camera_rejects_pitch_beyond_vertical():
    with pytest.raises(ValueError, match="pitch_deg"):
        DebugConfig.from_dict({"cameras": {"oblique": {"pitch_deg": 95.0}}})


# --------------------------------------------------------------------------- #
# 4. Tippfehler sind Fehler                                                   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cls,key", [
    (BevConfig, "peak_min_distanc"),
    (LaneConfig, "hough_threshhold"),
    (EventConfig, "confirmed_frames"),
    (PipelineConfig, "max_holds"),
])
def test_unknown_key_is_rejected(cls, key):
    with pytest.raises(ValueError, match="unbekannte Schluessel"):
        cls.from_dict({key: 1})


# --------------------------------------------------------------------------- #
# Besonderheiten einzelner Domaenen                                           #
# --------------------------------------------------------------------------- #
def test_indexing_zero_lane_width_means_estimate():
    # 0 ist in YAML bequemer zu schreiben als null und bedeutet dasselbe.
    assert IndexConfig.from_dict({"lane_width": 0}).lane_width is None
    assert IndexConfig.from_dict({"lane_width": 77}).lane_width == 77


def test_debug_config_merges_camera_partially():
    cfg = DebugConfig.from_dict({"cameras": {"oblique": {"pitch_deg": 30.0}}})
    assert cfg.camera("oblique").pitch_deg == 30.0
    assert cfg.camera("oblique").focal == DebugConfig().camera("oblique").focal


def test_debug_config_accepts_new_camera():
    # Eine neue Perspektive soll ohne Codeaenderung entstehen.
    cfg = DebugConfig.from_dict({"cameras": {"heli": {"pitch_deg": 60.0, "elevation": 900.0}}})
    assert "heli" in cfg.cameras
    assert cfg.camera("heli").pitch_deg == 60.0


def test_debug_config_unknown_camera_names_alternatives():
    with pytest.raises(ValueError, match="verfuegbar"):
        DebugConfig().camera("gibt-es-nicht")


# --------------------------------------------------------------------------- #
# Settings und Szenario-Overlay                                               #
# --------------------------------------------------------------------------- #
def _write(dir_, name: str, text: str):
    dir_.mkdir(parents=True, exist_ok=True)
    (dir_ / name).write_text(text, encoding="utf-8")


def test_settings_load_from_empty_dir_uses_defaults(tmp_path):
    settings = Settings.load(tmp_path)
    assert settings.bev == BevConfig()
    assert settings.detection is None       # detection.yaml ist optional


def test_require_detection_explains_the_missing_file(tmp_path):
    with pytest.raises(ValueError, match="detection.yaml"):
        Settings.load(tmp_path).require_detection()


def test_scenario_overrides_only_its_sections(tmp_path):
    _write(tmp_path, "bev.yaml", "peak_min_distance: 55\npeak_min_pixels: 10\n")
    _write(tmp_path, "indexing.yaml", "lane_width: 0\n")
    _write(tmp_path / "scenarios", "dreispurig.yaml",
           "indexing:\n  lane_width: 77\n")

    base = Settings.load(tmp_path)
    scenario = Settings.load(tmp_path, scenario="dreispurig")

    assert base.indexing.lane_width is None
    assert scenario.indexing.lane_width == 77
    # Nicht genannte Domaenen bleiben unangetastet.
    assert scenario.bev.peak_min_distance == base.bev.peak_min_distance == 55


def test_scenario_that_does_not_exist_is_a_no_op(tmp_path):
    # Ein fehlendes Overlay ist kein Fehler: die Basiskalibrierung gilt weiter.
    _write(tmp_path, "bev.yaml", "peak_min_distance: 60\n")
    assert Settings.load(tmp_path, scenario="unbekannt").bev.peak_min_distance == 60


def test_scenario_section_must_be_a_mapping(tmp_path):
    _write(tmp_path / "scenarios", "kaputt.yaml", "indexing: 77\n")
    with pytest.raises(ValueError, match="Mapping"):
        Settings.load(tmp_path, scenario="kaputt")


def test_load_section_rejects_non_mapping_root(tmp_path):
    _write(tmp_path, "bev.yaml", "- 1\n- 2\n")
    with pytest.raises(ValueError, match="Mapping"):
        load_section(BevConfig, tmp_path, "bev.yaml")


def test_weights_path_resolves_against_project_root(tmp_path):
    settings = Settings.load(tmp_path, root=tmp_path / "projekt")
    assert settings.weights_path().endswith("yolo11n.pt")
    assert "projekt" in settings.weights_path()


# --------------------------------------------------------------------------- #
# Aufloesungsanpassung der Spurkalibrierung                                   #
# --------------------------------------------------------------------------- #
def test_lane_config_is_unchanged_at_its_reference_size():
    cfg = LaneConfig()
    assert cfg.scaled_to(*cfg.reference_size) is cfg


def test_lane_config_scales_pixel_lengths_proportionally():
    """Derselbe Ausschnitt, nur groesser: 1428x534 gegen 1209x457."""
    cfg = LaneConfig().scaled_to(1428, 534)
    assert cfg.reference_size == (1428, 534)
    assert cfg.y_bottom == round(295 * 534 / 457)
    assert cfg.roi_polygon[0] == (round(15 * 1428 / 1209), round(300 * 534 / 457))
    assert cfg.ego_x_bottom == pytest.approx(600 * 1428 / 1209)


def test_lane_config_leaves_resolution_independent_values_alone():
    cfg = LaneConfig().scaled_to(1428, 534)
    assert cfg.min_line_angle_deg == LaneConfig().min_line_angle_deg
    assert cfg.white_l_min == LaneConfig().white_l_min
    assert cfg.min_cluster_support == LaneConfig().min_cluster_support


def test_lane_config_refuses_a_different_aspect_ratio():
    """1920x1080 ist kein groesseres 1209x457, sondern ein anderer Ausschnitt.

    Stillschweigend zu skalieren wuerde die ROI ueber falschen Bildinhalt legen
    und in einem Video voller leerer Frames enden, ohne dass jemand erfaehrt,
    warum. Die Meldung nennt stattdessen den Ausweg.
    """
    with pytest.raises(ValueError, match="anderen Ausschnitt"):
        LaneConfig().scaled_to(1920, 1080)


def test_lane_config_matches_reports_compatibility_without_raising():
    assert LaneConfig().matches(1428, 534)
    assert not LaneConfig().matches(1920, 1080)

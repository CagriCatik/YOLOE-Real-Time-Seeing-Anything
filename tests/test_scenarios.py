"""Szenario-Erkennung, Aufloesung und Zusammenfassung.

Der Vertrag: was in `scenarien/` liegt, wird gefunden; der Dateistamm ist der
Name und verbindet Aufnahme, Kalibrier-Ueberlagerung und Ergebnisordner.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from adascope.runner import summarise
from adascope.ground_truth import GroundTruth
from adascope.perception_ground_truth import ExpectedPerceptionFrame
from adascope.scenarios import RunSummary, discover, render_table, resolve


def video(path, frames: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (32, 32))
    for _ in range(frames):
        writer.write(np.zeros((32, 32, 3), np.uint8))
    writer.release()


def frame_dir(path, count: int = 2) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        cv2.imwrite(str(path / f"f{index:03d}.png"), np.zeros((16, 16, 3), np.uint8))


# --------------------------------------------------------------------------- #
# Erkennung                                                                   #
# --------------------------------------------------------------------------- #
def test_finds_videos_and_frame_directories(tmp_path):
    video(tmp_path / "scenarien" / "aus_video.mp4")
    frame_dir(tmp_path / "scenarien" / "aus_frames")
    found = discover(tmp_path / "scenarien", tmp_path / "config")
    assert [s.name for s in found] == ["aus_frames", "aus_video"]
    assert {s.name: s.is_video for s in found} == {"aus_frames": False, "aus_video": True}


def test_ignores_files_that_are_neither(tmp_path):
    (tmp_path / "scenarien").mkdir()
    (tmp_path / "scenarien" / "notizen.txt").write_text("nichts", encoding="utf-8")
    assert discover(tmp_path / "scenarien", tmp_path / "config") == []


def test_ignores_an_empty_directory(tmp_path):
    """Ein leerer Ordner waere ein Lauf, der sofort mit einem Fehler endet."""
    (tmp_path / "scenarien" / "leer").mkdir(parents=True)
    assert discover(tmp_path / "scenarien", tmp_path / "config") == []


def test_a_video_wins_against_a_directory_of_the_same_name(tmp_path):
    # Beide enthalten dieselben Frames; das Video ist die kompaktere Quelle.
    frame_dir(tmp_path / "scenarien" / "doppelt")
    video(tmp_path / "scenarien" / "doppelt.mp4")
    found = discover(tmp_path / "scenarien", tmp_path / "config")
    assert len(found) == 1 and found[0].is_video


def test_missing_scenario_directory_is_not_an_error(tmp_path):
    assert discover(tmp_path / "gibt-es-nicht", tmp_path / "config") == []


# --------------------------------------------------------------------------- #
# Verbindung zur Kalibrierung                                                 #
# --------------------------------------------------------------------------- #
def test_matching_overlay_is_picked_up_by_name(tmp_path):
    video(tmp_path / "scenarien" / "dreispurig.mp4")
    overlays = tmp_path / "config" / "scenarios"
    overlays.mkdir(parents=True)
    (overlays / "dreispurig.yaml").write_text("indexing:\n  lane_width: 77\n", encoding="utf-8")

    scenario = discover(tmp_path / "scenarien", tmp_path / "config")[0]
    assert scenario.config_overlay is not None
    assert scenario.overlay_name == "dreispurig"
    assert scenario.settings(tmp_path / "config").indexing.lane_width == 77


def test_without_an_overlay_the_base_calibration_applies(tmp_path):
    video(tmp_path / "scenarien" / "ohne.mp4")
    scenario = discover(tmp_path / "scenarien", tmp_path / "config")[0]
    assert scenario.config_overlay is None
    assert scenario.overlay_name is None


def test_result_dir_is_named_after_the_scenario(tmp_path):
    video(tmp_path / "scenarien" / "abc.mp4")
    scenario = discover(tmp_path / "scenarien", tmp_path / "config")[0]
    assert scenario.result_dir(tmp_path / "results").name == "abc"


# --------------------------------------------------------------------------- #
# Auswahl                                                                     #
# --------------------------------------------------------------------------- #
def test_no_names_means_every_scenario(tmp_path):
    video(tmp_path / "scenarien" / "a.mp4")
    video(tmp_path / "scenarien" / "b.mp4")
    assert len(resolve([], tmp_path / "scenarien", tmp_path / "config")) == 2


def test_names_are_resolved_in_the_given_order(tmp_path):
    video(tmp_path / "scenarien" / "a.mp4")
    video(tmp_path / "scenarien" / "b.mp4")
    chosen = resolve(["b", "a"], tmp_path / "scenarien", tmp_path / "config")
    assert [s.name for s in chosen] == ["b", "a"]


def test_unknown_name_lists_what_exists(tmp_path):
    video(tmp_path / "scenarien" / "a.mp4")
    with pytest.raises(ValueError, match="Verfuegbar"):
        resolve(["tippfehler"], tmp_path / "scenarien", tmp_path / "config")


# --------------------------------------------------------------------------- #
# Zusammenfassung                                                             #
# --------------------------------------------------------------------------- #
def rows(*specs) -> list[dict]:
    """(h_state, n_corridors, ego_lane_pos, n_lanes_rel, ego_in_lane) je Frame."""
    return [{"h_state": h, "n_corridors": c, "ego_lane_pos": e, "n_lanes_rel": r,
             "lane_width": 77.0, "ego_in_lane": ego}
            for h, c, e, r, ego in specs]


def test_summarise_counts_homography_states():
    summary = summarise("x", rows(("fresh", 4, 2, 3, 1.0), ("held", 4, 2, 3, 1.0),
                                  ("held", 0, -1, 0, 1.0)), [])
    assert summary.homography == {"fresh": 1, "held": 2}
    assert summary.fresh_pct == pytest.approx(100 / 3)


def test_summarise_counts_only_transitions_between_valid_frames():
    """Ein Stufenausfall dazwischen ist kein Sprung des Index.

    Sonst zaehlte jede Luecke doppelt: einmal als Ausfall, einmal als Sprung.
    """
    summary = summarise("x", rows(("fresh", 4, 2, 3, 1.0),
                                  ("none", 0, -1, 0, 1.0),     # faellt heraus
                                  ("fresh", 4, 2, 3, 1.0)), [])
    assert summary.index_transitions == 1
    assert summary.index_jumps == 0


def test_summarise_reports_a_real_index_jump():
    summary = summarise("x", rows(("fresh", 4, 2, 3, 1.0), ("fresh", 3, 1, 3, 1.0)), [])
    assert summary.index_jumps == 1 and summary.jump_pct == 100.0


def test_summarise_tracks_the_ego_departure_signal():
    summary = summarise("x", rows(("fresh", 4, 2, 3, 1.0), ("fresh", 4, 2, 3, 0.88),
                                  ("fresh", 4, 2, 3, 0.95)), [])
    assert summary.ego_in_lane_min == 0.88
    assert summary.ego_departing_frames == 2


def test_summarise_marks_an_empty_run_as_an_error():
    assert summarise("x", [], []).error


def test_table_renders_a_failed_run_without_dropping_it():
    """Ein gescheitertes Szenario muss in der Tabelle stehen, nicht fehlen."""
    table = render_table([RunSummary(scenario="kaputt", error="Aufloesung passt nicht")])
    assert "kaputt" in table and "FEHLER" in table


def test_table_is_empty_but_valid_without_runs():
    assert "Keine Szenarien" in render_table([])


def test_core_runner_writes_perception_score_csv(tmp_path, settings):
    from adascope.runner import run_debug

    source = tmp_path / "frames"
    source.mkdir()
    cv2.imwrite(str(source / "f000.png"),
                np.zeros((457, 1209, 3), np.uint8))
    truth = GroundTruth((), perception=(
        ExpectedPerceptionFrame(frame=0, lane_count=1),))
    out = tmp_path / "result"

    summary = run_debug(source, settings, [], out, truth=truth)

    assert summary.perception_score is not None
    assert not summary.perception_score.perfect
    assert (out / "debug_perception.csv").exists()
    assert "debug_perception.csv" in (out / "summary.txt").read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Zuschnitt je Quelle                                                         #
# --------------------------------------------------------------------------- #
def test_already_cropped_source_is_left_alone(settings):
    """Dieselbe Sammlung enthaelt beides -- ein globaler Schalter waere falsch."""
    from adascope.runner import choose_crop
    assert choose_crop(settings, *settings.lane.reference_size) is None


def test_full_frame_source_is_cropped_to_the_configured_box(settings, config):
    from dataclasses import replace

    from adascope.runner import choose_crop
    with_detection = replace(settings, detection=config)
    # Das Testmodell-crop_box ist [0.1, 0.1, 0.9, 0.9] -> quadratisch, passt nicht.
    assert choose_crop(with_detection, 1920, 1080) is None


def test_crop_is_chosen_when_it_makes_the_source_fit(settings, raw_config):
    from dataclasses import replace

    from adascope.config import DetectionConfig
    from adascope.runner import choose_crop
    # Ein Zuschnitt, der 1920x1080 auf das Referenz-Seitenverhaeltnis bringt.
    raw_config["crop_box"] = [0.1271, 0.1759, 0.8688, 0.6704]
    with_detection = replace(settings, detection=DetectionConfig.from_dict(raw_config))
    assert choose_crop(with_detection, 1920, 1080) == (0.1271, 0.1759, 0.8688, 0.6704)


def test_without_a_detection_config_nothing_is_cropped(settings):
    from adascope.runner import choose_crop
    assert settings.detection is None
    assert choose_crop(settings, 1920, 1080) is None

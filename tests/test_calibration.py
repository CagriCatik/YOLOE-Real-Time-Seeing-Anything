"""Auto-Kalibrierung: Sicherheitsgates, Berichte und kommentarschonendes Apply."""

from pathlib import Path

import yaml

from adascope.calibration import Calibration, Measurement, apply_calibration


def test_auto_fragments_exclude_manual_and_unstable_measurements():
    result = Calibration("sample", frames=30, measurements=[
        Measurement("white_l_min", 142, 60, 2, 130),
        Measurement("y_top", 48, 30, 3, 55, auto_apply=False),
        Measurement("peak_min_distance", 80, 30, 20, 55, stable=False),
    ])
    assert result.as_yaml_fragments(auto_only=True) == {
        "lane": {"white_l_min": 142}
    }


def test_apply_creates_backup_preserves_comments_and_reloads(tmp_path: Path):
    (tmp_path / "lane.yaml").write_text(
        "# wichtige Erklaerung\nwhite_l_min: 130  # kalibriert\n", encoding="utf-8")
    result = Calibration("sample", frames=30, measurements=[
        Measurement("white_l_min", 142, 60, 2, 130),
    ])

    backups = apply_calibration(result, tmp_path)

    text = (tmp_path / "lane.yaml").read_text(encoding="utf-8")
    assert "# wichtige Erklaerung" in text
    assert "white_l_min: 142 # kalibriert" in text
    assert len(backups) == 1 and backups[0].exists()
    assert "white_l_min: 130" in backups[0].read_text(encoding="utf-8")


def test_yaml_report_contains_proposals_and_apply_subset(tmp_path: Path):
    result = Calibration("sample", frames=30, measurements=[
        Measurement("white_l_min", 140, 60, 1, 130),
        Measurement("y_bottom", 296, 30, 2, 295, auto_apply=False),
    ])
    report = result.write_report(tmp_path / "report.yaml")
    data = yaml.safe_load(report.read_text(encoding="utf-8"))
    assert data["proposed"]["lane"] == {"white_l_min": 140, "y_bottom": 296}
    assert data["auto_apply"]["lane"] == {"white_l_min": 140}

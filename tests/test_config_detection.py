"""DetectionConfig parsing, validation, and comment-preserving writers."""

from __future__ import annotations

import pytest

from adascope.config import DetectionConfig, save_crop_box, save_rois


def test_from_dict_parses_all_sections(config):
    assert config.model.checkpoint == "yoloe-11l-seg.pt"
    assert config.model.classes == ["car", "truck"]
    assert config.model.conf == pytest.approx(0.1)
    assert set(config.rois) == {"left", "ego", "right"}
    assert config.crop_box == (0.1, 0.1, 0.9, 0.9)
    assert config.carpet.detect_in == ["left", "ego", "right"]
    assert len(config.carpet.red_hsv) == 2
    assert config.carpet.white_hsv is not None


def test_ego_box_parsed(config):
    assert config.ego_box == (0.40, 0.7, 0.60, 1.0)


def test_ego_box_optional(raw_config):
    del raw_config["ego_box"]
    assert DetectionConfig.from_dict(raw_config).ego_box is None


def test_invalid_ego_box_rejected(raw_config):
    raw_config["ego_box"] = [0.6, 0.7, 0.4, 1.0]  # x0 >= x1
    with pytest.raises(ValueError):
        DetectionConfig.from_dict(raw_config)


def test_invalid_crop_box_rejected(raw_config):
    raw_config["crop_box"] = [0.9, 0.1, 0.1, 0.9]  # x0 >= x1
    with pytest.raises(ValueError):
        DetectionConfig.from_dict(raw_config)


def test_roi_with_too_few_points_rejected(raw_config):
    raw_config["rois"]["left"] = [[0.0, 0.0], [0.3, 0.0]]
    with pytest.raises(ValueError):
        DetectionConfig.from_dict(raw_config)


def test_missing_section_raises(raw_config):
    del raw_config["model"]
    with pytest.raises(ValueError):
        DetectionConfig.from_dict(raw_config)


def test_save_crop_box_preserves_comments(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("# header comment\ncrop_box: [0.1, 0.1, 0.9, 0.9]\nmodel: x\n", encoding="utf-8")
    save_crop_box(cfg, [0.2, 0.2, 0.8, 0.8])
    text = cfg.read_text(encoding="utf-8")
    assert "# header comment" in text
    assert "crop_box: [0.2, 0.2, 0.8, 0.8]" in text
    assert "model: x" in text


def test_save_rois_rewrites_only_block(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "rois:\n  left: [[0.0, 0.0]]\n  ego: [[0.1, 0.1]]\n# keep me\ncrop_box: [0,0,1,1]\n",
        encoding="utf-8",
    )
    save_rois(cfg, {"left": [[0.5, 0.5], [0.6, 0.6]]})
    text = cfg.read_text(encoding="utf-8")
    assert "# keep me" in text
    assert "crop_box: [0,0,1,1]" in text
    assert "[0.5, 0.5]" in text

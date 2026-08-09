from pathlib import Path
from threading import Event
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from adascope.tool.core import (create_mask_outputs, crop_frames, extract_frames,
                            find_ffmpeg, frames_to_video, load_mask_config,
                            load_regions, probe_video, save_mask_config,
                            save_regions)


def test_region_roundtrip(tmp_path):
    target = tmp_path / "regions.json"
    rois = {"left": [[0.1, 0.2], [0.3, 0.4], [0.2, 0.6]]}
    save_regions(target, (100, 80), [0.1, 0.2, 0.9, 0.8], rois)
    loaded = load_regions(target)
    assert loaded["coordinate_system"] == "normalized"
    assert loaded["crop_box"] == [0.1, 0.2, 0.9, 0.8]
    assert loaded["rois"] == rois


def test_explicit_ffmpeg_resolution(tmp_path):
    executable = tmp_path / "ffmpeg.exe"
    executable.write_bytes(b"test")
    assert find_ffmpeg(executable) == executable.resolve()


def test_batch_crop(tmp_path):
    source, output = tmp_path / "in", tmp_path / "out"
    source.mkdir()
    image = np.zeros((100, 200, 3), np.uint8)
    cv2.imwrite(str(source / "frame_000000.jpg"), image)
    config = tmp_path / "regions.json"
    save_regions(config, (200, 100), [.25, .2, .75, .8], {})
    assert crop_frames(source, output, config) == 1
    cropped = cv2.imread(str(output / "frame_000000.jpg"))
    assert cropped.shape[:2] == (60, 100)


def test_extract_frames(tmp_path):
    video = tmp_path / "sample.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 24))
    for value in range(6):
        writer.write(np.full((24, 32, 3), value * 20, np.uint8))
    writer.release()
    output = tmp_path / "frames"
    assert extract_frames(video, output, every=2, extension="png", cancel=Event()) == 3
    assert len(list(output.glob("*.png"))) == 3


def test_frames_to_video(tmp_path):
    source = tmp_path / "frames"
    source.mkdir()
    for index in range(4):
        image = np.full((30, 40, 3), index * 40, np.uint8)
        cv2.imwrite(str(source / f"frame_{index:06d}.png"), image)
    output = tmp_path / "result.mp4"
    assert frames_to_video(source, output, fps=12) == output
    info = probe_video(output)
    assert info.width == 40
    assert info.height == 30
    assert info.frames == 4
    assert info.fps == 12


def test_mask_config_and_debug_outputs(tmp_path):
    source = tmp_path / "source.png"
    image = np.full((100, 100, 3), 200, np.uint8)
    cv2.imwrite(str(source), image)
    config = tmp_path / "masks.json"
    masks = {"dashboard": [[.25, .25], [.75, .25], [.75, .75], [.25, .75]]}
    save_mask_config(config, (100, 100), masks)
    assert load_mask_config(config)["masks"] == masks

    masked_path, debug_path = create_mask_outputs(
        source, config, tmp_path / "masked.png", tmp_path / "debug.png"
    )
    masked = cv2.imread(str(masked_path))
    debug = cv2.imread(str(debug_path))
    assert masked[50, 50].tolist() == [0, 0, 0]
    assert masked[10, 10].tolist() == [200, 200, 200]
    assert debug[50, 50, 2] > debug[50, 50, 0]

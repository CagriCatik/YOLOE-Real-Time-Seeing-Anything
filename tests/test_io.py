"""Frame-, Video- und Tabellen-I/O.

Diese Schicht war vor der Zusammenfuehrung dreimal implementiert, jedes Mal mit
anderen erlaubten Endungen und anderem Verhalten bei kaputten Dateien. Die
Tests halten das eine Verhalten fest.
"""

from __future__ import annotations

import csv

import cv2
import numpy as np
import pytest

from adascope.io import (
    VideoWriter, image_frames, is_video, iter_source, list_frames, read_image,
    wanted_exts, write_named, write_rows,
)


@pytest.fixture
def frame_dir(tmp_path):
    for index in range(4):
        cv2.imwrite(str(tmp_path / f"frame_{index:03d}.png"),
                    np.full((20, 30, 3), index * 10, np.uint8))
    (tmp_path / "notizen.txt").write_text("kein Bild", encoding="utf-8")
    return tmp_path


# --------------------------------------------------------------------------- #
# Bilder auflisten                                                            #
# --------------------------------------------------------------------------- #
def test_lists_only_image_files_sorted(frame_dir):
    paths = list_frames(frame_dir)
    assert [p.name for p in paths] == [f"frame_{i:03d}.png" for i in range(4)]


def test_every_takes_each_nth_frame(frame_dir):
    assert len(list_frames(frame_dir, every=2)) == 2


def test_every_below_one_is_an_error(frame_dir):
    with pytest.raises(ValueError, match="every"):
        list_frames(frame_dir, every=0)


def test_extension_filter_narrows_to_one_type(frame_dir):
    assert list_frames(frame_dir, ext="jpg") == []
    assert wanted_exts("JPG") == {".jpg"}


# --------------------------------------------------------------------------- #
# Lesen                                                                       #
# --------------------------------------------------------------------------- #
def test_unreadable_file_reads_as_none(tmp_path):
    broken = tmp_path / "kaputt.png"
    broken.write_bytes(b"kein PNG")
    assert read_image(broken) is None


def test_image_frames_skips_unreadable_files_instead_of_yielding_none(tmp_path):
    """Ein None mitten im Strom wuerde jeden Aufrufer zwingen, es abzufangen."""
    cv2.imwrite(str(tmp_path / "a.png"), np.zeros((5, 5, 3), np.uint8))
    (tmp_path / "b.png").write_bytes(b"kein PNG")
    names = [name for name, _ in image_frames(sorted(tmp_path.glob("*.png")))]
    assert names == ["a.png"]


# --------------------------------------------------------------------------- #
# Quelle erkennen                                                             #
# --------------------------------------------------------------------------- #
def test_recognises_videos_by_suffix():
    assert is_video("a/b.mp4") and is_video("A.MOV")
    assert not is_video("frames/") and not is_video("bild.png")


def test_iter_source_accepts_a_frame_directory(frame_dir):
    frames, fps = iter_source(frame_dir, every=1, fps=30.0)
    assert fps == 30.0
    assert len(list(frames)) == 4


def test_iter_source_rejects_a_path_that_is_neither(tmp_path):
    with pytest.raises(ValueError, match="Weder Video noch Ordner"):
        iter_source(tmp_path / "nichts.png")


def test_iter_source_rejects_an_empty_directory(tmp_path):
    with pytest.raises(ValueError, match="Keine Bilder"):
        iter_source(tmp_path)


# --------------------------------------------------------------------------- #
# Video schreiben                                                             #
# --------------------------------------------------------------------------- #
def test_writer_crops_odd_dimensions_to_even(tmp_path):
    """mp4v schneidet bei ungerader Kante stumm ab -- lieber explizit hier."""
    with VideoWriter(tmp_path / "out.mp4", 25.0) as writer:
        writer.write(np.zeros((457, 1209, 3), np.uint8))
        assert writer.size == (1208, 456)


def test_writer_accepts_frames_of_differing_size(tmp_path):
    # Die Groesse legt der erste Frame fest; spaetere werden angepasst, damit
    # ein einzelner abweichender Frame nicht die ganze Ausgabe kippt.
    with VideoWriter(tmp_path / "out.mp4", 25.0) as writer:
        writer.write(np.zeros((100, 100, 3), np.uint8))
        writer.write(np.zeros((80, 90, 3), np.uint8))
    assert (tmp_path / "out.mp4").exists()


def test_writer_creates_missing_parent_directories(tmp_path):
    with VideoWriter(tmp_path / "tief" / "drin" / "out.mp4", 25.0) as writer:
        writer.write(np.zeros((10, 10, 3), np.uint8))
    assert (tmp_path / "tief" / "drin" / "out.mp4").exists()


# --------------------------------------------------------------------------- #
# Tabellen                                                                    #
# --------------------------------------------------------------------------- #
def test_write_rows_takes_columns_from_the_first_row(tmp_path):
    target = tmp_path / "out.csv"
    assert write_rows([{"a": 1, "b": 2}, {"a": 3, "b": 4}], target) == 2
    assert list(csv.DictReader(target.open()))[0] == {"a": "1", "b": "2"}


def test_write_rows_writes_nothing_for_an_empty_input(tmp_path):
    """Ohne Zeile gibt es keine Spalten -- eine Datei mit erfundenem Kopf waere
    schlechter als keine."""
    target = tmp_path / "out.csv"
    assert write_rows([], target) == 0
    assert not target.exists()


def test_write_named_keeps_the_header_stable_when_a_field_is_missing(tmp_path):
    target = tmp_path / "out.csv"
    write_named([{"a": 1}], target, ["a", "b", "c"])
    row = next(iter(csv.DictReader(target.open())))
    assert row == {"a": "1", "b": "", "c": ""}

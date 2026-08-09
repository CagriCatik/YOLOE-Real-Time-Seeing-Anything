"""Boxen und Farben je Maske im Datenwerkzeug.

Die PyQt-Oberflaeche laesst sich nicht ohne Anzeige testen, die Logik dahinter
schon -- und dort sitzt alles, was schiefgehen kann: das Dateiformat, die
Rueckwaertskompatibilitaet und die Farbzuordnung beim Fuellen.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from adascope.tool.core import (
    DEFAULT_FILL_BGR, MASK_CONFIG_VERSION, box_polygon, create_mask_outputs,
    load_mask_config, mask_fill_colors, save_mask_config,
)

SIZE = (200, 100)          # Breite, Hoehe


@pytest.fixture
def image(tmp_path):
    path = tmp_path / "frame.png"
    cv2.imwrite(str(path), np.full((SIZE[1], SIZE[0], 3), 180, np.uint8))
    return path


# --------------------------------------------------------------------------- #
# Box als Vierpunkt-Polygon                                                   #
# --------------------------------------------------------------------------- #
def test_box_becomes_four_corners_clockwise():
    assert box_polygon(0.2, 0.3, 0.8, 0.7) == [[0.2, 0.3], [0.8, 0.3],
                                               [0.8, 0.7], [0.2, 0.7]]


def test_box_normalises_a_backwards_drag():
    """Von rechts unten nach links oben gezogen ist dieselbe Box.

    Ohne das haette die Box negative Kantenlaengen und `fillPoly` wuerde je
    nach Reihenfolge unterschiedlich fuellen.
    """
    assert box_polygon(0.8, 0.7, 0.2, 0.3) == box_polygon(0.2, 0.3, 0.8, 0.7)


def test_a_box_is_a_normal_polygon_downstream(tmp_path, image):
    """Der Kern der Entscheidung: keine zweite Datenstruktur.

    Eine Box durchlaeuft Speichern, Laden und Fuellen exakt wie ein von Hand
    gezeichnetes Polygon.
    """
    config = tmp_path / "masks.json"
    save_mask_config(config, SIZE, {"kasten": box_polygon(0.1, 0.1, 0.5, 0.5)},
                     mask_shapes={"kasten": "box"})
    loaded = load_mask_config(config)
    assert len(loaded["masks"]["kasten"]) == 4
    assert loaded["mask_shapes"]["kasten"] == "box"


# --------------------------------------------------------------------------- #
# Farbe je Maske                                                              #
# --------------------------------------------------------------------------- #
def test_each_mask_keeps_its_own_colour(tmp_path):
    config = tmp_path / "masks.json"
    save_mask_config(config, SIZE,
                     {"a": box_polygon(0, 0, .3, .3), "b": box_polygon(.6, .6, .9, .9)},
                     mask_colors={"a": (255, 0, 0), "b": (0, 0, 255)})
    colors = mask_fill_colors(load_mask_config(config))
    assert colors == {"a": (255, 0, 0), "b": (0, 0, 255)}


def test_a_mask_without_its_own_colour_falls_back_to_the_global_one(tmp_path):
    config = tmp_path / "masks.json"
    save_mask_config(config, SIZE,
                     {"a": box_polygon(0, 0, .3, .3), "b": box_polygon(.6, .6, .9, .9)},
                     fill_color=(10, 20, 30), mask_colors={"a": (255, 0, 0)})
    colors = mask_fill_colors(load_mask_config(config))
    assert colors["a"] == (255, 0, 0)
    assert colors["b"] == (10, 20, 30)


def test_colour_of_a_deleted_mask_is_dropped(tmp_path):
    """Sonst waechst die Datei bei jedem Umbenennen um eine Leiche."""
    config = tmp_path / "masks.json"
    save_mask_config(config, SIZE, {"a": box_polygon(0, 0, .3, .3)},
                     mask_colors={"a": (1, 2, 3), "geloescht": (9, 9, 9)})
    assert set(load_mask_config(config)["mask_colors"]) == {"a"}


def test_an_invalid_colour_is_rejected_on_save(tmp_path):
    with pytest.raises(ValueError, match="BGR"):
        save_mask_config(tmp_path / "m.json", SIZE, {"a": box_polygon(0, 0, .3, .3)},
                         mask_colors={"a": (300, 0, 0)})


# --------------------------------------------------------------------------- #
# Fuellen                                                                     #
# --------------------------------------------------------------------------- #
def test_each_mask_is_filled_with_its_own_colour(tmp_path, image):
    config = tmp_path / "masks.json"
    save_mask_config(config, SIZE,
                     {"links": box_polygon(.05, .05, .35, .95),
                      "rechts": box_polygon(.65, .05, .95, .95)},
                     mask_colors={"links": (255, 0, 0), "rechts": (0, 0, 255)})
    masked_path, debug_path = create_mask_outputs(
        image, config, tmp_path / "masked.png", tmp_path / "debug.png")

    masked = cv2.imread(str(masked_path))
    assert tuple(masked[50, 40]) == (255, 0, 0)      # in "links"
    assert tuple(masked[50, 160]) == (0, 0, 255)     # in "rechts"
    assert tuple(masked[50, 100]) == (180, 180, 180)  # dazwischen unveraendert
    assert debug_path.exists()


def test_later_masks_win_where_they_overlap(tmp_path, image):
    """Dieselbe Reihenfolge wie in der Maskenliste des Editors."""
    config = tmp_path / "masks.json"
    save_mask_config(config, SIZE,
                     {"unten": box_polygon(.1, .1, .9, .9),
                      "oben": box_polygon(.4, .4, .6, .6)},
                     mask_colors={"unten": (255, 0, 0), "oben": (0, 255, 0)})
    masked = cv2.imread(str(create_mask_outputs(
        image, config, tmp_path / "m.png", tmp_path / "d.png")[0]))
    assert tuple(masked[50, 100]) == (0, 255, 0)


# --------------------------------------------------------------------------- #
# Rueckwaertskompatibilitaet                                                  #
# --------------------------------------------------------------------------- #
def test_a_version_1_file_still_loads(tmp_path, image):
    """Dateien ohne Farb- und Formangaben bleiben gueltig.

    Sie fallen auf die globale Fuellfarbe zurueck und gelten als Polygon.
    """
    config = tmp_path / "alt.json"
    config.write_text(json.dumps({
        "version": 1,
        "coordinate_system": "normalized",
        "source_size": {"width": SIZE[0], "height": SIZE[1]},
        "fill_color_bgr": [0, 0, 0],
        "masks": {"alt": [[.1, .1], [.5, .1], [.5, .5], [.1, .5]]},
    }), encoding="utf-8")

    loaded = load_mask_config(config)
    assert loaded["mask_colors"] == {} and loaded["mask_shapes"] == {}
    assert mask_fill_colors(loaded) == {"alt": (0, 0, 0)}

    masked = cv2.imread(str(create_mask_outputs(
        image, config, tmp_path / "m.png", tmp_path / "d.png")[0]))
    assert tuple(masked[20, 40]) == (0, 0, 0)


def test_new_files_declare_version_2(tmp_path):
    config = tmp_path / "neu.json"
    save_mask_config(config, SIZE, {"a": box_polygon(0, 0, .3, .3)})
    assert json.loads(config.read_text(encoding="utf-8"))["version"] == MASK_CONFIG_VERSION


def test_a_colour_without_a_mask_is_rejected_on_load(tmp_path):
    config = tmp_path / "kaputt.json"
    config.write_text(json.dumps({
        "masks": {"a": [[.1, .1], [.5, .1], [.5, .5]]},
        "mask_colors": {"gibt_es_nicht": [1, 2, 3]},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="ohne zugehoerige Maske"):
        load_mask_config(config)


def test_default_fill_is_black():
    assert DEFAULT_FILL_BGR == (0, 0, 0)


# --------------------------------------------------------------------------- #
# Fuellfarbe und Anzeigefarbe sind zwei verschiedene Dinge                    #
# --------------------------------------------------------------------------- #
def test_without_a_choice_the_mask_is_blacked_out(tmp_path, image):
    """Maskieren heisst schwaerzen -- das darf keine Palettenfarbe aushebeln."""
    from adascope.tool.core import mask_display_colors

    config = tmp_path / "masks.json"
    save_mask_config(config, SIZE, {"a": box_polygon(.1, .1, .9, .9)})
    loaded = load_mask_config(config)

    assert mask_fill_colors(loaded)["a"] == (0, 0, 0)
    masked = cv2.imread(str(create_mask_outputs(
        image, config, tmp_path / "m.png", tmp_path / "d.png")[0]))
    assert tuple(masked[50, 100]) == (0, 0, 0)

    # Im Debugbild waere Schwarz nicht auffindbar -- dort tritt eine
    # Palettenfarbe an seine Stelle.
    assert mask_display_colors(loaded)["a"] != (0, 0, 0)


def test_a_chosen_colour_reaches_the_masked_image(tmp_path, image):
    config = tmp_path / "masks.json"
    save_mask_config(config, SIZE, {"a": box_polygon(.1, .1, .9, .9)},
                     mask_colors={"a": (0, 200, 0)})
    masked = cv2.imread(str(create_mask_outputs(
        image, config, tmp_path / "m.png", tmp_path / "d.png")[0]))
    assert tuple(masked[50, 100]) == (0, 200, 0)


def test_a_very_dark_choice_still_shows_up_in_the_debug_image(tmp_path):
    """Wer Dunkelgrau waehlt, soll die Maske trotzdem wiederfinden."""
    from adascope.tool.core import mask_display_colors

    config = tmp_path / "masks.json"
    save_mask_config(config, SIZE, {"a": box_polygon(.1, .1, .9, .9)},
                     mask_colors={"a": (10, 10, 10)})
    loaded = load_mask_config(config)
    assert mask_fill_colors(loaded)["a"] == (10, 10, 10)      # Fuellung bleibt
    assert max(mask_display_colors(loaded)["a"]) > 100        # Anzeige nicht

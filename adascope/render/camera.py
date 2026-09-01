"""Virtuelle Kamera auf die BEV-Bodenebene.

Weil BEV und virtuelle Ansicht Bilder *derselben* Ebene sind, ist die Abbildung
dazwischen wieder eine Homographie. Eine zusaetzliche Perspektive kostet damit
einen `warpPerspective` und keine neue Kalibrierung -- deshalb sind Kameras
reine Konfiguration (configs/debug.yaml), nicht Code.
"""

from __future__ import annotations

from math import cos, radians, sin

import cv2
import numpy as np

from ..config import BevConfig, VirtualCamConfig


class VirtualCamera:
    """Lochkamera-Projektion von BEV-Punkten in eine virtuelle Bildebene."""

    def __init__(self, config: VirtualCamConfig, bev: BevConfig):
        self.config, self.bev = config, bev

    @property
    def width(self) -> int:
        return self.config.width

    @property
    def height(self) -> int:
        return self.config.height

    def project(self, points_bev, elevation: float = 0.0) -> np.ndarray:
        """BEV-Punkte (N,2) -> Bildpunkte (N,2). `elevation` hebt ueber den Boden.

        Mit `elevation > 0` lassen sich Bodenquader zeichnen: Grundflaeche auf
        der Ebene, Deckflaeche darueber. Alles ausserhalb der Ebene ist eine
        Annahme, keine Messung -- deshalb wird die Fahrzeughoehe konfiguriert
        und nicht geschaetzt.
        """
        cfg = self.config
        pts = np.asarray(points_bev, np.float64).reshape(-1, 2)
        xw = pts[:, 0] - cfg.x
        zw = (self.bev.height - pts[:, 1]) + cfg.behind
        yw = np.full_like(xw, cfg.elevation - elevation)

        cy, sy = cos(radians(cfg.yaw_deg)), sin(radians(cfg.yaw_deg))
        xc, zc = cy * xw - sy * zw, sy * xw + cy * zw
        cp, sp = cos(radians(cfg.pitch_deg)), sin(radians(cfg.pitch_deg))
        yc, zc = cp * yw - sp * zc, sp * yw + cp * zc

        zc = np.maximum(zc, 1e-3)                      # hinter der Kamera abfangen
        return np.stack([cfg.focal * xc / zc + cfg.width / 2,
                         cfg.focal * yc / zc + cfg.height * cfg.cy_ratio], axis=1)

    def project_int(self, points_bev, elevation: float = 0.0) -> np.ndarray:
        return self.project(points_bev, elevation).astype(np.int32)

    @property
    def horizon_y(self) -> int:
        cfg = self.config
        return int(cfg.height * cfg.cy_ratio - cfg.focal * np.tan(radians(cfg.pitch_deg)))

    def homography(self) -> np.ndarray:
        """BEV-Bild -> virtuelles Bild, exakt fuer die Bodenebene."""
        src = np.float32([[0, 0], [self.bev.width, 0],
                          [self.bev.width, self.bev.height], [0, self.bev.height]])
        return cv2.getPerspectiveTransform(src, self.project(src).astype(np.float32))

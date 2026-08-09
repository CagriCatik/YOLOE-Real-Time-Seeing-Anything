"""Detector interface and the lazy-loaded Ultralytics YOLOE adapter."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
import numpy as np


@dataclass(frozen=True)
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0
    label: str | None = None

    @property
    def center(self):
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2


class Detector(Protocol):
    def detect(self, frame: np.ndarray, conf: float) -> list[BBox]: ...


class YoloeDetector:
    def __init__(self, checkpoint: str, classes: list[str] | None = None):
        try:
            from ultralytics import YOLOE
        except ImportError as exc:
            raise RuntimeError("Ultralytics is required for detection; run `pip install -e .`") from exc
        self.classes = classes or []
        self.model = YOLOE(checkpoint)
        if self.classes:
            embeddings = self.model.get_text_pe(self.classes)
            self.model.set_classes(self.classes, embeddings)

    def predict_labelled(self, frame, conf: float):
        result = self.model.predict(frame, conf=conf, verbose=False)[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return [], result
        xyxy = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        ids = boxes.cls.cpu().numpy().astype(int)
        names = getattr(result, "names", {})
        return [BBox(*map(float, coords), float(score), names.get(int(idx), str(idx)))
                for coords, score, idx in zip(xyxy, scores, ids)], result

    def detect(self, frame, conf: float) -> list[BBox]:
        return self.predict_labelled(frame, conf)[0]

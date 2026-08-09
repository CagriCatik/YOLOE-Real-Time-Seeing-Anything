"""Rendering: Overlays und Debug-Ansichten. Liest Domaenenobjekte, aendert sie nie."""

from .camera import VirtualCamera
from .debug_views import Dashboard, available_views, make_view
from .overlay import draw
from .primitives import dashed_line, dashed_polyline, fit, hud, placeholder

__all__ = [
    "Dashboard", "VirtualCamera", "available_views", "dashed_line",
    "dashed_polyline", "draw", "fit", "hud", "make_view", "placeholder",
]

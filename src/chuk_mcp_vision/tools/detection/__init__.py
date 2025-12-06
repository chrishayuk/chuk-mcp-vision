"""Detection tools - lines, contours, shapes."""

from .lines import detect_lines
from .contours import detect_contours

__all__ = [
    "detect_lines",
    "detect_contours",
]

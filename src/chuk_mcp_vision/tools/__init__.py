"""Vision tools organized by category."""

from .analyze.tool import analyze
from .detection.contours import detect_contours
from .detection.lines import detect_lines
from .embeddings.tool import embed_image
from .transform.crop import extract_region
from .ocr.tool import ocr_region
from .transform.perspective import perspective_transform

__all__ = [
    "analyze",
    "detect_lines",
    "detect_contours",
    "perspective_transform",
    "extract_region",
    "ocr_region",
    "embed_image",
]

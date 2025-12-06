"""Classify the content of a grid cell - generic for any content type."""

import cv2
import numpy as np
from chuk_mcp_server import tool

from ...models import CellContent, CellContentType
from ..common import load_image_from_artifact


@tool
async def classify_cell_content(
    cell_uri: str, expected_type: str = "auto", ocr_backend: str = "paddleocr"
) -> CellContent:
    """Classify what's in a grid cell.

    This is a generic tool that works for any grid content:
    - Digits (Sudoku, phone keypads)
    - Text (spreadsheets, forms)
    - Checkboxes (forms, checklists)
    - Images/icons (periodic table, emoji keyboards)
    - Empty cells

    Args:
        cell_uri: URI to the cell image (from extract_grid_cells)
        expected_type: Type hint - "digit", "text", "checkbox", "auto" (default)
        ocr_backend: OCR engine to use for text/digit recognition

    Returns:
        CellContent with:
        - content_type: What type of content was detected
        - value: The actual content (digit, text, boolean, None)
        - confidence: Detection confidence (0.0-1.0)
        - is_empty: Whether cell appears empty
    """
    img = await load_image_from_artifact(cell_uri)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Quick empty check
    mean_val = np.mean(gray)
    if mean_val > 250 or mean_val < 5:  # Very white or very black = likely empty
        return CellContent(
            content_type=CellContentType.EMPTY, value=None, confidence=1.0, is_empty=True
        )

    # Determine content type
    if expected_type == "auto":
        expected_type = _auto_detect_type(gray)

    # Route to appropriate classifier
    if expected_type == "digit":
        return await _classify_digit(img, ocr_backend)
    elif expected_type == "text":
        return await _classify_text(img, ocr_backend)
    elif expected_type == "checkbox":
        return await _classify_checkbox(gray)
    else:
        # Default to text
        return await _classify_text(img, ocr_backend)


def _auto_detect_type(gray) -> str:
    """Auto-detect likely content type based on image features."""
    h, w = gray.shape

    # Small square image with binary content → likely checkbox
    aspect = w / h
    if 0.8 <= aspect <= 1.2 and w < 100:
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        unique_vals = len(np.unique(binary))
        if unique_vals <= 3:  # Very binary
            return "checkbox"

    # Check for numeric-looking content
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Single large contour → likely single digit/character
    if len(contours) == 1:
        return "digit"  # Could be digit or single letter

    # Multiple contours or complex → text
    return "text"


async def _classify_digit(img, backend: str) -> CellContent:
    """Classify as a single digit (0-9)."""
    # Use the existing OCR tool instead of calling PaddleOCR directly
    from ..common import save_image_as_artifact
    from ..ocr.tool import ocr_region
    from ...models import OCRBackend

    # Save cell image temporarily
    cell_uri = await save_image_as_artifact(img, suffix=".jpg", category="temp_cell")

    # Use existing OCR tool
    ocr_result = await ocr_region(cell_uri, backend=OCRBackend.PADDLEOCR, language="en")

    if ocr_result.full_text:
        text = ocr_result.full_text.strip()
        # Extract digits only
        digits = "".join(c for c in text if c.isdigit())
        if digits:
            return CellContent(
                content_type=CellContentType.DIGIT,
                value=int(digits[0]),
                confidence=ocr_result.mean_confidence,
                is_empty=False,
                raw_text=text,
            )

    # No digit found
    return CellContent(
        content_type=CellContentType.EMPTY, value=None, confidence=0.5, is_empty=True
    )


async def _classify_text(img, backend: str) -> CellContent:
    """Classify as text (words, phrases)."""
    # Use the existing OCR tool
    from ..common import save_image_as_artifact
    from ..ocr.tool import ocr_region
    from ...models import OCRBackend

    # Save cell image temporarily
    cell_uri = await save_image_as_artifact(img, suffix=".jpg", category="temp_cell")

    # Use existing OCR tool
    ocr_result = await ocr_region(cell_uri, backend=OCRBackend.PADDLEOCR, language="en")

    if ocr_result.full_text:
        return CellContent(
            content_type=CellContentType.TEXT,
            value=ocr_result.full_text,
            confidence=ocr_result.mean_confidence,
            is_empty=False,
            raw_text=ocr_result.full_text,
        )

    # No text found
    return CellContent(
        content_type=CellContentType.EMPTY, value=None, confidence=0.5, is_empty=True
    )


async def _classify_checkbox(gray) -> CellContent:
    """Classify as a checkbox (checked/unchecked)."""
    # Threshold image
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Calculate fill percentage
    fill_pct = np.sum(binary > 0) / binary.size

    # Checked if > 20% filled (allows for X, checkmark, or filled square)
    is_checked = fill_pct > 0.2

    return CellContent(
        content_type=CellContentType.CHECKBOX,
        value=is_checked,
        confidence=0.9 if fill_pct > 0.3 or fill_pct < 0.1 else 0.7,
        is_empty=not is_checked,
        metadata={"fill_percentage": fill_pct},
    )

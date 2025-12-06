"""OCR tool with multiple backend support."""

import os
from pathlib import Path

# Set TESSDATA_PREFIX before importing pytesseract
if "TESSDATA_PREFIX" not in os.environ:
    tessdata_dirs = [
        "/opt/homebrew/share/tessdata",
        "/usr/share/tessdata",
        "/usr/local/share/tessdata",
    ]
    for tdir in tessdata_dirs:
        if Path(tdir).exists():
            os.environ["TESSDATA_PREFIX"] = str(Path(tdir).parent) + "/"
            break

import cv2
from chuk_mcp_server import tool
from PIL import Image

from ...models import (
    BoundingBox,
    OCRBackend,
    OCRLine,
    OCRRegionResponse,
    OCRWord,
)
from ..common import load_image_from_artifact


@tool
async def ocr_region(
    image_uri: str,
    backend: OCRBackend = OCRBackend.PADDLEOCR,
    language: str = "en",
) -> OCRRegionResponse:
    """Extract text and digits from an image using OCR.

    Supports multiple OCR backends with different tradeoffs:
    - PaddleOCR: High accuracy, GPU-accelerated, good for complex layouts
    - Tesseract: CPU-friendly, good for simple printed text

    Args:
        image_uri: URI or file path to the image
        backend: OCR engine to use (paddleocr or tesseract)
        language: Language code (e.g., "en", "zh", "es")

    Returns:
        OCRRegionResponse: Contains:
            - full_text: All detected text concatenated
            - lines: List of text lines with individual words
            - backend_used: Which OCR engine was used
            - language: Language detected/used
            - mean_confidence: Average confidence score

    Example:
        # Read a digit from Sudoku cell
        result = await ocr_region("cell.jpg", backend="paddleocr")
        # result.full_text = "5"
        # result.mean_confidence = 0.95

        # Read a receipt
        result = await ocr_region("receipt.jpg")
        # result.full_text contains all text
        # result.lines[0].text = "ACME Store"
        # result.lines[1].text = "Total: $42.50"

    Use cases:
        - Digit recognition (Sudoku, captchas, meters)
        - Document text extraction
        - Receipt/invoice processing
        - License plate reading
        - Handwriting recognition (with appropriate backend)
    """
    img = await load_image_from_artifact(image_uri)

    if backend == OCRBackend.PADDLEOCR:
        return await _ocr_paddleocr(img, language)
    elif backend == OCRBackend.TESSERACT:
        return await _ocr_tesseract(img, language)
    else:
        raise ValueError(f"Unknown OCR backend: {backend}")


async def _ocr_paddleocr(img, language: str) -> OCRRegionResponse:
    """OCR using PaddleOCR."""
    try:
        from paddleocr import PaddleOCR

        # Initialize PaddleOCR (v3.x API)
        # Note: PaddleOCR 3.x changed the API significantly
        ocr = PaddleOCR(lang=language)

        # Run OCR (PaddleOCR v3.x API)
        # result is a list of OCRResult objects (one per image/page)
        results = ocr.predict(img)

        if not results or not results[0]:
            return OCRRegionResponse(
                full_text="",
                lines=[],
                backend_used=OCRBackend.PADDLEOCR,
                language=language,
                mean_confidence=0.0,
            )

        # Extract text and scores from first result
        ocr_result = results[0]
        rec_texts = ocr_result.get("rec_texts", [])
        rec_scores = ocr_result.get("rec_scores", [])
        rec_polys = ocr_result.get("rec_polys", [])

        if not rec_texts:
            return OCRRegionResponse(
                full_text="",
                lines=[],
                backend_used=OCRBackend.PADDLEOCR,
                language=language,
                mean_confidence=0.0,
            )

        # Parse results
        lines = []
        all_confidences = []

        for i, text in enumerate(rec_texts):
            confidence = rec_scores[i] if i < len(rec_scores) else 0.0
            poly = rec_polys[i] if i < len(rec_polys) else None

            # Bounding box from polygon
            if poly is not None and len(poly) > 0:
                x_coords = [pt[0] for pt in poly]
                y_coords = [pt[1] for pt in poly]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                bbox = BoundingBox(
                    x=float(x_min),
                    y=float(y_min),
                    width=float(x_max - x_min),
                    height=float(y_max - y_min),
                )
            else:
                bbox = BoundingBox(x=0.0, y=0.0, width=0.0, height=0.0)

            # Create word (PaddleOCR gives line-level, not word-level)
            word = OCRWord(text=text, confidence=float(confidence), bounding_box=bbox)

            lines.append(OCRLine(text=text, words=[word], bounding_box=bbox))
            all_confidences.append(float(confidence))

        full_text = "\n".join(line.text for line in lines)
        mean_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0

        return OCRRegionResponse(
            full_text=full_text,
            lines=lines,
            backend_used=OCRBackend.PADDLEOCR,
            language=language,
            mean_confidence=mean_confidence,
        )

    except ImportError:
        raise RuntimeError("PaddleOCR not installed. Install with: pip install paddleocr")


async def _ocr_tesseract(img, language: str) -> OCRRegionResponse:
    """OCR using Tesseract."""
    try:
        import pytesseract

        # Convert to PIL Image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        # Run OCR with detailed data
        # TESSDATA_PREFIX was set at module level, so tesseract should find it
        data = pytesseract.image_to_data(
            pil_img, lang=language, output_type=pytesseract.Output.DICT
        )

        # Parse results
        lines_dict = {}  # Group by line number
        all_confidences = []

        for i, text in enumerate(data["text"]):
            if not text.strip():
                continue

            conf = float(data["conf"][i]) / 100.0  # Convert to 0-1
            if conf < 0:
                conf = 0.0

            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            bbox = BoundingBox(x=float(x), y=float(y), width=float(w), height=float(h))

            word = OCRWord(text=text, confidence=conf, bounding_box=bbox)

            line_num = data["line_num"][i]
            if line_num not in lines_dict:
                lines_dict[line_num] = []
            lines_dict[line_num].append(word)
            all_confidences.append(conf)

        # Build lines
        lines = []
        for line_words in lines_dict.values():
            line_text = " ".join(w.text for w in line_words)

            # Compute line bounding box
            xs = [w.bounding_box.x for w in line_words]
            ys = [w.bounding_box.y for w in line_words]
            ws = [w.bounding_box.width for w in line_words]
            hs = [w.bounding_box.height for w in line_words]

            x_min = min(xs)
            y_min = min(ys)
            x_max = max(x + w for x, w in zip(xs, ws))
            y_max = max(y + h for y, h in zip(ys, hs))

            line_bbox = BoundingBox(x=x_min, y=y_min, width=x_max - x_min, height=y_max - y_min)

            lines.append(OCRLine(text=line_text, words=line_words, bounding_box=line_bbox))

        full_text = "\n".join(line.text for line in lines)
        mean_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0

        return OCRRegionResponse(
            full_text=full_text,
            lines=lines,
            backend_used=OCRBackend.TESSERACT,
            language=language,
            mean_confidence=mean_confidence,
        )

    except ImportError:
        raise RuntimeError("pytesseract not installed. Install with: pip install pytesseract")

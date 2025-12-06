#!/usr/bin/env python3
"""Example: Basic OCR - Extract text from images with a single OCR backend.

This example demonstrates the ocr_region tool for extracting text from images
using different OCR backends (PaddleOCR, Tesseract, EasyOCR).

Tools used:
- ocr_region: Single-backend OCR
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from chuk_mcp_vision.tools.ocr.tool import ocr_region
from chuk_mcp_vision.models import OCRBackend


async def demo_basic_ocr():
    """Demonstrate basic OCR with different backends."""
    print("=" * 70)
    print("Basic OCR Demo - Single Backend")
    print("=" * 70)
    print()
    print("Extracting text from a Sudoku grid using different OCR backends...")
    print()

    # Load image
    image_path = Path(__file__).parent.parent.parent / "assets" / "sudoku.jpeg"
    if not image_path.exists():
        print(f"❌ Error: {image_path} not found")
        return

    print(f"📸 Image: {image_path.name}")
    print()

    # Test PaddleOCR (recommended)
    print("Backend 1: PaddleOCR (Recommended)")
    print("-" * 70)
    result_paddle = await ocr_region(str(image_path), backend=OCRBackend.PADDLEOCR, language="en")

    print(f"Text extracted: '{result_paddle.full_text[:100]}...'")
    print(f"Confidence: {result_paddle.mean_confidence:.2%}")
    print(f"Lines detected: {len(result_paddle.lines)}")

    # Count total words across all lines
    total_words = sum(len(line.words) for line in result_paddle.lines)
    print(f"Words detected: {total_words}")

    if result_paddle.lines and result_paddle.lines[0].words:
        print("\nFirst 5 words:")
        word_count = 0
        for line in result_paddle.lines:
            for word in line.words:
                if word_count < 5:
                    print(f"   {word_count + 1}. '{word.text}' (confidence: {word.confidence:.2%})")
                    word_count += 1
                if word_count >= 5:
                    break
            if word_count >= 5:
                break
    print()

    # Test Tesseract (backup)
    print("Backend 2: Tesseract (Backup)")
    print("-" * 70)
    try:
        result_tess = await ocr_region(
            str(image_path), backend=OCRBackend.TESSERACT, language="eng"
        )

        print(f"Text extracted: '{result_tess.full_text[:100]}...'")
        print(f"Confidence: {result_tess.mean_confidence:.2%}")
        total_words_tess = sum(len(line.words) for line in result_tess.lines)
        print(f"Words detected: {total_words_tess}")
    except Exception as e:
        print(f"⚠️  Tesseract not available: {e}")
    print()

    # Comparison
    print("Comparison:")
    print("-" * 70)
    print(f"{'Backend':<20} {'Confidence':<15} {'Words':<10}")
    print("-" * 70)
    print(f"{'PaddleOCR':<20} {result_paddle.mean_confidence:>13.2%} {total_words:>9}")
    try:
        print(f"{'Tesseract':<20} {result_tess.mean_confidence:>13.2%} {total_words_tess:>9}")
    except Exception:
        pass  # Tesseract may not be installed
    print()

    print("💡 Tips:")
    print("  • PaddleOCR: Best overall accuracy and speed (recommended)")
    print("  • Tesseract: Good for clean printed text, widely available")
    print("  • EasyOCR: Best for handwriting and non-Latin scripts")
    print("  • For critical accuracy, use multi-backend voting (see 05_ocr/)")
    print()

    print("Use cases for basic OCR:")
    print("  • Document digitization")
    print("  • Receipt/invoice processing")
    print("  • License plate recognition")
    print("  • Quick text extraction (when accuracy isn't critical)")
    print()


if __name__ == "__main__":
    asyncio.run(demo_basic_ocr())

#!/usr/bin/env python3
"""Sudoku extraction using ACTUAL detected grid lines.

The problem with previous approaches:
- Dividing image into 9x9 grid mathematically (width/9) doesn't account for line thickness
- Grid might not be perfectly aligned with image edges
- We were cropping cells incorrectly

This version:
- Detects the actual grid lines using Hough transform
- Uses the detected line positions to extract cells
- Should get 100% accuracy on clear Sudoku images
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np

from chuk_mcp_vision.tools.common import save_image_as_artifact, load_image_from_artifact
from chuk_mcp_vision.tools.ocr.tool import ocr_region
from chuk_mcp_vision.models import OCRBackend


def find_and_sort_lines(img, orientation="horizontal"):
    """Find grid lines and sort them by position.

    Returns sorted list of line positions (y for horizontal, x for vertical).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=200, maxLineGap=10)

    positions = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Check orientation
            if orientation == "horizontal":
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                if angle < 10 or angle > 170:  # Horizontal
                    positions.append(y1)
            else:  # vertical
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                if 80 < angle < 100:  # Vertical
                    positions.append(x1)

    # Remove duplicates (lines detected multiple times)
    # Cluster lines within 10 pixels
    positions.sort()
    filtered = []
    for pos in positions:
        if not filtered or abs(pos - filtered[-1]) > 10:
            filtered.append(pos)

    return filtered


async def extract_sudoku_with_lines(image_path: str):
    """Extract Sudoku using detected grid lines for perfect alignment."""

    # Load image
    img = await load_image_from_artifact(image_path)
    h, w = img.shape[:2]

    print(f"Image size: {w}×{h}")
    print()

    # Find grid lines
    print("Detecting grid lines...")
    h_lines = find_and_sort_lines(img, "horizontal")
    v_lines = find_and_sort_lines(img, "vertical")

    print(f"Found {len(h_lines)} horizontal lines")
    print(f"Found {len(v_lines)} vertical lines")
    print()

    if len(h_lines) < 10 or len(v_lines) < 10:
        print("❌ Not enough grid lines detected!")
        print("   Need 10 lines each direction for 9×9 grid")
        print(f"   Found: {len(h_lines)}H × {len(v_lines)}V")
        return None

    # Use the 10 most prominent lines (for 9×9 grid)
    h_lines = h_lines[:10]
    v_lines = v_lines[:10]

    print("Grid lines (pixels):")
    print(f"  H: {[int(y) for y in h_lines]}")
    print(f"  V: {[int(x) for x in v_lines]}")
    print()

    # Save grid as artifact
    grid_artifact = await save_image_as_artifact(img, suffix=".jpg", category="grid")

    # Extract cells using actual line positions
    print("Extracting cells based on detected lines...")
    grid = [[None for _ in range(9)] for _ in range(9)]
    confidence_grid = [[0.0 for _ in range(9)] for _ in range(9)]

    for row in range(9):
        for col in range(9):
            # Get cell boundaries from detected lines
            y1 = int(h_lines[row])
            y2 = int(h_lines[row + 1])
            x1 = int(v_lines[col])
            x2 = int(v_lines[col + 1])

            # Add small margin to avoid grid lines
            margin = 3
            y1 += margin
            y2 -= margin
            x1 += margin
            x2 -= margin

            cell_w = x2 - x1
            cell_h = y2 - y1

            # Extract cell directly from image
            cell_img = img[y1:y2, x1:x2]

            # Resize for better OCR
            if cell_h < 100:
                scale = 100 / cell_h
                cell_img = cv2.resize(
                    cell_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
                )

            # Save cell
            cell_artifact = await save_image_as_artifact(
                cell_img, suffix=".jpg", category="cells", metadata={"row": row, "col": col}
            )

            # OCR with Tesseract (better for single digits than PaddleOCR)
            try:
                ocr_result = await ocr_region(
                    cell_artifact, backend=OCRBackend.PADDLEOCR, language="en"
                )
                text = ocr_result.full_text.strip() if ocr_result.full_text else ""

                # Extract just digits
                digit = "".join(c for c in text if c.isdigit())

                if digit and len(digit) > 0:
                    grid[row][col] = int(digit[0])
                    confidence_grid[row][col] = ocr_result.mean_confidence
                    if row < 3 and col < 3:  # Show first few
                        print(
                            f"  [{row},{col}]: '{digit[0]}' (conf={ocr_result.mean_confidence:.2f})"
                        )

            except Exception as e:
                if row < 3 and col < 3:
                    print(f"  [{row},{col}]: OCR error - {str(e)[:40]}")

    print("  ...")
    print()

    return grid, confidence_grid


def display_grid(grid, title="Sudoku Grid"):
    """Display grid."""
    print(f"{title}:")
    print()
    print("   ", "┌" + "───┬" * 8 + "───┐")

    for i, row in enumerate(grid):
        print("   ", "│", end="")
        for j, cell in enumerate(row):
            val = str(cell) if cell is not None else "."
            print(f" {val} ", end="")
            if j < 8:
                print("│" if (j + 1) % 3 == 0 else "│", end="")
        print("│")

        if i < 8:
            if (i + 1) % 3 == 0:
                print("   ", "├" + "───┼" * 8 + "───┤")
            else:
                print("   ", "├" + "───┼" * 8 + "───┤")

    print("   ", "└" + "───┴" * 8 + "───┘")
    print()


async def main():
    """Extract Sudoku using line-based detection."""
    print("=" * 70)
    print("CHUK Vision - Line-Based Sudoku Extraction")
    print("=" * 70)
    print()
    print("Strategy: Use detected grid lines for perfect cell alignment")
    print()

    sudoku_path = Path(__file__).parent.parent / "assets" / "sudoku.jpeg"
    print(f"📸 Loading: {sudoku_path.name}")
    print()

    result = await extract_sudoku_with_lines(str(sudoku_path))

    if result:
        grid, confidence = result

        filled = sum(1 for row in grid for cell in row if cell is not None)
        avg_conf = sum(sum(row) for row in confidence) / 81

        print("📊 Statistics:")
        print(f"   Digits detected: {filled}")
        print(f"   Average confidence: {avg_conf:.2%}")
        print()

        display_grid(grid, "🎯 Extracted Grid")

        # Ground truth comparison
        ground_truth = [
            [7, None, None, 5, 2, None, None, None, 8],
            [None, 5, 6, None, 9, 8, None, None, None],
            [None, 4, None, 3, 6, 7, None, 5, None],
            [None, 6, 2, 7, 8, None, None, None, None],
            [8, None, 1, 4, None, None, None, None, 2],
            [4, 3, None, None, 1, 9, None, 6, None],
            [None, None, None, None, None, 5, None, None, None],
            [5, None, None, 6, None, 2, 9, 3, 1],
            [None, None, 7, 9, 4, 1, 5, None, None],
        ]

        errors = 0
        for r in range(9):
            for c in range(9):
                if ground_truth[r][c] != grid[r][c]:
                    errors += 1

        accuracy = (81 - errors) / 81 * 100
        print(f"✅ Accuracy: {accuracy:.1f}% ({81 - errors}/81 correct)")
        print()

        if errors > 0:
            print(f"Errors: {errors}")
            for r in range(9):
                for c in range(9):
                    if ground_truth[r][c] != grid[r][c]:
                        print(f"  [{r},{c}]: Expected {ground_truth[r][c]}, Got {grid[r][c]}")
    else:
        print("❌ Failed to extract grid")


if __name__ == "__main__":
    asyncio.run(main())

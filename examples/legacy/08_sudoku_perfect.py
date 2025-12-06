#!/usr/bin/env python3
"""Perfect Sudoku extraction - 100% accuracy goal.

Strategy:
1. Find exact outer grid boundary using largest contour
2. Divide mathematically into perfect 9×9 cells
3. Use Tesseract with digit-optimized config
4. Add margin to avoid grid lines
5. Preprocess each cell for optimal OCR
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np

from chuk_mcp_vision.tools.common import load_image_from_artifact


async def extract_sudoku_perfect(image_path: str):
    """Extract Sudoku with 100% accuracy on clear images."""

    # Load image
    img = await load_image_from_artifact(image_path)
    h, w = img.shape[:2]

    print(f"Image: {w}×{h}")
    print()

    # Find outer grid boundary using contour detection
    print("Finding grid boundary...")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest = max(contours, key=cv2.contourArea)
    grid_x, grid_y, grid_w, grid_h = cv2.boundingRect(largest)

    print(f"Grid boundary: ({grid_x}, {grid_y}) size {grid_w}×{grid_h}")
    print(f"Aspect ratio: {grid_w / grid_h:.3f} (should be ~1.0)")
    print()

    # Calculate cell size
    cell_w = grid_w / 9
    cell_h = grid_h / 9

    print(f"Cell size: {cell_w:.1f}×{cell_h:.1f} pixels")
    print()

    # Extract and OCR each cell
    print("Extracting cells...")
    grid = [[None for _ in range(9)] for _ in range(9)]

    # Use PaddleOCR (we know it works)
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(lang="en", use_angle_cls=False, use_gpu=False)

    cells_processed = 0
    for row in range(9):
        for col in range(9):
            # Calculate cell position with margin to avoid grid lines
            margin = 8  # pixels to crop from edges
            x1 = int(grid_x + col * cell_w + margin)
            y1 = int(grid_y + row * cell_h + margin)
            x2 = int(grid_x + (col + 1) * cell_w - margin)
            y2 = int(grid_y + (row + 1) * cell_h - margin)

            # Extract cell
            cell = img[y1:y2, x1:x2].copy()

            # Preprocess for OCR
            cell_gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)

            # Check if cell is mostly empty (white)
            mean_val = np.mean(cell_gray)
            if mean_val > 250:  # Very white = empty
                grid[row][col] = None
                continue

            # Otsu's thresholding for clean binary image
            _, cell_binary = cv2.threshold(cell_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Invert so digit is black on white (Tesseract prefers this)
            cell_binary = cv2.bitwise_not(cell_binary)

            # Remove small noise
            kernel = np.ones((2, 2), np.uint8)
            cell_binary = cv2.morphologyEx(cell_binary, cv2.MORPH_OPEN, kernel)

            # Scale up for better OCR (Tesseract works better with larger text)
            scale = 4
            cell_large = cv2.resize(
                cell_binary, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )

            # Add white border (helps Tesseract)
            bordered = cv2.copyMakeBorder(
                cell_large, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255
            )

            # OCR with Tesseract
            try:
                text = pytesseract.image_to_string(bordered, config=config)
                text = text.strip()

                # Extract digit
                digit = "".join(c for c in text if c.isdigit())

                if digit:
                    grid[row][col] = int(digit[0])
                    cells_processed += 1
                    if cells_processed <= 10:  # Show first 10
                        print(f"  [{row},{col}]: '{digit[0]}'")
                else:
                    grid[row][col] = None

            except Exception as e:
                print(f"  [{row},{col}]: Error - {str(e)[:40]}")
                grid[row][col] = None

    if cells_processed > 10:
        print(f"  ... ({cells_processed} total digits)")
    print()

    return grid


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
    """Extract Sudoku with perfect accuracy."""
    print("=" * 70)
    print("CHUK Vision - Perfect Sudoku Extraction (100% Goal)")
    print("=" * 70)
    print()

    sudoku_path = Path(__file__).parent.parent / "assets" / "sudoku.jpeg"
    print(f"📸 Loading: {sudoku_path.name}")
    print()

    grid = await extract_sudoku_perfect(str(sudoku_path))

    if grid:
        filled = sum(1 for row in grid for cell in row if cell is not None)
        print(f"📊 Extracted {filled} digits")
        print()

        display_grid(grid, "🎯 Extracted Grid")

        # Ground truth for verification
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

        errors = []
        for r in range(9):
            for c in range(9):
                if ground_truth[r][c] != grid[r][c]:
                    errors.append((r, c, ground_truth[r][c], grid[r][c]))

        accuracy = (81 - len(errors)) / 81 * 100

        print(f"✅ Accuracy: {accuracy:.1f}% ({81 - len(errors)}/81 correct)")
        print()

        if errors:
            print(f"❌ Errors: {len(errors)}")
            for r, c, expected, got in errors[:20]:  # Show first 20
                print(f"   [{r},{c}]: Expected {expected}, Got {got}")
            if len(errors) > 20:
                print(f"   ... and {len(errors) - 20} more")
        else:
            print("🎉 PERFECT! 100% accuracy achieved!")


if __name__ == "__main__":
    asyncio.run(main())

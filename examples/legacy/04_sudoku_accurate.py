#!/usr/bin/env python3
"""Accurate Sudoku extraction with proper grid boundary detection.

This example demonstrates:
1. Finding the actual Sudoku grid boundary (largest square contour)
2. Extracting and warping just the grid region
3. Properly dividing into 9x9 cells
4. OCR with preprocessing for better digit recognition
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2

from chuk_mcp_vision.tools.common import save_image_as_artifact, load_image_from_artifact
from chuk_mcp_vision.tools.extract.region import extract_region
from chuk_mcp_vision.tools.ocr.tool import ocr_region
from chuk_mcp_vision.models import OCRBackend


def find_sudoku_grid_boundary(img):
    """Find the Sudoku grid boundary using contour detection.

    Returns:
        Bounding box (x, y, width, height) of the grid, or None if not found
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply adaptive threshold to get binary image
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest rectangular contour (likely the grid)
    max_area = 0
    best_box = None

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area and area > 10000:  # Minimum size threshold
            # Approximate to polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            # Check if it's roughly rectangular (4 corners)
            if len(approx) >= 4:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h)

                # Sudoku grid should be roughly square
                if 0.8 < aspect_ratio < 1.2:
                    max_area = area
                    best_box = (x, y, w, h)

    return best_box


def preprocess_cell_for_ocr(cell_img):
    """Preprocess a cell image for better OCR accuracy.

    Args:
        cell_img: Cell image (BGR)

    Returns:
        Preprocessed image ready for OCR
    """
    # Resize to larger size for better OCR (PaddleOCR works better with larger text)
    # Target at least 100px for better recognition
    h, w = cell_img.shape[:2]
    if h < 100:
        scale = 100 / h
        cell_img = cv2.resize(cell_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Just return the resized image - let PaddleOCR handle the rest
    # (too much preprocessing can confuse the model)
    return cell_img


async def main():
    """Process Sudoku with accurate grid detection."""
    print("=" * 70)
    print("CHUK Vision - Accurate Sudoku Extraction")
    print("=" * 70)
    print()

    # Load Sudoku image
    sudoku_path = Path(__file__).parent.parent / "assets" / "sudoku.jpeg"
    print(f"📸 Loading: {sudoku_path.name}")

    img = await load_image_from_artifact(str(sudoku_path))
    height, width = img.shape[:2]
    print(f"   Original size: {width}×{height} pixels")
    print()

    # Step 1: Find grid boundary
    print("Step 1: Finding Sudoku grid boundary...")
    grid_box = find_sudoku_grid_boundary(img)

    if not grid_box:
        print("❌ Could not find Sudoku grid!")
        return

    x, y, w, h = grid_box
    print("✅ Grid found:")
    print(f"   Position: ({x}, {y})")
    print(f"   Size: {w}×{h} pixels")
    print(f"   Aspect ratio: {w / h:.2f} (should be ~1.0 for square)")
    print()

    # Step 2: Extract grid region
    print("Step 2: Extracting grid region...")
    grid_img = img[y : y + h, x : x + w]
    grid_artifact = await save_image_as_artifact(
        grid_img,
        suffix=".jpg",
        category="grid",
        metadata={"source": "grid_extraction", "grid_box": {"x": x, "y": y, "w": w, "h": h}},
    )
    print(f"✅ Grid extracted: {grid_artifact}")
    print(f"   Grid size: {w}×{h} pixels")
    print()

    # Step 3: Divide into 9×9 cells
    print("Step 3: Dividing into 9×9 cells...")
    cell_width = w / 9
    cell_height = h / 9
    print(f"   Cell size: {cell_width:.1f}×{cell_height:.1f} pixels")
    print()

    # Step 4: Extract and OCR all cells
    print("Step 4: Extracting and OCR all 81 cells...")
    print()

    grid_digits = []
    cells_processed = 0
    cells_with_digits = 0

    for row in range(9):
        row_digits = []
        for col in range(9):
            # Extract cell
            cell_x = col * cell_width
            cell_y = row * cell_height

            cell_result = await extract_region(
                grid_artifact, x=cell_x, y=cell_y, width=cell_width, height=cell_height
            )

            # Load cell and preprocess
            cell_img = await load_image_from_artifact(cell_result.output_image_uri)
            processed_cell = preprocess_cell_for_ocr(cell_img)

            # Save preprocessed cell
            processed_artifact = await save_image_as_artifact(
                processed_cell,
                suffix=".jpg",
                category="processed_cells",
                metadata={"row": row, "col": col},
            )

            # OCR
            try:
                ocr_result = await ocr_region(processed_artifact, backend=OCRBackend.PADDLEOCR)
                text = ocr_result.full_text.strip() if ocr_result.full_text else ""

                # Extract just digits
                digit = "".join(c for c in text if c.isdigit())

                # Take only first digit if multiple detected
                if len(digit) > 1:
                    digit = digit[0]

                if digit:
                    cells_with_digits += 1
                    row_digits.append(digit)
                    if cells_processed < 20:  # Show first 20 for progress
                        print(
                            f"   [{row},{col}]: '{digit}' (conf={ocr_result.mean_confidence:.2f})"
                        )
                else:
                    row_digits.append(".")

            except Exception as e:
                print(f"   [{row},{col}]: OCR error - {str(e)[:50]}")
                row_digits.append(".")

            cells_processed += 1

        grid_digits.append(row_digits)

    if cells_processed > 20:
        print(f"   ... processed {cells_processed} cells total")
    print()

    # Step 5: Display results
    print("=" * 70)
    print("✅ Sudoku Extraction Complete!")
    print("=" * 70)
    print()
    print("📊 Statistics:")
    print(f"   Cells processed: {cells_processed}/81")
    print(f"   Digits detected: {cells_with_digits}")
    print(f"   Empty cells: {81 - cells_with_digits}")
    print(f"   Detection rate: {cells_with_digits / 81 * 100:.1f}%")
    print()

    print("🔢 Extracted Sudoku Grid:")
    print()
    print("   ", "┌" + "───┬" * 8 + "───┐")
    for i, row in enumerate(grid_digits):
        # Print row
        print("   ", "│", end="")
        for j, digit in enumerate(row):
            print(f" {digit} ", end="")
            if j < 8:
                print("│" if (j + 1) % 3 == 0 else "│", end="")
        print("│")

        # Print separator
        if i < 8:
            if (i + 1) % 3 == 0:
                print("   ", "├" + "───┼" * 8 + "───┤")
            else:
                print("   ", "├" + "───┼" * 8 + "───┤")
    print("   ", "└" + "───┴" * 8 + "───┘")
    print()

    print("Next steps:")
    print("  1. Feed this grid to chuk-mcp-solver")
    print("  2. Compare with ground truth to verify accuracy")
    print("  3. Improve preprocessing if detection rate < 95%")


if __name__ == "__main__":
    asyncio.run(main())

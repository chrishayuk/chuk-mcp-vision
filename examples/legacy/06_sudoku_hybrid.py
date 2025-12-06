#!/usr/bin/env python3
"""Hybrid Sudoku extraction - combines Vision LLM + Traditional OCR.

This example demonstrates merging results from multiple approaches:
1. Vision LLM (GPT-4V/Claude) - High accuracy, sees whole grid
2. Traditional OCR (PaddleOCR) - Independent verification
3. Merge strategy - Use confidence voting to get best result

This hybrid approach can achieve higher reliability than either method alone.
"""

import asyncio
import sys
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2

from chuk_mcp_vision.tools.common import save_image_as_artifact, load_image_from_artifact
from chuk_mcp_vision.tools.extract.region import extract_region
from chuk_mcp_vision.tools.ocr.tool import ocr_region
from chuk_mcp_vision.models import OCRBackend


# Import vision LLM function from example 05
sys.path.insert(0, str(Path(__file__).parent))
from sudoku_vision_llm_05 import extract_sudoku_with_vision_llm


async def extract_with_traditional_ocr(image_path: str):
    """Extract Sudoku using traditional OCR pipeline.

    Returns:
        Tuple of (grid, confidence_grid) where confidence is 0-1 per cell
    """
    # Load image
    img = await load_image_from_artifact(image_path)

    # Find grid boundary
    grid_box = find_sudoku_grid_boundary(img)
    if not grid_box:
        return None, None

    x, y, w, h = grid_box
    grid_img = img[y : y + h, x : x + w]

    # Save as artifact
    grid_artifact = await save_image_as_artifact(grid_img, suffix=".jpg", category="grid")

    # Divide into 9x9 cells
    cell_width = w / 9
    cell_height = h / 9

    grid = [[None for _ in range(9)] for _ in range(9)]
    confidence_grid = [[0.0 for _ in range(9)] for _ in range(9)]

    for row in range(9):
        for col in range(9):
            # Extract cell
            cell_x = col * cell_width
            cell_y = row * cell_height

            cell_result = await extract_region(
                grid_artifact, x=cell_x, y=cell_y, width=cell_width, height=cell_height
            )

            # Load and resize cell for better OCR
            cell_img = await load_image_from_artifact(cell_result.output_image_uri)
            h_cell, w_cell = cell_img.shape[:2]
            if h_cell < 100:
                scale = 100 / h_cell
                cell_img = cv2.resize(
                    cell_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
                )

            # Save resized
            resized_artifact = await save_image_as_artifact(
                cell_img, suffix=".jpg", category="cells"
            )

            # OCR
            try:
                ocr_result = await ocr_region(resized_artifact, backend=OCRBackend.PADDLEOCR)
                text = ocr_result.full_text.strip() if ocr_result.full_text else ""

                # Extract just digits
                digit = "".join(c for c in text if c.isdigit())

                if digit and len(digit) > 0:
                    grid[row][col] = int(digit[0])
                    confidence_grid[row][col] = ocr_result.mean_confidence

            except Exception:
                pass  # Keep as None

    return grid, confidence_grid


def find_sudoku_grid_boundary(img):
    """Find the Sudoku grid boundary using contour detection."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    best_box = None

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area and area > 10000:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            if len(approx) >= 4:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h)

                if 0.8 < aspect_ratio < 1.2:
                    max_area = area
                    best_box = (x, y, w, h)

    return best_box


def merge_grids(llm_grid, ocr_grid, ocr_confidence_grid):
    """Merge results from Vision LLM and traditional OCR.

    Strategy:
    1. If both agree -> use that value (high confidence)
    2. If LLM has value and OCR is None -> use LLM (LLM is more reliable)
    3. If OCR has high confidence (>0.9) and LLM disagrees -> flag for review
    4. If OCR has value and LLM is None -> use OCR with caution

    Returns:
        Tuple of (merged_grid, confidence_grid, disagreements)
    """
    merged = [[None for _ in range(9)] for _ in range(9)]
    confidence = [[0.0 for _ in range(9)] for _ in range(9)]
    disagreements = []

    for row in range(9):
        for col in range(9):
            llm_val = llm_grid[row][col]
            ocr_val = ocr_grid[row][col] if ocr_grid else None
            ocr_conf = ocr_confidence_grid[row][col] if ocr_confidence_grid else 0.0

            # Both agree
            if llm_val is not None and ocr_val is not None and llm_val == ocr_val:
                merged[row][col] = llm_val
                confidence[row][col] = 0.99  # Very high confidence

            # Both have values but disagree
            elif llm_val is not None and ocr_val is not None and llm_val != ocr_val:
                # LLM is generally more reliable, but if OCR has very high confidence, flag it
                if ocr_conf > 0.95:
                    disagreements.append(
                        {
                            "position": (row, col),
                            "llm": llm_val,
                            "ocr": ocr_val,
                            "ocr_confidence": ocr_conf,
                        }
                    )
                    # Use LLM but lower confidence
                    merged[row][col] = llm_val
                    confidence[row][col] = 0.85
                else:
                    # Trust LLM
                    merged[row][col] = llm_val
                    confidence[row][col] = 0.90

            # Only LLM has value
            elif llm_val is not None:
                merged[row][col] = llm_val
                confidence[row][col] = 0.95  # High confidence in LLM

            # Only OCR has value
            elif ocr_val is not None:
                merged[row][col] = ocr_val
                confidence[row][col] = ocr_conf * 0.7  # Discount OCR-only results

            # Both None
            else:
                merged[row][col] = None
                confidence[row][col] = 0.99  # High confidence it's empty

    return merged, confidence, disagreements


def display_grid(grid, confidence_grid=None, title="Sudoku Grid"):
    """Display grid with optional confidence scores."""
    print(f"\n{title}:")
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

    if confidence_grid:
        avg_conf = sum(sum(row) for row in confidence_grid) / 81
        print(f"\n   Average confidence: {avg_conf:.2%}")


async def main():
    """Run hybrid Sudoku extraction."""
    print("=" * 70)
    print("CHUK Vision - Hybrid Sudoku Extraction")
    print("=" * 70)
    print()
    print("Strategy: Combine Vision LLM + Traditional OCR for best results")
    print()

    sudoku_path = Path(__file__).parent.parent / "assets" / "sudoku.jpeg"
    print(f"📸 Loading: {sudoku_path.name}")
    print()

    # Check for API key
    import os

    has_llm = bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"))

    if not has_llm:
        print("⚠️  No Vision LLM API key found - running OCR only")
        print()

    # Method 1: Vision LLM
    llm_grid = None
    if has_llm:
        print("Step 1: Extracting with Vision LLM...")
        try:
            provider = "openai" if os.environ.get("OPENAI_API_KEY") else "anthropic"
            llm_grid = await extract_sudoku_with_vision_llm(str(sudoku_path), provider)

            filled_llm = sum(1 for row in llm_grid for cell in row if cell is not None)
            print(f"✅ Vision LLM: {filled_llm} digits detected")
        except Exception as e:
            print(f"❌ Vision LLM failed: {e}")
            llm_grid = [[None for _ in range(9)] for _ in range(9)]
    else:
        llm_grid = [[None for _ in range(9)] for _ in range(9)]

    print()

    # Method 2: Traditional OCR
    print("Step 2: Extracting with Traditional OCR...")
    ocr_grid, ocr_conf = await extract_with_traditional_ocr(str(sudoku_path))

    if ocr_grid:
        filled_ocr = sum(1 for row in ocr_grid for cell in row if cell is not None)
        print(f"✅ Traditional OCR: {filled_ocr} digits detected")
    else:
        print("❌ Traditional OCR failed")
        ocr_grid = [[None for _ in range(9)] for _ in range(9)]
        ocr_conf = [[0.0 for _ in range(9)] for _ in range(9)]

    print()

    # Method 3: Merge
    print("Step 3: Merging results...")
    merged_grid, merged_conf, disagreements = merge_grids(llm_grid, ocr_grid, ocr_conf)

    filled_merged = sum(1 for row in merged_grid for cell in row if cell is not None)
    print(f"✅ Merged: {filled_merged} digits")
    print()

    # Display results
    if has_llm:
        display_grid(llm_grid, title="Vision LLM Result")
    display_grid(ocr_grid, ocr_conf, title="Traditional OCR Result")
    display_grid(merged_grid, merged_conf, title="🎯 MERGED Result (Best of Both)")

    # Show disagreements
    if disagreements:
        print()
        print(f"⚠️  Found {len(disagreements)} disagreements:")
        for d in disagreements:
            row, col = d["position"]
            print(
                f"   Cell [{row},{col}]: LLM={d['llm']}, OCR={d['ocr']} (conf={d['ocr_confidence']:.2f})"
            )
        print()
        print("   → Using LLM values (generally more reliable)")

    print()
    print("=" * 70)
    print("✅ Hybrid Extraction Complete!")
    print("=" * 70)
    print()
    print("Benefits of hybrid approach:")
    print("  ✅ Cross-validation between two independent methods")
    print("  ✅ Higher confidence when both agree")
    print("  ✅ Falls back gracefully if one method fails")
    print("  ✅ Flags suspicious cells for review")
    print("  ✅ Can work without API key (OCR-only mode)")


if __name__ == "__main__":
    asyncio.run(main())

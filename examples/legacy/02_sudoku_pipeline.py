#!/usr/bin/env python3
"""Sudoku pipeline example - full workflow from image to digit grid.

This example demonstrates:
1. Analyzing a grid image
2. Detecting grid lines
3. Extracting individual cells
4. OCR on each cell
5. Assembling into 9x9 digit array
6. All using chuk-artifacts for intermediate storage

This is the full vision pipeline that would feed into chuk-mcp-solver.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np

from chuk_mcp_vision.tools.analyze.tool import analyze
from chuk_mcp_vision.tools.common import save_image_as_artifact
from chuk_mcp_vision.tools.detect.lines import detect_lines
from chuk_mcp_vision.tools.extract.region import extract_region
from chuk_mcp_vision.tools.ocr.tool import ocr_region
from chuk_mcp_vision.models import OCRBackend


async def main():
    """Run the Sudoku pipeline example."""
    print("=" * 70)
    print("CHUK Vision - Sudoku Pipeline Example")
    print("=" * 70)
    print()
    print("This example demonstrates the full vision pipeline:")
    print("  Image → Analyze → Detect → Extract → OCR → Digit Grid")
    print()

    # Step 1: Create test Sudoku image with some digits
    print("Step 1: Creating test Sudoku grid with digits...")
    sudoku_img = create_sudoku_test_image()
    artifact_id = await save_image_as_artifact(
        sudoku_img, category="input", metadata={"source": "test", "type": "sudoku"}
    )
    print("✅ Sudoku image created and saved as artifact")
    print(f"   Artifact ID: {artifact_id}")
    print()

    # Step 2: Analyze the image
    print("Step 2: Analyzing image to detect structure...")
    analysis = await analyze(artifact_id, use_vision_model=False)
    print("✅ Analysis complete:")
    print(f"   Scene type: {analysis.scene_type}")
    print(f"   Structure: {analysis.structure}")
    print(f"   Content hints: {', '.join([str(h) for h in analysis.content_hints])}")
    print(f"   Suggested tools: {', '.join(analysis.suggested_tools)}")
    print(f"   Confidence: {analysis.confidence:.2f}")
    print()

    # Step 3: Detect grid lines
    print("Step 3: Detecting grid lines...")
    lines_result = await detect_lines(artifact_id, min_line_length=100, threshold=150)
    print("✅ Grid lines detected:")
    print(f"   Horizontal lines: {lines_result.horizontal_lines}")
    print(f"   Vertical lines: {lines_result.vertical_lines}")
    print()

    # Step 4: Calculate cell positions from lines
    print("Step 4: Calculating cell positions...")
    cells = calculate_cell_positions(lines_result, 9, 9)
    print(f"✅ Cell positions calculated: {len(cells)} cells")
    print(
        f"   Example cell 0: x={cells[0]['x']:.0f}, y={cells[0]['y']:.0f}, "
        f"w={cells[0]['width']:.0f}, h={cells[0]['height']:.0f}"
    )
    print()

    # Step 5: Extract first few cells as examples
    print("Step 5: Extracting cells (demo: first 9 cells)...")
    cell_artifacts = []
    for i in range(min(9, len(cells))):
        cell = cells[i]
        cell_artifact = await extract_region(
            artifact_id, x=cell["x"], y=cell["y"], width=cell["width"], height=cell["height"]
        )
        cell_artifacts.append(cell_artifact.output_image_uri)
        if i < 3:  # Show first 3
            print(f"   Cell {i}: extracted → {cell_artifact.output_image_uri[:16]}...")
    if len(cell_artifacts) > 3:
        print(f"   ... and {len(cell_artifacts) - 3} more cells")
    print()

    # Step 6: OCR demo (would normally do all 81)
    print("Step 6: OCR demonstration (first 3 cells)...")
    print("   Note: Test image has no actual digits, OCR will return empty")
    for i, cell_id in enumerate(cell_artifacts[:3]):
        try:
            ocr_result = await ocr_region(cell_id, backend=OCRBackend.TESSERACT)
            digit = ocr_result.full_text.strip() if ocr_result.full_text else ""
            confidence = ocr_result.mean_confidence
            print(f"   Cell {i}: text='{digit}' (confidence={confidence:.2f})")
        except Exception as e:
            print(f"   Cell {i}: OCR failed - {str(e)[:50]}")
    print()

    # Step 7: Show what the full pipeline would look like
    print("Step 7: Full pipeline overview:")
    print()
    print("   In a real scenario with an actual Sudoku image:")
    print()
    print("   ```python")
    print("   # 1. Extract all 81 cells")
    print("   cell_ids = []")
    print("   for cell in cells:")
    print("       result = await extract_region(image_id, **cell)")
    print("       cell_ids.append(result.output_image_uri)")
    print()
    print("   # 2. OCR all cells")
    print("   digits = []")
    print("   for cell_id in cell_ids:")
    print("       ocr = await ocr_region(cell_id, backend='paddleocr')")
    print("       digit = parse_digit(ocr.full_text)  # '5' or None")
    print("       digits.append(digit)")
    print()
    print("   # 3. Reshape to 9x9 grid")
    print("   grid = np.array(digits).reshape(9, 9)")
    print()
    print("   # 4. Send to chuk-mcp-solver")
    print("   solution = await solver.solve_sudoku(grid)")
    print("   ```")
    print()

    print("=" * 70)
    print("✅ Pipeline demonstration complete!")
    print("=" * 70)
    print()
    print("Artifacts created in this pipeline:")
    print("  • 1 input image (Sudoku grid)")
    print(f"  • {len(cell_artifacts)} extracted cells")
    print("  • All stored as SESSION-scoped BLOB artifacts")
    print("  • Auto-cleanup when session ends")
    print()
    print("Next steps:")
    print("  1. Test with real Sudoku image (photo or scan)")
    print("  2. Integrate with chuk-mcp-solver for solving")
    print("  3. Create orchestration playbook for chuk-ai-planner")


def create_sudoku_test_image(size: int = 900) -> np.ndarray:
    """Create a test Sudoku grid image.

    Args:
        size: Image size (square)

    Returns:
        OpenCV image with 9x9 grid
    """
    # Create white image
    img = np.ones((size, size, 3), dtype=np.uint8) * 255

    cell_size = size // 9

    # Draw grid lines
    for i in range(10):
        pos = i * cell_size
        thickness = 3 if i % 3 == 0 else 1  # Thicker lines for 3x3 boxes

        # Horizontal lines
        cv2.line(img, (0, pos), (size, pos), (0, 0, 0), thickness)
        # Vertical lines
        cv2.line(img, (pos, 0), (pos, size), (0, 0, 0), thickness)

    # Outer border
    cv2.rectangle(img, (0, 0), (size - 1, size - 1), (0, 0, 0), 4)

    # Add a few sample digits (for visual reference)
    font = cv2.FONT_HERSHEY_SIMPLEX
    sample_positions = [(0, 0, "5"), (1, 1, "3"), (2, 2, "7")]

    for row, col, digit in sample_positions:
        x = col * cell_size + cell_size // 3
        y = row * cell_size + 2 * cell_size // 3
        cv2.putText(img, digit, (x, y), font, 2, (0, 0, 0), 2)

    return img


def calculate_cell_positions(lines_result, rows: int = 9, cols: int = 9):
    """Calculate cell positions from detected grid lines.

    Args:
        lines_result: Result from detect_lines
        rows: Number of rows
        cols: Number of columns

    Returns:
        List of dicts with x, y, width, height for each cell
    """
    # Simplified: Use image dimensions to calculate grid
    # In production, would use actual detected lines

    width = lines_result.image_width
    height = lines_result.image_height

    cell_width = width / cols
    cell_height = height / rows

    cells = []
    for row in range(rows):
        for col in range(cols):
            cells.append(
                {
                    "x": col * cell_width,
                    "y": row * cell_height,
                    "width": cell_width,
                    "height": cell_height,
                }
            )

    return cells


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""Real Sudoku image processing example.

This example demonstrates processing an actual Sudoku photo from the assets folder.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chuk_mcp_vision.tools.analyze.tool import analyze
from chuk_mcp_vision.tools.common import save_image_as_artifact, load_image_from_artifact
from chuk_mcp_vision.tools.detect.lines import detect_lines
from chuk_mcp_vision.tools.extract.region import extract_region
from chuk_mcp_vision.tools.ocr.tool import ocr_region
from chuk_mcp_vision.models import OCRBackend


async def main():
    """Process a real Sudoku image."""
    print("=" * 70)
    print("CHUK Vision - Real Sudoku Image Processing")
    print("=" * 70)
    print()

    # Path to real Sudoku image
    sudoku_path = Path(__file__).parent.parent / "assets" / "sudoku.jpeg"
    print(f"📸 Loading Sudoku image: {sudoku_path.name}")
    print(f"   Path: {sudoku_path}")
    print()

    if not sudoku_path.exists():
        print("❌ Error: sudoku.jpeg not found in assets folder")
        return

    # Step 1: Analyze the image
    print("Step 1: Analyzing image structure...")
    analysis = await analyze(str(sudoku_path), use_vision_model=False)
    print("✅ Analysis complete:")
    print(f"   Scene type: {analysis.scene_type}")
    print(f"   Structure: {analysis.structure}")
    print(f"   Content hints: {', '.join([str(h) for h in analysis.content_hints])}")
    print(f"   Suggested tools: {', '.join(analysis.suggested_tools)}")
    print(f"   Confidence: {analysis.confidence:.2f}")
    print(f"   Description: {analysis.description}")
    print()

    # Step 2: Load and save as artifact
    print("Step 2: Loading image and saving as artifact...")
    img = await load_image_from_artifact(str(sudoku_path))
    height, width = img.shape[:2]
    print(f"   Image dimensions: {width}x{height} pixels")

    artifact_id = await save_image_as_artifact(
        img,
        suffix=".jpg",
        category="input",
        metadata={"source": "real_sudoku", "original_path": str(sudoku_path)},
    )
    print(f"   ✅ Saved as artifact: {artifact_id}")
    print()

    # Step 3: Detect grid lines
    print("Step 3: Detecting grid lines...")
    lines_result = await detect_lines(artifact_id, min_line_length=100, threshold=150)
    print("✅ Grid lines detected:")
    print(f"   Horizontal lines: {lines_result.horizontal_lines}")
    print(f"   Vertical lines: {lines_result.vertical_lines}")
    print(f"   Total lines: {len(lines_result.lines)}")
    print(f"   Image size: {lines_result.image_width}x{lines_result.image_height}")
    print()

    # Step 4: Show first few detected lines
    if lines_result.lines:
        print("Step 4: Sample detected lines:")
        for i, line in enumerate(lines_result.lines[:10]):
            orientation = (
                "H"
                if (line.angle < 10 or line.angle > 170)
                else "V"
                if (80 < line.angle < 100)
                else "D"
            )
            print(
                f"   Line {i + 1} [{orientation}]: ({line.x1:.0f},{line.y1:.0f}) → ({line.x2:.0f},{line.y2:.0f}), "
                f"angle={line.angle:.1f}°, length={line.length:.0f}px"
            )
        if len(lines_result.lines) > 10:
            print(f"   ... and {len(lines_result.lines) - 10} more lines")
        print()

    # Step 5: Calculate cell positions (if 9x9 grid detected)
    print("Step 5: Calculating cell positions...")

    # Use image dimensions to calculate a 9x9 grid
    cell_width = lines_result.image_width / 9
    cell_height = lines_result.image_height / 9

    cells = []
    for row in range(9):
        for col in range(9):
            cells.append(
                {
                    "x": col * cell_width,
                    "y": row * cell_height,
                    "width": cell_width,
                    "height": cell_height,
                }
            )

    print(f"✅ Calculated {len(cells)} cell positions (9x9 grid)")
    print(f"   Cell size: ~{cell_width:.0f}x{cell_height:.0f} pixels")
    print()

    # Step 6: Extract a few sample cells
    print("Step 6: Extracting sample cells (first 9 cells from top row)...")
    sample_cells = []
    for i in range(min(9, len(cells))):
        cell = cells[i]
        cell_artifact = await extract_region(
            artifact_id, x=cell["x"], y=cell["y"], width=cell["width"], height=cell["height"]
        )
        sample_cells.append(cell_artifact.output_image_uri)
        if i < 5:
            print(f"   Cell {i} (row 0, col {i}): {cell_artifact.output_image_uri}")
    if len(sample_cells) > 5:
        print(f"   ... and {len(sample_cells) - 5} more cells")
    print()

    # Step 7: Try OCR on first few cells
    print("Step 7: OCR demonstration (first 5 cells)...")
    print("   Note: This will attempt to read digits from the cells")
    print()

    for i, cell_id in enumerate(sample_cells[:5]):
        try:
            ocr_result = await ocr_region(cell_id, backend=OCRBackend.PADDLEOCR)
            digit = ocr_result.full_text.strip() if ocr_result.full_text else ""
            confidence = ocr_result.mean_confidence

            # Clean up the OCR result (extract just digits)
            digit_clean = "".join(c for c in digit if c.isdigit())

            print(f"   Cell {i}: '{digit_clean}' (confidence={confidence:.2f})")
        except Exception as e:
            error_msg = str(e)[:60]
            print(f"   Cell {i}: OCR error - {error_msg}")
    print()

    print("=" * 70)
    print("✅ Real Sudoku processing complete!")
    print("=" * 70)
    print()
    print("Summary:")
    print(f"  • Analyzed image structure: {analysis.structure}")
    print(f"  • Detected {len(lines_result.lines)} lines")
    print(f"  • Extracted {len(sample_cells)} sample cells")
    print("  • All artifacts stored in chuk-artifacts (session-scoped)")
    print()
    print("Next steps:")
    print("  1. If OCR worked, extract all 81 cells and build digit grid")
    print("  2. Apply perspective correction if image is skewed")
    print("  3. Feed digit grid to chuk-mcp-solver")
    print("  4. Create orchestration playbook for full automation")


if __name__ == "__main__":
    asyncio.run(main())

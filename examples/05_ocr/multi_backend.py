#!/usr/bin/env python3
"""Grid tools + Multi-backend OCR demo - Higher accuracy through ensemble voting.

This demo shows how combining:
1. Generic grid tools (detect_grid, extract_grid_cells)
2. Multi-backend OCR (ensemble voting)

Can achieve higher accuracy than any single method.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from chuk_mcp_vision.tools.grid.detect import detect_grid
from chuk_mcp_vision.tools.grid.extract_cells import extract_grid_cells
from chuk_mcp_vision.tools.ocr.multi_backend import ocr_multi_backend


# Ground truth for Sudoku
GROUND_TRUTH = [
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


async def extract_with_multi_backend():
    """Extract Sudoku using grid tools + multi-backend OCR."""
    print("=" * 70)
    print("Grid Tools + Multi-Backend OCR Demo")
    print("=" * 70)
    print()
    print("Strategy: Grid detection + Ensemble OCR voting")
    print()

    sudoku_path = Path(__file__).parent.parent.parent / "assets" / "sudoku.jpeg"
    print(f"📸 Image: {sudoku_path.name}")
    print()

    # Step 1: Detect grid
    print("Step 1: Detect Grid Structure")
    print("-" * 70)
    grid = await detect_grid(str(sudoku_path))
    print(f"✅ Grid: {grid.rows}×{grid.cols} cells")
    print(
        f"   Boundary: ({grid.boundary.x:.0f}, {grid.boundary.y:.0f}) "
        f"{grid.boundary.width:.0f}×{grid.boundary.height:.0f}px"
    )
    print(f"   Cell size: {grid.cell_width:.1f}×{grid.cell_height:.1f}px")
    print()

    # Step 2: Extract cells
    print("Step 2: Extract Grid Cells")
    print("-" * 70)
    cells = await extract_grid_cells(str(sudoku_path), grid, margin=8, preprocess=True)
    print(f"✅ Extracted {cells.total_cells} cells")
    print()

    # Step 3: Classify with multi-backend OCR
    print("Step 3: Multi-Backend OCR Classification")
    print("-" * 70)
    print("Using ensemble voting across backends...")
    print()

    grid_values = [[None for _ in range(9)] for _ in range(9)]
    confidences = [[0.0 for _ in range(9)] for _ in range(9)]

    single_backend_grid = [[None for _ in range(9)] for _ in range(9)]
    single_backend_conf = [[0.0 for _ in range(9)] for _ in range(9)]

    # Track improvements
    corrections = []

    for cell_loc in cells.cells:
        # First try single backend (PaddleOCR) for comparison
        from chuk_mcp_vision.tools.ocr.tool import ocr_region
        from chuk_mcp_vision.models import OCRBackend

        single_result = await ocr_region(cell_loc.image_uri, backend=OCRBackend.PADDLEOCR)
        single_text = single_result.full_text.strip()
        single_digit = (
            int(single_text) if single_text.isdigit() and 1 <= int(single_text) <= 9 else None
        )
        single_backend_grid[cell_loc.row][cell_loc.col] = single_digit
        single_backend_conf[cell_loc.row][cell_loc.col] = single_result.mean_confidence

        # Now try multi-backend
        multi_result = await ocr_multi_backend(
            cell_loc.image_uri, backends=["paddleocr", "tesseract"]
        )

        # Extract digit from consensus
        consensus_text = multi_result.consensus_text.strip()
        digit = None
        if consensus_text:
            digits = "".join(c for c in consensus_text if c.isdigit())
            if digits and 1 <= int(digits[0]) <= 9:
                digit = int(digits[0])

        grid_values[cell_loc.row][cell_loc.col] = digit
        confidences[cell_loc.row][cell_loc.col] = multi_result.consensus_confidence

        # Track if multi-backend corrected single backend
        if single_digit != digit:
            corrections.append(
                {
                    "position": [cell_loc.row, cell_loc.col],
                    "single_backend": single_digit,
                    "multi_backend": digit,
                    "voting": multi_result.voting_details,
                }
            )

        # Show first few results
        if cell_loc.row == 0 and cell_loc.col < 5:
            print(
                f"   [{cell_loc.row},{cell_loc.col}]: "
                f"Single={single_digit}, Multi={digit} "
                f"(conf={multi_result.consensus_confidence:.2f}, "
                f"agree={multi_result.voting_details.get('agreement_rate', 0):.0%})"
            )

    print("   ...")
    print()

    # Display results
    print("=" * 70)
    print("Results Comparison")
    print("=" * 70)
    print()

    print("Single Backend (PaddleOCR only):")
    print()
    display_grid(single_backend_grid)
    single_accuracy = calculate_accuracy(single_backend_grid)
    print(f"Accuracy: {single_accuracy:.1f}%")
    print()

    print("Multi-Backend (PaddleOCR + Tesseract voting):")
    print()
    display_grid(grid_values)
    multi_accuracy = calculate_accuracy(grid_values)
    print(f"Accuracy: {multi_accuracy:.1f}%")
    print()

    # Show improvements
    if corrections:
        print(f"🔧 Corrections made by multi-backend: {len(corrections)}")
        for corr in corrections[:5]:
            pos = corr["position"]
            expected = GROUND_TRUTH[pos[0]][pos[1]]
            print(
                f"   [{pos[0]},{pos[1]}]: {corr['single_backend']} → {corr['multi_backend']} "
                f"(expected: {expected})"
            )
        if len(corrections) > 5:
            print(f"   ... and {len(corrections) - 5} more")
        print()

    # Final stats
    improvement = multi_accuracy - single_accuracy
    print(f"📊 Improvement: +{improvement:.1f}% accuracy")
    print(f"   Single backend: {single_accuracy:.1f}%")
    print(f"   Multi-backend: {multi_accuracy:.1f}%")
    print()

    # Show remaining errors
    errors = []
    for r in range(9):
        for c in range(9):
            if GROUND_TRUTH[r][c] != grid_values[r][c]:
                errors.append((r, c, GROUND_TRUTH[r][c], grid_values[r][c]))

    if errors:
        print(f"❌ Remaining errors: {len(errors)}")
        for r, c, expected, got in errors[:10]:
            print(f"   [{r},{c}]: Expected {expected}, Got {got}")
    else:
        print("🎉 PERFECT! 100% accuracy achieved!")

    print()
    print("=" * 70)
    print("Key Insights")
    print("=" * 70)
    print()
    print("✨ Multi-backend OCR uses ensemble voting:")
    print("   - Runs multiple OCR engines (PaddleOCR, Tesseract)")
    print("   - Uses majority vote for final answer")
    print("   - Weights by confidence scores")
    print()
    print(f"🎯 Result: {improvement:.1f}% accuracy improvement")
    print("   Different engines have different strengths")
    print("   Voting corrects individual engine errors")
    print()
    print("💡 For critical applications:")
    print("   - Add Vision LLM verification for remaining errors")
    print("   - Use domain validation (Sudoku constraints)")
    print("   - Expected final accuracy: 99%+")


def display_grid(grid):
    """Display Sudoku grid."""
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


def calculate_accuracy(grid):
    """Calculate accuracy against ground truth."""
    correct = 0
    for r in range(9):
        for c in range(9):
            if GROUND_TRUTH[r][c] == grid[r][c]:
                correct += 1
    return (correct / 81) * 100


if __name__ == "__main__":
    asyncio.run(extract_with_multi_backend())

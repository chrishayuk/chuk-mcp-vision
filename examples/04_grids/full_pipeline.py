#!/usr/bin/env python3
"""Demo of improved grid tools - generic and composable.

These tools work for ANY grid structure:
- Sudoku (9×9 digits)
- Spreadsheets (variable rows/cols, text)
- Crosswords (grids with text)
- Forms with checkboxes
- Calendars
- Periodic tables
- Any rectangular grid

The tools are:
1. detect_grid(uri) → GridStructure
2. extract_grid_cells(uri, grid) → List[CellLocation]
3. classify_cell_content(cell_uri, expected_type) → CellContent
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from chuk_mcp_vision.tools.grid.detect import detect_grid
from chuk_mcp_vision.tools.grid.extract_cells import extract_grid_cells
from chuk_mcp_vision.tools.grid.classify_cell import classify_cell_content


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


async def demo_grid_tools():
    """Demonstrate the grid tools on a Sudoku image."""
    print("=" * 70)
    print("Grid Tools Demo - Generic Tools for Any Grid")
    print("=" * 70)
    print()

    sudoku_path = Path(__file__).parent.parent.parent / "assets" / "sudoku.jpeg"
    print(f"📸 Image: {sudoku_path.name}")
    print()

    # Step 1: Detect grid structure
    print("Step 1: Detect Grid Structure")
    print("-" * 70)
    grid = await detect_grid(str(sudoku_path))

    print("✅ Grid detected:")
    print(f"   Size: {grid.rows}×{grid.cols} cells")
    print(
        f"   Boundary: ({grid.boundary.x:.0f}, {grid.boundary.y:.0f}) "
        f"{grid.boundary.width:.0f}×{grid.boundary.height:.0f}px"
    )
    print(f"   Cell size: {grid.cell_width:.1f}×{grid.cell_height:.1f}px")
    print(f"   Regular grid: {grid.is_regular}")
    print(f"   Confidence: {grid.confidence:.2%}")
    print(f"   Row dividers: {len(grid.row_dividers)} positions")
    print(f"   Col dividers: {len(grid.col_dividers)} positions")
    print()

    # Step 2: Extract all cells
    print("Step 2: Extract Individual Cells")
    print("-" * 70)
    cells_response = await extract_grid_cells(str(sudoku_path), grid, margin=8, preprocess=True)

    print(f"✅ Extracted {cells_response.total_cells} cells")
    print(f"   Grid size: {cells_response.grid_size[0]}×{cells_response.grid_size[1]}")
    print(f"   First cell URI: {cells_response.cells[0].image_uri}")
    print()

    # Step 3: Classify each cell
    print("Step 3: Classify Cell Contents")
    print("-" * 70)
    print("Classifying all cells (this may take a moment)...")
    print()

    grid_values = [[None for _ in range(9)] for _ in range(9)]
    confidences = [[0.0 for _ in range(9)] for _ in range(9)]

    for cell_loc in cells_response.cells:
        # Classify as digit (since we know it's Sudoku)
        content = await classify_cell_content(cell_loc.image_uri, expected_type="digit")

        grid_values[cell_loc.row][cell_loc.col] = content.value
        confidences[cell_loc.row][cell_loc.col] = content.confidence

        # Show first few classifications
        if cell_loc.row == 0 and cell_loc.col < 5:
            print(
                f"   [{cell_loc.row},{cell_loc.col}]: "
                f"{content.value if content.value else '.'} "
                f"({content.content_type.value}, conf={content.confidence:.2f})"
            )

    print("   ...")
    print()

    # Display extracted grid
    print("Step 4: Results")
    print("-" * 70)
    print()
    print("Extracted Grid:")
    print()
    display_grid(grid_values)

    # Validate
    errors = []
    for r in range(9):
        for c in range(9):
            if GROUND_TRUTH[r][c] != grid_values[r][c]:
                errors.append((r, c, GROUND_TRUTH[r][c], grid_values[r][c]))

    accuracy = (81 - len(errors)) / 81 * 100
    avg_conf = sum(sum(row) for row in confidences) / 81

    print(f"✅ Accuracy: {accuracy:.1f}% ({81 - len(errors)}/81 correct)")
    print(f"📊 Average confidence: {avg_conf:.2%}")
    print()

    if errors:
        print(f"❌ Errors: {len(errors)}")
        for r, c, expected, got in errors[:10]:
            print(f"   [{r},{c}]: Expected {expected}, Got {got}")
        if len(errors) > 10:
            print(f"   ... and {len(errors) - 10} more")
    else:
        print("🎉 PERFECT! 100% accuracy!")

    print()
    print("=" * 70)
    print("Key Insights:")
    print("=" * 70)
    print()
    print("✨ These are GENERIC grid tools, not Sudoku-specific:")
    print("   - detect_grid works for ANY rectangular grid")
    print("   - extract_grid_cells works for ANY grid size")
    print("   - classify_cell_content works for digits, text, checkboxes")
    print()
    print("🤖 An AI agent can now orchestrate these tools for:")
    print("   - Sudoku puzzles")
    print("   - Spreadsheet tables")
    print("   - Crossword puzzles")
    print("   - Form checkboxes")
    print("   - Calendars")
    print("   - Any grid-based layout")
    print()
    print("🎯 The tools are atomic and composable - perfect for AI orchestration!")


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


if __name__ == "__main__":
    asyncio.run(demo_grid_tools())

#!/usr/bin/env python3
"""Example: Rotation and Classification - Handle orientation-dependent content.

This example demonstrates rotation-based classification for content that
looks different at different angles (6 vs 9, b vs q, etc.).

Tools used:
- rotate_and_classify: Test multiple rotations and classify
- detect_loop_position: Detect loops in characters for 6/9 disambiguation
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from chuk_mcp_vision.tools.transform.rotate import rotate_and_classify, detect_loop_position
from chuk_mcp_vision.tools.grid import detect_grid, extract_grid_cells


async def demo_rotation_classification():
    """Demonstrate rotation-based classification."""
    print("=" * 70)
    print("Rotation and Classification Demo")
    print("=" * 70)
    print()
    print("Testing rotation for orientation-dependent characters...")
    print()

    # Load image
    image_path = Path(__file__).parent.parent.parent / "assets" / "sudoku.jpeg"
    if not image_path.exists():
        print(f"❌ Error: {image_path} not found")
        return

    print(f"📸 Image: {image_path.name}")
    print()

    # Extract a cell that might be ambiguous (looking for 6 or 9)
    print("Step 1: Extract Test Cell")
    print("-" * 70)

    grid = await detect_grid(str(image_path))
    cells = await extract_grid_cells(str(image_path), grid)

    # Find a cell with a digit (not empty)
    test_cell = None
    for cell in cells.cells:
        # Try row 0, col 0 - should be "7"
        if cell.row == 0 and cell.col == 0:
            test_cell = cell
            break

    if not test_cell:
        print("❌ No suitable test cell found")
        return

    print(f"✅ Testing cell [{test_cell.row},{test_cell.col}]")
    print(f"   Cell URI: {test_cell.image_uri}")
    print()

    # Step 2: Test rotation with classification
    print("Step 2: Rotate and Classify at Multiple Angles")
    print("-" * 70)

    result = await rotate_and_classify(
        test_cell.image_uri,
        angles=[0, 90, 180, 270],  # Test all 4 orientations
    )

    print("✅ Rotation test complete:")
    print(f"   Recommended angle: {result.recommended_angle}°")
    print(f"   Recommended value: '{result.recommended_value}'")
    print(f"   Confidence: {result.confidence:.2%}")
    print(f"   Disambiguation: {result.disambiguation}")
    print()

    print("Results by angle:")
    for angle_result in result.results_by_angle:
        print(
            f"   {angle_result.angle:>3}°: '{angle_result.classification_value}' "
            f"(confidence: {angle_result.confidence:.2%})"
        )
    print()

    # Step 3: Detect loop position (for 6/9 disambiguation)
    print("Step 3: Loop Position Analysis")
    print("-" * 70)

    loop_result = await detect_loop_position(test_cell.image_uri)

    print("✅ Loop analysis:")
    print(f"   Dominant position: {loop_result['dominant_position']}")
    print(f"   Interpretation: {loop_result['interpretation']}")

    # Only show detailed loop counts if loops were detected
    if "total_loops" in loop_result:
        print(f"   Total loops: {loop_result['total_loops']}")
        print(f"   Top loops: {loop_result['top_loops']}")
        print(f"   Bottom loops: {loop_result['bottom_loops']}")
    print()

    print("💡 When to use rotation classification:")
    print("-" * 70)
    print("Use rotation testing when characters are orientation-dependent:")
    print("  • Digits: 6 ↔ 9 (180° rotation)")
    print("  • Letters: b ↔ q, d ↔ p, n ↔ u (180° rotation)")
    print("  • Symbols: ! ↔ ¡, ? ↔ ¿ (180° rotation)")
    print("  • Arrows: ← → ↑ ↓ (90° rotations)")
    print("  • Asian characters with directional features")
    print()

    print("Strategies:")
    print("  1. OCR at all angles → vote on result")
    print("  2. Geometric analysis (loop position)")
    print("  3. Template matching at multiple angles")
    print("  4. Combine with Vision LLM for verification")
    print()


if __name__ == "__main__":
    asyncio.run(demo_rotation_classification())

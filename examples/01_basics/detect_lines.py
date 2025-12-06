#!/usr/bin/env python3
"""Example: Line Detection - Detect horizontal and vertical lines in images.

This example demonstrates the detect_lines tool which uses Hough Line Transform
to find straight lines in images. Useful for:
- Grid detection
- Document structure analysis
- Table borders
- Horizon detection
- Any structured content with lines

Tools used:
- detect_lines: Hough Line Transform for line detection
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from chuk_mcp_vision.tools.detection.lines import detect_lines


async def demo_line_detection():
    """Demonstrate line detection on a Sudoku grid."""
    print("=" * 70)
    print("Line Detection Demo")
    print("=" * 70)
    print()
    print("Detecting horizontal and vertical lines in a Sudoku grid...")
    print()

    # Load image
    image_path = Path(__file__).parent.parent.parent / "assets" / "sudoku.jpeg"
    if not image_path.exists():
        print(f"❌ Error: {image_path} not found")
        return

    print(f"📸 Image: {image_path.name}")
    print()

    # Detect lines
    print("Step 1: Detect Lines")
    print("-" * 70)
    result = await detect_lines(str(image_path))

    print("✅ Lines detected:")
    print(f"   Horizontal lines: {result.horizontal_lines}")
    print(f"   Vertical lines: {result.vertical_lines}")
    print(f"   Total lines: {len(result.lines)}")
    print()

    # Show first few lines
    if result.lines:
        print("First 5 line segments:")
        for i, line in enumerate(result.lines[:5]):
            angle_str = f", angle={line.angle:.1f}°" if line.angle is not None else ""
            length_str = f", length={line.length:.0f}px" if line.length is not None else ""
            print(
                f"   {i + 1}. ({line.x1:.0f},{line.y1:.0f}) → ({line.x2:.0f},{line.y2:.0f})"
                f"{angle_str}{length_str}"
            )
        print()

    # Analysis
    print("Analysis:")
    print("-" * 70)

    h_count = result.horizontal_lines
    v_count = result.vertical_lines

    if h_count == v_count:
        if h_count == 10 and v_count == 10:
            print("✅ This appears to be a 9x9 grid (Sudoku)")
            print(f"   {h_count - 1} rows × {v_count - 1} columns")
        else:
            print(f"✅ This appears to be a {h_count - 1}x{v_count - 1} grid")
    else:
        print(f"⚠️  Asymmetric grid detected: {h_count} horizontal, {v_count} vertical")

    print()
    print("Use cases for line detection:")
    print("  • Grid structure detection (tables, Sudoku, spreadsheets)")
    print("  • Document layout analysis")
    print("  • Border detection")
    print("  • Horizon/vanishing point detection")
    print("  • Structural analysis of buildings/architecture")
    print()


if __name__ == "__main__":
    asyncio.run(demo_line_detection())

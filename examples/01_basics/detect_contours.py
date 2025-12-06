#!/usr/bin/env python3
"""Example: Contour Detection - Detect shapes and boundaries in images.

This example demonstrates the detect_contours tool which finds boundaries
of objects in images. Useful for:
- Shape detection
- Object segmentation
- Boundary finding
- Component isolation
- Region of interest detection

Tools used:
- detect_contours: OpenCV contour detection
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from chuk_mcp_vision.tools.detection.contours import detect_contours


async def demo_contour_detection():
    """Demonstrate contour detection on a Sudoku grid."""
    print("=" * 70)
    print("Contour Detection Demo")
    print("=" * 70)
    print()
    print("Detecting contours (shapes and boundaries) in a Sudoku grid...")
    print()

    # Load image
    image_path = Path(__file__).parent.parent.parent / "assets" / "sudoku.jpeg"
    if not image_path.exists():
        print(f"❌ Error: {image_path} not found")
        return

    print(f"📸 Image: {image_path.name}")
    print()

    # Detect contours
    print("Step 1: Detect All Contours")
    print("-" * 70)
    result = await detect_contours(
        str(image_path),
        min_area=100,  # Ignore tiny contours
        max_contours=100,  # Limit results
    )

    print(f"✅ Contours detected: {result.total_contours}")
    print()

    # Analyze contours by size
    if result.contours:
        # Sort by area
        sorted_contours = sorted(result.contours, key=lambda c: c.area, reverse=True)

        print("Top 5 largest contours:")
        for i, contour in enumerate(sorted_contours[:5]):
            bbox = contour.bounding_box
            print(
                f"   {i + 1}. Area: {contour.area:.0f}px², "
                f"Perimeter: {contour.perimeter:.0f}px, "
                f"Bbox: {bbox['width']:.0f}×{bbox['height']:.0f}px"
            )
        print()

        # Analyze shapes
        print("Step 2: Analyze Contour Shapes")
        print("-" * 70)

        # Calculate extent for each contour
        def get_extent(contour):
            bbox = contour.bounding_box
            bbox_area = bbox["width"] * bbox["height"]
            return contour.area / bbox_area if bbox_area > 0 else 0

        rectangles = sum(1 for c in result.contours if 0.8 < get_extent(c) < 1.0)
        complex_shapes = sum(1 for c in result.contours if get_extent(c) < 0.8)

        print(f"Rectangle-like contours: {rectangles}")
        print(f"Complex shapes: {complex_shapes}")
        print()

        # Find the grid boundary (largest contour)
        largest = sorted_contours[0]
        bbox = largest.bounding_box
        extent = get_extent(largest)
        print("Grid Boundary (largest contour):")
        print(f"   Position: ({bbox['x']:.0f}, {bbox['y']:.0f})")
        print(f"   Size: {bbox['width']:.0f}×{bbox['height']:.0f}px")
        print(f"   Area: {largest.area:.0f}px²")
        print(f"   Extent: {extent:.2f} (area/bbox ratio)")
        print()

    print("Use cases for contour detection:")
    print("  • Object segmentation and isolation")
    print("  • Shape detection and analysis")
    print("  • Boundary finding (documents, grids, regions)")
    print("  • Component counting")
    print("  • Quality inspection (defect detection)")
    print("  • Image preprocessing (crop to content)")
    print()


if __name__ == "__main__":
    asyncio.run(demo_contour_detection())

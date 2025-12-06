#!/usr/bin/env python3
"""Example: Perspective Transform - Correct skewed/rotated documents.

This example demonstrates the perspective_transform tool which corrects
perspective distortion in images. Useful for:
- Deskewing documents
- Correcting camera angles
- Straightening photos
- Document scanning
- QR code/barcode correction

Tools used:
- detect_contours: Find document boundary
- perspective_transform: Apply perspective correction
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from chuk_mcp_vision.tools.detection.contours import detect_contours
from chuk_mcp_vision.tools.transform.perspective import perspective_transform


async def demo_perspective_transform():
    """Demonstrate perspective correction on a Sudoku grid."""
    print("=" * 70)
    print("Perspective Transform Demo")
    print("=" * 70)
    print()
    print("Correcting perspective distortion in images...")
    print()

    # Load image
    image_path = Path(__file__).parent.parent.parent / "assets" / "sudoku.jpeg"
    if not image_path.exists():
        print(f"❌ Error: {image_path} not found")
        return

    print(f"📸 Image: {image_path.name}")
    print()

    # Step 1: Detect document boundary
    print("Step 1: Detect Document Boundary")
    print("-" * 70)
    contours = await detect_contours(
        str(image_path),
        min_area=10000,  # Large contours only
        max_contours=100,
    )

    if not contours.contours:
        print("❌ No contours found")
        return

    # Find largest contour (likely the grid boundary)
    largest = max(contours.contours, key=lambda c: c.area)
    bbox = largest.bounding_box
    print("✅ Document boundary detected:")
    print(f"   Area: {largest.area:.0f}px²")
    print(f"   Bounding box: {bbox['width']:.0f}×{bbox['height']:.0f}px")
    print(f"   Position: ({bbox['x']:.0f}, {bbox['y']:.0f})")
    print()

    # Step 2: Apply perspective transform
    print("Step 2: Apply Perspective Transform")
    print("-" * 70)

    # Get corner points from contour
    # For demo, use bounding box corners
    tl_x, tl_y = bbox["x"], bbox["y"]  # Top-left
    tr_x, tr_y = bbox["x"] + bbox["width"], bbox["y"]  # Top-right
    br_x, br_y = bbox["x"] + bbox["width"], bbox["y"] + bbox["height"]  # Bottom-right
    bl_x, bl_y = bbox["x"], bbox["y"] + bbox["height"]  # Bottom-left

    result = await perspective_transform(
        str(image_path),
        top_left_x=tl_x,
        top_left_y=tl_y,
        top_right_x=tr_x,
        top_right_y=tr_y,
        bottom_right_x=br_x,
        bottom_right_y=br_y,
        bottom_left_x=bl_x,
        bottom_left_y=bl_y,
        output_width=int(bbox["width"]),
        output_height=int(bbox["height"]),
    )

    print("✅ Perspective corrected:")
    print(f"   Output image: {result.output_image_uri}")
    print("   Transform matrix:")
    for i, row in enumerate(result.transform_matrix):
        print(f"      [{row[0]:.3f}, {row[1]:.3f}, {row[2]:.3f}]")
    print()

    print("💡 What is perspective transform?")
    print("-" * 70)
    print("Perspective transform corrects the distortion that occurs when")
    print("taking photos at an angle. It maps 4 points from the source image")
    print("to 4 points in the destination, creating a 'bird's eye view'.")
    print()

    print("Use cases:")
    print("  • Document scanning (photos of paper)")
    print("  • Whiteboard capture")
    print("  • Book/magazine scanning")
    print("  • License plate/ID card straightening")
    print("  • QR code correction")
    print("  • Architectural photos (building facades)")
    print()

    print("Tips:")
    print("  • Accurate corner detection is critical")
    print("  • Use contour detection to find boundaries")
    print("  • Order points correctly: TL, TR, BR, BL")
    print("  • Preprocessing (edge detection) helps")
    print()


if __name__ == "__main__":
    asyncio.run(demo_perspective_transform())

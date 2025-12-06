#!/usr/bin/env python3
"""Example: Region Extraction - Extract specific regions from images.

This example demonstrates the extract_region tool for cropping and
extracting specific areas of interest from images.

Tools used:
- detect_contours: Find regions of interest
- extract_region: Crop specific regions
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from chuk_mcp_vision.tools.detection.contours import detect_contours
from chuk_mcp_vision.tools.transform.crop import extract_region


async def demo_region_extraction():
    """Demonstrate region extraction."""
    print("=" * 70)
    print("Region Extraction Demo")
    print("=" * 70)
    print()
    print("Extracting specific regions from an image...")
    print()

    # Load image
    image_path = Path(__file__).parent.parent.parent / "assets" / "sudoku.jpeg"
    if not image_path.exists():
        print(f"❌ Error: {image_path} not found")
        return

    print(f"📸 Image: {image_path.name}")
    print()

    # Step 1: Detect regions
    print("Step 1: Detect Regions of Interest")
    print("-" * 70)

    contours = await detect_contours(str(image_path), min_area=1000, max_contours=100)

    print(f"✅ Found {contours.total_contours} contours")
    print()

    # Find the largest contour (main grid)
    if contours.contours:
        largest = max(contours.contours, key=lambda c: c.area)
        bbox = largest.bounding_box
        print("Largest region (grid boundary):")
        print(f"   Area: {largest.area:.0f}px²")
        print(f"   Position: ({bbox['x']:.0f}, {bbox['y']:.0f})")
        print(f"   Size: {bbox['width']:.0f}×{bbox['height']:.0f}px")
        print()

        # Step 2: Extract the grid region
        print("Step 2: Extract Grid Region")
        print("-" * 70)

        grid_region = await extract_region(
            str(image_path),
            x=int(bbox["x"]),
            y=int(bbox["y"]),
            width=int(bbox["width"]),
            height=int(bbox["height"]),
        )

        print("✅ Grid extracted:")
        print(f"   Output URI: {grid_region.output_image_uri}")
        print(f"   Dimensions: {int(bbox['width'])}×{int(bbox['height'])}px")
        print()

        # Step 3: Extract smaller regions (top-left quadrant)
        print("Step 3: Extract Smaller Region (top-left quadrant)")
        print("-" * 70)

        quad_width = int(bbox["width"] // 2)
        quad_height = int(bbox["height"] // 2)

        quadrant = await extract_region(
            str(image_path),
            x=int(bbox["x"]),
            y=int(bbox["y"]),
            width=quad_width,
            height=quad_height,
        )

        print("✅ Quadrant extracted:")
        print(f"   Output URI: {quadrant.output_image_uri}")
        print(f"   Dimensions: {quad_width}×{quad_height}px")
        print()

        # Step 4: Extract center cell (approximate)
        print("Step 4: Extract Center Region")
        print("-" * 70)

        cell_width = int(bbox["width"] / 9)
        cell_height = int(bbox["height"] / 9)
        center_x = int(bbox["x"] + cell_width * 4)  # 5th column
        center_y = int(bbox["y"] + cell_height * 4)  # 5th row

        center_cell = await extract_region(
            str(image_path), x=center_x, y=center_y, width=cell_width, height=cell_height
        )

        print("✅ Center cell extracted:")
        print(f"   Output URI: {center_cell.output_image_uri}")
        print(f"   Position: ({center_x}, {center_y})")
        print(f"   Dimensions: {cell_width}×{cell_height}px")
        print()

    print("💡 Use cases for region extraction:")
    print("-" * 70)
    print("  • Crop to content (remove margins)")
    print("  • Extract specific fields (forms, invoices)")
    print("  • Focus processing on region of interest")
    print("  • Split images into tiles")
    print("  • Extract faces/objects from scenes")
    print("  • Grid cell extraction")
    print()

    print("Tips:")
    print("  • Use contour detection to find boundaries automatically")
    print("  • Padding helps avoid clipping important content")
    print("  • Validate coordinates are within image bounds")
    print("  • Extracted regions can be processed independently")
    print()


if __name__ == "__main__":
    asyncio.run(demo_region_extraction())

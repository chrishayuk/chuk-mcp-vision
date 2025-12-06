#!/usr/bin/env python3
"""Basic image pipeline example - demonstrates artifact-based image flow.

This example shows:
1. Loading an image from disk
2. Detecting lines (grid detection)
3. Saving artifacts to chuk-artifacts
4. Loading artifacts by UUID
5. Automatic cleanup

No Sudoku solving - just the vision pipeline.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np

from chuk_mcp_vision.tools.common import (
    init_vision_store,
    load_image_from_artifact,
    save_image_as_artifact,
)
from chuk_mcp_vision.tools.detect.lines import detect_lines


async def main():
    """Run the basic image pipeline example."""
    print("=" * 60)
    print("CHUK Vision - Basic Image Pipeline Example")
    print("=" * 60)
    print()

    # Initialize the vision artifact store
    print("1. Initializing chuk-artifacts store...")
    store = await init_vision_store()
    print(f"   ✅ Store initialized: {store}")
    print("   📦 Provider: vfs-memory (in-memory, ephemeral)")
    print()

    # Create a simple test image (simulated Sudoku grid)
    print("2. Creating test image (9x9 grid)...")
    img = create_test_grid(size=900, cells=9)
    print(f"   ✅ Image created: {img.shape[1]}x{img.shape[0]} pixels")
    print()

    # Save as artifact
    print("3. Saving image as artifact...")
    artifact_id = await save_image_as_artifact(
        img, suffix=".jpg", category="test", metadata={"source": "generated", "type": "grid"}
    )
    print("   ✅ Artifact saved!")
    print(f"   📝 Artifact ID: {artifact_id}")
    print("   🏷️  Category: test")
    print("   📊 Metadata: source=generated, type=grid")
    print()

    # Load artifact back
    print("4. Loading artifact by UUID...")
    loaded_img = await load_image_from_artifact(artifact_id)
    print(f"   ✅ Image loaded: {loaded_img.shape[1]}x{loaded_img.shape[0]} pixels")
    print(f"   ✓ Verified: {np.array_equal(img, loaded_img)}")
    print()

    # Detect lines in the grid
    print("5. Detecting lines (grid structure)...")
    lines_result = await detect_lines(artifact_id, min_line_length=100, threshold=150)
    print("   ✅ Lines detected!")
    print(f"   ➖ Horizontal lines: {lines_result.horizontal_lines}")
    print(f"   | Vertical lines: {lines_result.vertical_lines}")
    print(f"   📏 Total lines: {len(lines_result.lines)}")
    print()

    # Show detected lines info
    if lines_result.lines:
        print("6. Line detection details:")
        print("   First few lines:")
        for i, line in enumerate(lines_result.lines[:5]):
            print(
                f"   - Line {i + 1}: ({line.x1:.0f},{line.y1:.0f}) → "
                f"({line.x2:.0f},{line.y2:.0f}), angle={line.angle:.1f}°"
            )
        if len(lines_result.lines) > 5:
            print(f"   ... and {len(lines_result.lines) - 5} more lines")
        print()

    # Demonstrate artifact storage
    print("7. Artifact store statistics:")
    namespaces = store.list_namespaces()
    print(f"   📦 Total artifacts in store: {len(namespaces)}")
    for ns in namespaces:
        print(f"   - {ns.namespace_id[:8]}... (type={ns.type}, scope={ns.scope})")
    print()

    # Demonstrate cleanup behavior
    print("8. Session cleanup demonstration:")
    print("   ℹ️  All artifacts are SESSION-scoped")
    print("   ℹ️  They will auto-delete when session ends")
    print("   ℹ️  No manual cleanup needed!")
    print()

    print("=" * 60)
    print("✅ Example completed successfully!")
    print("=" * 60)
    print()
    print("Key takeaways:")
    print("  • Images stored as BLOB artifacts (not temp files)")
    print("  • UUID-based references (clean pipeline composition)")
    print("  • Metadata tracking (category, dimensions, custom)")
    print("  • Automatic cleanup (session-scoped)")
    print("  • Pluggable backends (memory → S3 with zero code changes)")


def create_test_grid(size: int = 900, cells: int = 9) -> np.ndarray:
    """Create a test grid image (simulates Sudoku board).

    Args:
        size: Image size (square)
        cells: Number of cells per side (9 for Sudoku)

    Returns:
        OpenCV image (white background, black grid lines)
    """
    # Create white image
    img = np.ones((size, size, 3), dtype=np.uint8) * 255

    cell_size = size // cells

    # Draw grid lines
    for i in range(cells + 1):
        pos = i * cell_size
        # Horizontal lines
        cv2.line(img, (0, pos), (size, pos), (0, 0, 0), 2)
        # Vertical lines
        cv2.line(img, (pos, 0), (pos, size), (0, 0, 0), 2)

    # Make outer border thicker
    cv2.rectangle(img, (0, 0), (size - 1, size - 1), (0, 0, 0), 4)

    return img


if __name__ == "__main__":
    asyncio.run(main())

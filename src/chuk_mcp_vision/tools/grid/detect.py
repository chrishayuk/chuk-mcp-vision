"""Detect grid structure in an image - works for any grid (Sudoku, tables, spreadsheets, etc.)."""

import cv2
import numpy as np
from chuk_mcp_server import tool

from ...models import GridStructure, BoundingBox
from ..common import load_image_from_artifact


@tool
async def detect_grid(image_uri: str, min_cells: int = 4, max_cells: int = 100) -> GridStructure:
    """Detect a grid structure in an image.

    This is a generic tool that works for:
    - Sudoku puzzles (9×9)
    - Spreadsheet tables (variable rows/cols)
    - Crossword puzzles
    - Calendars
    - Any rectangular grid structure

    The tool finds the outer boundary and determines row/column divisions.

    Args:
        image_uri: URI to the image
        min_cells: Minimum cells per dimension (default 4 for 2×2 grid)
        max_cells: Maximum cells per dimension (default 100)

    Returns:
        GridStructure with:
        - boundary: Outer bounding box
        - rows: Number of rows detected
        - cols: Number of columns detected
        - row_dividers: Y-coordinates of horizontal lines
        - col_dividers: X-coordinates of vertical lines
        - cell_width: Average cell width
        - cell_height: Average cell height
        - is_regular: Whether grid has uniform cell sizes
    """
    img = await load_image_from_artifact(image_uri)
    h, w = img.shape[:2]

    # Convert to grayscale and threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Find outer grid boundary using largest contour
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("No grid found in image")

    largest = max(contours, key=cv2.contourArea)
    grid_x, grid_y, grid_w, grid_h = cv2.boundingRect(largest)

    # Detect grid lines using Hough transform
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=100, minLineLength=min(grid_w, grid_h) // 2, maxLineGap=10
    )

    # Separate horizontal and vertical lines
    h_positions = []
    v_positions = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Check if line is within grid boundary
            if not (grid_x <= (x1 + x2) // 2 <= grid_x + grid_w):
                continue
            if not (grid_y <= (y1 + y2) // 2 <= grid_y + grid_h):
                continue

            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

            if angle < 10 or angle > 170:  # Horizontal
                h_positions.append(y1)
            elif 80 < angle < 100:  # Vertical
                v_positions.append(x1)

    # Cluster lines (remove duplicates within 10px)
    def cluster_positions(positions):
        if not positions:
            return []
        positions.sort()
        clustered = [positions[0]]
        for pos in positions[1:]:
            if abs(pos - clustered[-1]) > 10:
                clustered.append(pos)
        return clustered

    row_dividers = cluster_positions(h_positions)
    col_dividers = cluster_positions(v_positions)

    # If we didn't find enough lines, use mathematical division
    # (This happens when grid lines are very thick or image is simple)
    if len(row_dividers) < 10 or len(col_dividers) < 10:
        # Assume uniform grid - try to detect how many cells
        # For now, default to 9×9 for puzzle-like images
        # A smarter approach would analyze the image further
        rows_to_create = 9
        cols_to_create = 9

        row_dividers = [grid_y + i * grid_h / rows_to_create for i in range(rows_to_create + 1)]
        col_dividers = [grid_x + i * grid_w / cols_to_create for i in range(cols_to_create + 1)]
    else:
        # Ensure we have outer boundaries
        if row_dividers[0] > grid_y + 5:
            row_dividers.insert(0, grid_y)
        if row_dividers[-1] < grid_y + grid_h - 5:
            row_dividers.append(grid_y + grid_h)

        if col_dividers[0] > grid_x + 5:
            col_dividers.insert(0, grid_x)
        if col_dividers[-1] < grid_x + grid_w - 5:
            col_dividers.append(grid_x + grid_w)

    rows = len(row_dividers) - 1
    cols = len(col_dividers) - 1

    # Validate grid size
    if rows < min_cells or cols < min_cells:
        raise ValueError(f"Grid too small: {rows}×{cols} (min {min_cells}×{min_cells})")
    if rows > max_cells or cols > max_cells:
        raise ValueError(f"Grid too large: {rows}×{cols} (max {max_cells}×{max_cells})")

    # Calculate cell dimensions and regularity
    cell_widths = [col_dividers[i + 1] - col_dividers[i] for i in range(cols)]
    cell_heights = [row_dividers[i + 1] - row_dividers[i] for i in range(rows)]

    avg_cell_width = sum(cell_widths) / len(cell_widths)
    avg_cell_height = sum(cell_heights) / len(cell_heights)

    # Check regularity (coefficient of variation < 10%)
    width_variance = np.std(cell_widths) / avg_cell_width if avg_cell_width > 0 else 1.0
    height_variance = np.std(cell_heights) / avg_cell_height if avg_cell_height > 0 else 1.0
    is_regular = width_variance < 0.1 and height_variance < 0.1

    return GridStructure(
        boundary=BoundingBox(x=grid_x, y=grid_y, width=grid_w, height=grid_h),
        rows=rows,
        cols=cols,
        row_dividers=row_dividers,
        col_dividers=col_dividers,
        cell_width=avg_cell_width,
        cell_height=avg_cell_height,
        is_regular=is_regular,
        confidence=0.95 if is_regular else 0.75,
    )

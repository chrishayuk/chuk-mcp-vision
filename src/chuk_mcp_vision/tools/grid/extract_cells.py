"""Extract individual cells from a detected grid."""

import cv2
import numpy as np
from chuk_mcp_server import tool

from ...models import GridStructure, CellLocation, ExtractGridCellsResponse
from ..common import load_image_from_artifact, save_image_as_artifact


@tool
async def extract_grid_cells(
    image_uri: str, grid: GridStructure, margin: int = 5, preprocess: bool = True
) -> ExtractGridCellsResponse:
    """Extract all cells from a grid as individual images.

    This is a generic tool that works for any grid structure.
    It divides the grid into cells based on detected row/column dividers
    and saves each cell as a separate artifact.

    Args:
        image_uri: URI to the image
        grid: Grid structure from detect_grid
        margin: Pixels to crop from each edge (to avoid grid lines)
        preprocess: Whether to preprocess cells (clean, enhance, resize)

    Returns:
        ExtractGridCellsResponse with:
        - cells: List of cell locations with their image URIs
        - total_cells: Total number of cells extracted
        - grid_size: (rows, cols)
    """
    img = await load_image_from_artifact(image_uri)

    cells = []

    for row in range(grid.rows):
        for col in range(grid.cols):
            # Get cell boundaries from dividers
            y1 = int(grid.row_dividers[row])
            y2 = int(grid.row_dividers[row + 1])
            x1 = int(grid.col_dividers[col])
            x2 = int(grid.col_dividers[col + 1])

            # Apply margin to avoid grid lines
            y1 += margin
            y2 -= margin
            x1 += margin
            x2 -= margin

            # Ensure valid bounds
            if y2 <= y1 or x2 <= x1:
                continue

            # Extract cell
            cell_img = img[y1:y2, x1:x2].copy()

            # Preprocess if requested
            if preprocess:
                cell_img = _preprocess_cell(cell_img)

            # Save as artifact
            cell_uri = await save_image_as_artifact(
                cell_img, suffix=".jpg", category="grid_cells", metadata={"row": row, "col": col}
            )

            cells.append(
                CellLocation(
                    row=row, col=col, image_uri=cell_uri, x=x1, y=y1, width=x2 - x1, height=y2 - y1
                )
            )

    return ExtractGridCellsResponse(
        cells=cells, total_cells=len(cells), grid_size=(grid.rows, grid.cols)
    )


def _preprocess_cell(cell_img):
    """Preprocess a cell for better OCR/classification.

    - Convert to grayscale
    - Apply Otsu thresholding
    - Remove noise
    - Scale up if too small
    - Add border
    """
    # Convert to grayscale
    if len(cell_img.shape) == 3:
        gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cell_img

    # Otsu's thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert if digit is dark on light background
    # (we want black on white for most OCR)
    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)

    # Remove small noise
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Scale up if too small (OCR works better on larger images)
    h, w = binary.shape
    if h < 64 or w < 64:
        scale = max(64 / h, 64 / w)
        binary = cv2.resize(binary, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Add white border (helps OCR focus on content)
    bordered = cv2.copyMakeBorder(binary, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)

    # Convert back to BGR for storage
    return cv2.cvtColor(bordered, cv2.COLOR_GRAY2BGR)

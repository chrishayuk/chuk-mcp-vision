"""Line detection tool using Hough transform."""

import math

import cv2
import numpy as np
from chuk_mcp_server import tool

from ...models import DetectLinesResponse, LineSegment
from ..common import load_image_from_artifact


@tool
async def detect_lines(
    image_uri: str,
    min_line_length: int = 50,
    max_line_gap: int = 10,
    threshold: int = 100,
) -> DetectLinesResponse:
    """Detect straight lines in an image using Hough Line Transform.

    This is a fundamental primitive for finding grids, tables, document edges,
    and any structured content with straight lines.

    Args:
        image_uri: URI or file path to the image
        min_line_length: Minimum line length in pixels (shorter lines ignored)
        max_line_gap: Maximum gap in pixels between line segments to treat as single line
        threshold: Hough transform accumulator threshold (higher = stricter detection)

    Returns:
        DetectLinesResponse: Contains:
            - lines: List of line segments with (x1, y1, x2, y2) coordinates
            - horizontal_lines: Count of horizontal lines detected
            - vertical_lines: Count of vertical lines detected
            - image_width/height: Image dimensions

    Example:
        # Find grid lines in a Sudoku puzzle
        result = await detect_lines("sudoku.jpg", min_line_length=100)
        # result.horizontal_lines = 10  (9x9 grid has 10 lines each direction)
        # result.vertical_lines = 10

    Use cases:
        - Grid detection (Sudoku, chess, spreadsheets)
        - Table structure extraction
        - Document edge detection
        - Perspective correction (find edges to warp)
    """
    img = await load_image_from_artifact(image_uri)
    height, width = img.shape[:2]

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Probabilistic Hough Line Transform
    lines_raw = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    lines = []
    horizontal_count = 0
    vertical_count = 0

    if lines_raw is not None:
        for line in lines_raw:
            x1, y1, x2, y2 = line[0]

            # Calculate angle and length
            dx = x2 - x1
            dy = y2 - y1
            length = math.sqrt(dx * dx + dy * dy)
            angle = math.degrees(math.atan2(dy, dx)) % 180

            # Count horizontal vs vertical (within 10 degree tolerance)
            if angle < 10 or angle > 170:
                horizontal_count += 1
            elif 80 < angle < 100:
                vertical_count += 1

            lines.append(
                LineSegment(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    angle=angle,
                    length=length,
                )
            )

    return DetectLinesResponse(
        lines=lines,
        image_width=width,
        image_height=height,
        horizontal_lines=horizontal_count,
        vertical_lines=vertical_count,
    )

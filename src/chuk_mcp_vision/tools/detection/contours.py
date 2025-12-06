"""Contour detection tool."""

import cv2
from chuk_mcp_server import tool

from ...models import Contour, DetectContoursResponse, Point
from ..common import load_image_from_artifact


@tool
async def detect_contours(
    image_uri: str,
    min_area: float = 100.0,
    max_contours: int = 100,
) -> DetectContoursResponse:
    """Detect contours (shape boundaries) in an image.

    Finds closed shapes and object boundaries. Useful for detecting regions,
    separating objects, or finding document/card boundaries.

    Args:
        image_uri: URI or file path to the image
        min_area: Minimum contour area in square pixels (filters noise)
        max_contours: Maximum number of contours to return (largest first)

    Returns:
        DetectContoursResponse: Contains:
            - contours: List of contours with points, area, perimeter, bounding box
            - total_contours: How many contours found
            - image_width/height: Image dimensions

    Example:
        # Find document boundary in a photo
        result = await detect_contours("document_photo.jpg", min_area=1000)
        # result.contours[0] is the largest contour (likely the document)
        # Use its points for perspective_transform

    Use cases:
        - Document/card boundary detection
        - Object segmentation
        - Shape detection (circles, rectangles, etc.)
        - Region extraction
    """
    img = await load_image_from_artifact(image_uri)
    height, width = img.shape[:2]

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold to binary
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours_raw, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filter and convert to our model
    contours = []
    for cnt in contours_raw:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        perimeter = cv2.arcLength(cnt, True)
        x, y, w, h = cv2.boundingRect(cnt)

        # Convert points
        points = [Point(x=float(pt[0][0]), y=float(pt[0][1])) for pt in cnt]

        contours.append(
            Contour(
                points=points,
                area=area,
                perimeter=perimeter,
                bounding_box={"x": float(x), "y": float(y), "width": float(w), "height": float(h)},
                is_closed=True,
            )
        )

    # Sort by area (largest first) and limit
    contours.sort(key=lambda c: c.area, reverse=True)
    contours = contours[:max_contours]

    return DetectContoursResponse(
        contours=contours,
        image_width=width,
        image_height=height,
        total_contours=len(contours),
    )

"""Perspective transformation tool."""

import cv2
import numpy as np
from chuk_mcp_server import tool

from ...models import PerspectiveTransformResponse
from ..common import load_image_from_artifact, save_image_as_artifact


@tool
async def perspective_transform(
    image_uri: str,
    top_left_x: float,
    top_left_y: float,
    top_right_x: float,
    top_right_y: float,
    bottom_right_x: float,
    bottom_right_y: float,
    bottom_left_x: float,
    bottom_left_y: float,
    output_width: int = 800,
    output_height: int = 800,
) -> PerspectiveTransformResponse:
    """Apply perspective transformation to correct skew and distortion.

    Given 4 corner points of a quadrilateral, warps it to a rectangle.
    Essential for document scanning and grid straightening.

    Args:
        image_uri: URI or file path to the image
        top_left_x, top_left_y: Top-left corner coordinates
        top_right_x, top_right_y: Top-right corner coordinates
        bottom_right_x, bottom_right_y: Bottom-right corner coordinates
        bottom_left_x, bottom_left_y: Bottom-left corner coordinates
        output_width: Desired width of output image
        output_height: Desired height of output image

    Returns:
        PerspectiveTransformResponse: Contains:
            - output_image_uri: URI of transformed image (saved to temp/virtual-fs)
            - output_width/height: Dimensions of output
            - transform_matrix: The 3x3 transformation matrix (optional, for debugging)

    Example:
        # Straighten a document photo
        contours = await detect_contours("document.jpg")
        doc_corners = contours.contours[0].points  # Assume largest is document
        result = await perspective_transform(
            "document.jpg",
            doc_corners[0].x, doc_corners[0].y,  # top-left
            doc_corners[1].x, doc_corners[1].y,  # top-right
            doc_corners[2].x, doc_corners[2].y,  # bottom-right
            doc_corners[3].x, doc_corners[3].y,  # bottom-left
            output_width=1000, output_height=1400
        )
        # result.output_image_uri points to straightened document

    Use cases:
        - Document scanning (straighten photos of documents)
        - Grid extraction (straighten angled Sudoku/chess boards)
        - Business card scanning
        - License plate normalization
    """
    img = await load_image_from_artifact(image_uri)

    # Source points (the quadrilateral corners)
    src_points = np.array(
        [
            [top_left_x, top_left_y],
            [top_right_x, top_right_y],
            [bottom_right_x, bottom_right_y],
            [bottom_left_x, bottom_left_y],
        ],
        dtype=np.float32,
    )

    # Destination points (rectangle)
    dst_points = np.array(
        [
            [0, 0],
            [output_width - 1, 0],
            [output_width - 1, output_height - 1],
            [0, output_height - 1],
        ],
        dtype=np.float32,
    )

    # Get transformation matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # Apply transformation
    warped = cv2.warpPerspective(img, matrix, (output_width, output_height))

    # Save output as artifact (returns namespace ID)
    output_uri = await save_image_as_artifact(
        warped, suffix=".jpg", category="transformed", metadata={"transform": "perspective"}
    )

    return PerspectiveTransformResponse(
        output_image_uri=output_uri,
        output_width=output_width,
        output_height=output_height,
        transform_matrix=matrix.tolist(),
    )

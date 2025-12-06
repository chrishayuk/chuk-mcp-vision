"""Region extraction tool."""

from chuk_mcp_server import tool

from ...models import BoundingBox, ExtractRegionResponse
from ..common import load_image_from_artifact, save_image_as_artifact


@tool
async def extract_region(
    image_uri: str,
    x: float,
    y: float,
    width: float,
    height: float,
) -> ExtractRegionResponse:
    """Extract (crop) a rectangular region from an image.

    Simple cropping operation - extracts a sub-region and saves it for
    further processing in the pipeline.

    Args:
        image_uri: URI or file path to the image
        x: Top-left X coordinate of region
        y: Top-left Y coordinate of region
        width: Width of region to extract
        height: Height of region to extract

    Returns:
        ExtractRegionResponse: Contains:
            - output_image_uri: URI of extracted region (saved to temp/virtual-fs)
            - region: The bounding box that was extracted
            - output_width/height: Dimensions of extracted region

    Example:
        # Extract a single Sudoku cell
        cell = await extract_region("sudoku.jpg", x=50, y=50, width=60, height=60)
        # Now OCR just this cell
        digit = await ocr_region(cell.output_image_uri)

    Use cases:
        - Cell extraction from grids
        - Region-specific OCR
        - Focus on specific areas for further processing
        - Pipeline intermediate step
    """
    img = await load_image_from_artifact(image_uri)

    # Convert to integers
    x1 = int(x)
    y1 = int(y)
    x2 = int(x + width)
    y2 = int(y + height)

    # Clip to image bounds
    h, w = img.shape[:2]
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    # Extract region
    region_img = img[y1:y2, x1:x2]

    # Save output as artifact (returns namespace ID)
    output_uri = await save_image_as_artifact(
        region_img,
        suffix=".jpg",
        category="regions",
        metadata={
            "source_region": {
                "x": float(x),
                "y": float(y),
                "width": float(width),
                "height": float(height),
            }
        },
    )

    return ExtractRegionResponse(
        output_image_uri=output_uri,
        region=BoundingBox(x=float(x1), y=float(y1), width=float(x2 - x1), height=float(y2 - y1)),
        output_width=x2 - x1,
        output_height=y2 - y1,
    )

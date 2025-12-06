"""Thresholding tools - binarization for better OCR."""

import cv2
from chuk_mcp_server import tool

from ...models import PreprocessResult
from ..common import load_image_from_artifact, save_image_as_artifact


@tool
async def adaptive_threshold(
    image_uri: str, block_size: int = 11, c_constant: int = 2, method: str = "gaussian"
) -> PreprocessResult:
    """Apply adaptive thresholding for better OCR on varying lighting.

    Adaptive thresholding calculates threshold for small regions,
    working better than global thresholding for:
    - Uneven lighting
    - Shadows
    - Gradients
    - Document scans

    Args:
        image_uri: URI to input image
        block_size: Neighborhood size (must be odd, default: 11)
        c_constant: Constant subtracted from mean (default: 2)
        method: "gaussian" (weighted) or "mean" (simple average)

    Returns:
        PreprocessResult with:
        - output_image_uri: Thresholded image
        - operation: "adaptive_threshold"
        - parameters: Settings used
        - improvement_estimate: Expected OCR improvement
    """
    # Load image
    img = await load_image_from_artifact(image_uri)

    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Ensure block_size is odd
    if block_size % 2 == 0:
        block_size += 1

    # Apply adaptive threshold
    if method == "gaussian":
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c_constant
        )
    else:  # mean
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, c_constant
        )

    # Save result
    output_uri = await save_image_as_artifact(
        thresh,
        suffix=".png",
        category="preprocessed",
        metadata={
            "operation": "adaptive_threshold",
            "block_size": block_size,
            "c_constant": c_constant,
            "method": method,
        },
    )

    # Estimate improvement (adaptive threshold typically helps 10-30% on uneven lighting)
    improvement = 0.2

    return PreprocessResult(
        output_image_uri=output_uri,
        operation="adaptive_threshold",
        parameters={"block_size": block_size, "c_constant": c_constant, "method": method},
        improvement_estimate=improvement,
        metadata={"input_size": f"{gray.shape[1]}x{gray.shape[0]}", "output_format": "binary"},
    )


@tool
async def otsu_threshold(image_uri: str, blur_kernel: int = 5) -> PreprocessResult:
    """Apply Otsu's binarization - automatic threshold detection.

    Otsu's method automatically finds optimal threshold value
    by maximizing between-class variance. Works best for:
    - Bimodal histograms (clear foreground/background)
    - Uniform lighting
    - High contrast images

    Args:
        image_uri: URI to input image
        blur_kernel: Gaussian blur kernel size before thresholding (0 = no blur)

    Returns:
        PreprocessResult with:
        - output_image_uri: Binary image
        - operation: "otsu_threshold"
        - parameters: Settings + computed threshold value
        - improvement_estimate: Expected OCR improvement
    """
    # Load image
    img = await load_image_from_artifact(image_uri)

    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Optional blur to reduce noise
    if blur_kernel > 0:
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

    # Otsu's thresholding
    threshold_value, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Save result
    output_uri = await save_image_as_artifact(
        thresh,
        suffix=".png",
        category="preprocessed",
        metadata={
            "operation": "otsu_threshold",
            "blur_kernel": blur_kernel,
            "computed_threshold": int(threshold_value),
        },
    )

    # Estimate improvement (Otsu works great on clean images)
    improvement = 0.15

    return PreprocessResult(
        output_image_uri=output_uri,
        operation="otsu_threshold",
        parameters={"blur_kernel": blur_kernel, "computed_threshold": int(threshold_value)},
        improvement_estimate=improvement,
        metadata={
            "input_size": f"{gray.shape[1]}x{gray.shape[0]}",
            "threshold_value": int(threshold_value),
            "output_format": "binary",
        },
    )

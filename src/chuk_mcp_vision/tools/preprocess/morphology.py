"""Morphological operations - noise removal and shape enhancement."""

import cv2
import numpy as np
from chuk_mcp_server import tool

from ...models import PreprocessResult
from ..common import load_image_from_artifact, save_image_as_artifact


@tool
async def morphological_clean(
    image_uri: str, operation: str = "open", kernel_size: int = 3, iterations: int = 1
) -> PreprocessResult:
    """Apply morphological operations to clean image.

    Morphological operations use structuring elements to:
    - Remove noise
    - Fill holes
    - Separate touching objects
    - Enhance shapes

    Operations:
    - "open": Remove small white noise (erosion then dilation)
    - "close": Fill small black holes (dilation then erosion)
    - "erode": Shrink foreground
    - "dilate": Expand foreground
    - "gradient": Edge detection (dilation - erosion)

    Args:
        image_uri: URI to input image (preferably binary)
        operation: Morphological operation ("open", "close", "erode", "dilate", "gradient")
        kernel_size: Structuring element size (default: 3)
        iterations: Number of times to apply (default: 1)

    Returns:
        PreprocessResult with cleaned image
    """
    # Load image
    img = await load_image_from_artifact(image_uri)

    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Create structuring element (rectangular kernel)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Apply morphological operation
    if operation == "open":
        result = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=iterations)
    elif operation == "close":
        result = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    elif operation == "erode":
        result = cv2.erode(gray, kernel, iterations=iterations)
    elif operation == "dilate":
        result = cv2.dilate(gray, kernel, iterations=iterations)
    elif operation == "gradient":
        result = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    else:
        raise ValueError(
            f"Unknown operation: {operation}. Must be: open, close, erode, dilate, gradient"
        )

    # Save result
    output_uri = await save_image_as_artifact(
        result,
        suffix=".png",
        category="preprocessed",
        metadata={
            "operation": f"morphology_{operation}",
            "kernel_size": kernel_size,
            "iterations": iterations,
        },
    )

    # Estimate improvement based on operation
    improvement_map = {
        "open": 0.1,  # Removes noise
        "close": 0.08,  # Fills holes
        "erode": 0.05,  # Thins
        "dilate": 0.05,  # Thickens
        "gradient": 0.15,  # Edges
    }
    improvement = improvement_map.get(operation, 0.1)

    return PreprocessResult(
        output_image_uri=output_uri,
        operation=f"morphology_{operation}",
        parameters={"operation": operation, "kernel_size": kernel_size, "iterations": iterations},
        improvement_estimate=improvement,
        metadata={"input_size": f"{gray.shape[1]}x{gray.shape[0]}"},
    )


@tool
async def remove_noise(
    image_uri: str, method: str = "bilateral", strength: int = 5
) -> PreprocessResult:
    """Remove noise while preserving edges.

    Noise removal methods:
    - "bilateral": Preserves edges (best for text/documents)
    - "gaussian": Simple blur (fast)
    - "median": Removes salt-and-pepper noise
    - "nlm": Non-local means (slow but best quality)

    Args:
        image_uri: URI to input image
        method: Denoising method
        strength: Denoising strength (higher = more blur)

    Returns:
        PreprocessResult with denoised image
    """
    # Load image
    img = await load_image_from_artifact(image_uri)

    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Apply denoising based on method
    if method == "bilateral":
        # Bilateral filter - preserves edges
        result = cv2.bilateralFilter(gray, strength * 2 + 1, strength * 15, strength * 15)
    elif method == "gaussian":
        # Gaussian blur - fast but blurs edges
        ksize = strength * 2 + 1
        result = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    elif method == "median":
        # Median filter - removes salt-and-pepper noise
        ksize = strength * 2 + 1
        result = cv2.medianBlur(gray, ksize)
    elif method == "nlm":
        # Non-local means - best quality but slow
        result = cv2.fastNlMeansDenoising(gray, None, strength * 3, 7, 21)
    else:
        raise ValueError(f"Unknown method: {method}. Must be: bilateral, gaussian, median, nlm")

    # Save result
    output_uri = await save_image_as_artifact(
        result,
        suffix=".png",
        category="preprocessed",
        metadata={"operation": f"denoise_{method}", "strength": strength},
    )

    # Estimate improvement based on method
    improvement_map = {
        "bilateral": 0.15,  # Best for text
        "gaussian": 0.08,  # Simple
        "median": 0.12,  # Good for speckle noise
        "nlm": 0.18,  # Best quality
    }
    improvement = improvement_map.get(method, 0.1)

    return PreprocessResult(
        output_image_uri=output_uri,
        operation=f"denoise_{method}",
        parameters={"method": method, "strength": strength},
        improvement_estimate=improvement,
        metadata={"input_size": f"{gray.shape[1]}x{gray.shape[0]}"},
    )

"""Enhancement tools - contrast, sharpness, and quality improvement."""

import cv2
import numpy as np
from chuk_mcp_server import tool

from ...models import PreprocessResult
from ..common import load_image_from_artifact, save_image_as_artifact


@tool
async def enhance_contrast(
    image_uri: str, method: str = "clahe", clip_limit: float = 2.0, tile_size: int = 8
) -> PreprocessResult:
    """Enhance image contrast for better OCR.

    Methods:
    - "clahe": Adaptive histogram equalization (best for text)
    - "histogram": Global histogram equalization
    - "normalize": Stretch histogram to full range

    Args:
        image_uri: URI to input image
        method: Enhancement method
        clip_limit: CLAHE clip limit (prevents over-amplification)
        tile_size: CLAHE tile grid size

    Returns:
        PreprocessResult with enhanced image
    """
    # Load image
    img = await load_image_from_artifact(image_uri)

    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Apply enhancement based on method
    if method == "clahe":
        # CLAHE - best for varying lighting
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        result = clahe.apply(gray)
    elif method == "histogram":
        # Global histogram equalization - simple but can over-enhance
        result = cv2.equalizeHist(gray)
    elif method == "normalize":
        # Normalize to full 0-255 range
        result = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    else:
        raise ValueError(f"Unknown method: {method}. Must be: clahe, histogram, normalize")

    # Save result
    output_uri = await save_image_as_artifact(
        result,
        suffix=".png",
        category="preprocessed",
        metadata={
            "operation": f"enhance_contrast_{method}",
            "clip_limit": clip_limit if method == "clahe" else None,
            "tile_size": tile_size if method == "clahe" else None,
        },
    )

    # Estimate improvement
    improvement_map = {
        "clahe": 0.25,  # Very effective
        "histogram": 0.15,  # Can over-enhance
        "normalize": 0.1,  # Basic
    }
    improvement = improvement_map.get(method, 0.15)

    return PreprocessResult(
        output_image_uri=output_uri,
        operation=f"enhance_contrast_{method}",
        parameters={
            "method": method,
            "clip_limit": clip_limit if method == "clahe" else None,
            "tile_size": tile_size if method == "clahe" else None,
        },
        improvement_estimate=improvement,
        metadata={"input_size": f"{gray.shape[1]}x{gray.shape[0]}"},
    )


@tool
async def sharpen_image(
    image_uri: str, strength: float = 1.5, method: str = "unsharp"
) -> PreprocessResult:
    """Sharpen image for better OCR on blurry text.

    Methods:
    - "unsharp": Unsharp masking (best quality)
    - "laplacian": Laplacian sharpening (fast)

    Args:
        image_uri: URI to input image
        strength: Sharpening strength (1.0 = original, >1.0 = sharper)
        method: Sharpening method

    Returns:
        PreprocessResult with sharpened image
    """
    # Load image
    img = await load_image_from_artifact(image_uri)

    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Apply sharpening based on method
    if method == "unsharp":
        # Unsharp masking - gaussian blur then subtract
        gaussian = cv2.GaussianBlur(gray, (0, 0), 2.0)
        result = cv2.addWeighted(gray, strength, gaussian, -(strength - 1.0), 0)
        result = np.clip(result, 0, 255).astype(np.uint8)
    elif method == "laplacian":
        # Laplacian sharpening - fast edge enhancement
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        result = gray - (strength - 1.0) * laplacian
        result = np.clip(result, 0, 255).astype(np.uint8)
    else:
        raise ValueError(f"Unknown method: {method}. Must be: unsharp, laplacian")

    # Save result
    output_uri = await save_image_as_artifact(
        result,
        suffix=".png",
        category="preprocessed",
        metadata={"operation": f"sharpen_{method}", "strength": strength},
    )

    # Estimate improvement
    improvement_map = {
        "unsharp": 0.2,  # Very effective on blur
        "laplacian": 0.15,  # Fast but can amplify noise
    }
    improvement = improvement_map.get(method, 0.15)

    return PreprocessResult(
        output_image_uri=output_uri,
        operation=f"sharpen_{method}",
        parameters={"method": method, "strength": strength},
        improvement_estimate=improvement,
        metadata={"input_size": f"{gray.shape[1]}x{gray.shape[0]}"},
    )


@tool
async def upscale_image(
    image_uri: str, scale: int = 2, method: str = "bicubic"
) -> PreprocessResult:
    """Upscale image for better OCR on small text.

    Larger images give OCR engines more pixels to work with.
    Useful for:
    - Small text (< 20px height)
    - Low-resolution scans
    - Mobile phone photos

    Methods:
    - "bicubic": Smooth interpolation (best quality)
    - "lanczos": Sharp interpolation (preserves edges)
    - "nearest": Pixel replication (fast, pixelated)

    Args:
        image_uri: URI to input image
        scale: Upscaling factor (2 = double size, 4 = quadruple)
        method: Interpolation method

    Returns:
        PreprocessResult with upscaled image
    """
    # Load image
    img = await load_image_from_artifact(image_uri)

    # Get dimensions
    h, w = img.shape[:2]
    new_w, new_h = w * scale, h * scale

    # Apply upscaling based on method
    if method == "bicubic":
        result = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    elif method == "lanczos":
        result = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    elif method == "nearest":
        result = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    else:
        raise ValueError(f"Unknown method: {method}. Must be: bicubic, lanczos, nearest")

    # Save result
    output_uri = await save_image_as_artifact(
        result,
        suffix=".png",
        category="preprocessed",
        metadata={
            "operation": f"upscale_{method}",
            "scale": scale,
            "original_size": f"{w}x{h}",
            "new_size": f"{new_w}x{new_h}",
        },
    )

    # Estimate improvement (upscaling helps significantly on small text)
    improvement = 0.3 if scale >= 2 else 0.15

    return PreprocessResult(
        output_image_uri=output_uri,
        operation=f"upscale_{method}",
        parameters={"method": method, "scale": scale},
        improvement_estimate=improvement,
        metadata={
            "original_size": f"{w}x{h}",
            "new_size": f"{new_w}x{new_h}",
            "scale_factor": scale,
        },
    )

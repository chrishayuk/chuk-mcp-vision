"""Rotation-based classification tool - generic for orientation-dependent content."""

import cv2
import numpy as np
from typing import List, Dict, Optional
from chuk_mcp_server import tool

from ...models import RotateAndClassifyResponse, OrientationResult
from ..common import load_image_from_artifact, save_image_as_artifact


@tool
async def rotate_and_classify(
    image_uri: str, angles: Optional[List[int]] = None, classifier_callback=None
) -> RotateAndClassifyResponse:
    """Classify content at multiple rotations to find correct orientation.

    Generic tool for orientation-dependent content:
    - Digits: 6 vs 9 (180° rotation)
    - Letters: b vs q, d vs p, n vs u (180° rotation)
    - Symbols: ? vs ¿, ! vs ¡ (180° rotation)
    - Directional arrows: ← → ↑ ↓
    - Characters in non-Latin scripts

    Strategy:
    1. Rotate image to each angle
    2. Classify at each orientation
    3. Compare consistency and confidence
    4. Return most likely correct orientation + value

    Args:
        image_uri: URI to image
        angles: Angles to test (default: [0, 90, 180, 270])
        classifier_callback: Optional custom classifier function

    Returns:
        RotateAndClassifyResponse with:
        - results_by_angle: Classification at each angle
        - recommended_angle: Most likely correct orientation
        - recommended_value: Most likely correct value
        - confidence: Overall confidence
        - disambiguation: What orientations disambiguate
    """
    # Default angles if not provided
    if angles is None:
        angles = [0, 90, 180, 270]

    img = await load_image_from_artifact(image_uri)
    h, w = img.shape[:2]

    results = []

    for angle in angles:
        # Rotate image
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            img,
            rotation_matrix,
            (w, h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )

        # Save rotated image
        rotated_uri = await save_image_as_artifact(
            rotated, suffix=".jpg", category="rotated", metadata={"angle": angle}
        )

        # Classify rotated image
        if classifier_callback:
            classification = await classifier_callback(rotated_uri)
        else:
            # Default: use OCR
            from ..ocr.tool import ocr_region
            from ...models import OCRBackend

            ocr_result = await ocr_region(rotated_uri, backend=OCRBackend.PADDLEOCR)
            classification = {
                "value": ocr_result.full_text.strip(),
                "confidence": ocr_result.mean_confidence,
            }

        results.append(
            OrientationResult(
                angle=angle,
                rotated_image_uri=rotated_uri,
                classification_value=classification.get("value"),
                confidence=classification.get("confidence", 0.0),
                metadata=classification.get("metadata", {}),
            )
        )

    # Analyze results to find correct orientation
    analysis = _analyze_rotation_results(results)

    return RotateAndClassifyResponse(
        results_by_angle=results,
        recommended_angle=analysis["best_angle"],
        recommended_value=analysis["best_value"],
        confidence=analysis["confidence"],
        disambiguation=analysis["disambiguation"],
    )


def _analyze_rotation_results(results: List[OrientationResult]) -> Dict:
    """Analyze rotation results to determine correct orientation.

    Strategy:
    1. Look for inversions (6→9, b→q, etc.)
    2. Prefer higher confidence results
    3. Check for consistency across non-inverted rotations
    """
    # Group by classification value
    value_groups = {}
    for result in results:
        val = result.classification_value
        if val:
            if val not in value_groups:
                value_groups[val] = []
            value_groups[val].append(result)

    # Check for 180° inversions (common case: 6/9, b/q, d/p)
    inversions = []
    for i, r1 in enumerate(results):
        for r2 in results[i + 1 :]:
            if abs(r1.angle - r2.angle) == 180:
                if r1.classification_value != r2.classification_value:
                    inversions.append(
                        {
                            "angle1": r1.angle,
                            "value1": r1.classification_value,
                            "conf1": r1.confidence,
                            "angle2": r2.angle,
                            "value2": r2.classification_value,
                            "conf2": r2.confidence,
                        }
                    )

    # Determine best result
    if inversions:
        # Found orientation-dependent content
        # Choose based on confidence
        best_inversion = max(inversions, key=lambda x: max(x["conf1"], x["conf2"]))

        if best_inversion["conf1"] > best_inversion["conf2"]:
            best_angle = best_inversion["angle1"]
            best_value = best_inversion["value1"]
            confidence = best_inversion["conf1"]
        else:
            best_angle = best_inversion["angle2"]
            best_value = best_inversion["value2"]
            confidence = best_inversion["conf2"]

        disambiguation = (
            f"180° rotation changes {best_inversion['value1']} ↔ {best_inversion['value2']}"
        )
    else:
        # No orientation-dependent content detected
        # Return highest confidence result
        best_result = max(results, key=lambda r: r.confidence)
        best_angle = best_result.angle
        best_value = best_result.classification_value
        confidence = best_result.confidence
        disambiguation = "No orientation dependency detected"

    return {
        "best_angle": best_angle,
        "best_value": best_value,
        "confidence": confidence,
        "disambiguation": disambiguation,
        "inversions": inversions,
    }


@tool
async def detect_loop_position(image_uri: str) -> Dict:
    """Detect position of loops in characters (generic geometric analysis).

    Works for:
    - Digits: 6 (loop bottom), 9 (loop top), 8 (loops both)
    - Letters: b (loop bottom), d (loop top), p (loop bottom), q (loop top)
    - Any character with circular/oval features

    Args:
        image_uri: URI to image

    Returns:
        Dict with:
        - loop_positions: List of (x, y, area) for detected loops
        - dominant_position: "top", "bottom", "middle", "both", "none"
        - confidence: Detection confidence
    """
    img = await load_image_from_artifact(image_uri)

    # Convert to grayscale and threshold
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filter for loop-like contours (circular, medium-sized)
    h, w = gray.shape
    loops = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 50 or area > (h * w * 0.5):  # Too small or too large
            continue

        # Check circularity
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)

        if circularity > 0.5:  # Reasonably circular
            # Get center
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                loops.append({"x": cx, "y": cy, "area": area, "circularity": circularity})

    # Determine dominant position
    if not loops:
        return {
            "loop_positions": [],
            "dominant_position": "none",
            "confidence": 1.0,
            "interpretation": "No loops detected",
        }

    # Categorize by position (top third, middle third, bottom third)
    third_height = h / 3

    top_loops = [loop for loop in loops if loop["y"] < third_height]
    bottom_loops = [loop for loop in loops if loop["y"] >= 2 * third_height]

    # Determine dominant position
    if len(top_loops) > 0 and len(bottom_loops) > 0:
        dominant = "both"
        interpretation = "Loops in both top and bottom (possibly '8' or similar)"
    elif len(top_loops) > len(bottom_loops):
        dominant = "top"
        interpretation = "Loop at top (possibly '9', 'd', etc.)"
    elif len(bottom_loops) > len(top_loops):
        dominant = "bottom"
        interpretation = "Loop at bottom (possibly '6', 'b', 'p', etc.)"
    else:
        dominant = "middle"
        interpretation = "Loop in middle"

    confidence = 0.8 if loops else 0.5

    return {
        "loop_positions": loops,
        "dominant_position": dominant,
        "confidence": confidence,
        "interpretation": interpretation,
        "total_loops": len(loops),
        "top_loops": len(top_loops),
        "bottom_loops": len(bottom_loops),
    }


@tool
async def compare_to_reference(
    image_uri: str, reference_uris: List[str], method: str = "ssim"
) -> Dict:
    """Compare image to reference images using similarity metrics.

    Generic tool for:
    - Finding most similar character in a set
    - Validating OCR results (compare to known instances)
    - Detecting outliers (dissimilar to all references)
    - Font matching

    Args:
        image_uri: URI to image to classify
        reference_uris: List of URIs to reference images (labeled)
        method: Similarity method ("ssim", "mse", "histogram", "orb")

    Returns:
        Dict with:
        - similarities: List of (reference_uri, score) sorted by similarity
        - best_match: Most similar reference
        - confidence: Match confidence
        - outlier: Whether image is dissimilar to all references
    """
    from skimage.metrics import structural_similarity as ssim

    query_img = await load_image_from_artifact(image_uri)
    query_gray = (
        cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY) if len(query_img.shape) == 3 else query_img
    )

    similarities = []

    for ref_uri in reference_uris:
        ref_img = await load_image_from_artifact(ref_uri)
        ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY) if len(ref_img.shape) == 3 else ref_img

        # Resize to same size for comparison
        h, w = query_gray.shape
        ref_resized = cv2.resize(ref_gray, (w, h))

        # Calculate similarity based on method
        if method == "ssim":
            # Structural Similarity Index
            score, _ = ssim(query_gray, ref_resized, full=True)
        elif method == "mse":
            # Mean Squared Error (lower is better, so invert)
            mse = np.mean((query_gray.astype(float) - ref_resized.astype(float)) ** 2)
            score = 1.0 / (1.0 + mse / 1000)  # Normalize
        elif method == "histogram":
            # Histogram comparison
            query_hist = cv2.calcHist([query_gray], [0], None, [256], [0, 256])
            ref_hist = cv2.calcHist([ref_resized], [0], None, [256], [0, 256])
            score = cv2.compareHist(query_hist, ref_hist, cv2.HISTCMP_CORREL)
        elif method == "orb":
            # ORB feature matching
            orb = cv2.ORB_create()
            kp1, des1 = orb.detectAndCompute(query_gray, None)
            kp2, des2 = orb.detectAndCompute(ref_resized, None)

            if des1 is not None and des2 is not None:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(des1, des2)
                score = len(matches) / max(len(kp1), len(kp2))
            else:
                score = 0.0
        else:
            raise ValueError(f"Unknown method: {method}")

        similarities.append({"reference_uri": ref_uri, "score": float(score)})

    # Sort by score (descending)
    similarities_sorted = sorted(similarities, key=lambda x: x["score"], reverse=True)

    best_match = similarities_sorted[0] if similarities_sorted else None
    best_score = best_match["score"] if best_match else 0.0

    # Check if outlier (dissimilar to all references)
    avg_score = np.mean([s["score"] for s in similarities]) if similarities else 0.0
    is_outlier = avg_score < 0.5

    return {
        "similarities": similarities_sorted,
        "best_match": best_match,
        "best_score": best_score,
        "average_similarity": float(avg_score),
        "is_outlier": is_outlier,
        "confidence": best_score,
    }

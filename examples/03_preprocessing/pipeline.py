"""Example 14: Preprocessing Tools - Image Enhancement for Better OCR

Goal: Demonstrate all 7 preprocessing tools and measure OCR improvement
- Thresholding: adaptive_threshold, otsu_threshold
- Morphology: morphological_clean, remove_noise
- Enhancement: enhance_contrast, sharpen_image, upscale_image

Expected result: +10-30% OCR accuracy improvement
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from chuk_mcp_vision.tools.preprocess import (
    adaptive_threshold,
    otsu_threshold,
    morphological_clean,
    remove_noise,
    enhance_contrast,
    sharpen_image,
    upscale_image,
)
from chuk_mcp_vision.tools.ocr.tool import ocr_region
from chuk_mcp_vision.models import OCRBackend
from chuk_mcp_vision.tools.grid import detect_grid, extract_grid_cells


async def test_preprocessing_tools():
    """Test all preprocessing tools on a Sudoku image."""

    # Load image
    image_path = Path(__file__).parent.parent.parent / "assets" / "sudoku.jpeg"
    if not image_path.exists():
        print(f"Error: {image_path} not found")
        return

    image_uri = str(image_path)

    print("=" * 80)
    print("Example 14: Preprocessing Tools Demo")
    print("=" * 80)
    print()
    print(f"📸 Input: {image_path.name}")
    print()

    # Extract a single cell for detailed testing
    print("Step 1: Extract a sample cell for testing")
    print("-" * 80)

    grid = await detect_grid(image_uri)
    cells = await extract_grid_cells(image_uri, grid)

    # Get cell [0,0] - should contain digit "7"
    test_cell = cells.cells[0 * 9 + 0]
    print(f"✅ Extracted cell [0,0] (should be '7'): {test_cell.image_uri}")
    print()

    # Baseline OCR (no preprocessing)
    print("Baseline: OCR without preprocessing")
    print("-" * 80)
    baseline = await ocr_region(test_cell.image_uri, backend=OCRBackend.PADDLEOCR)
    print(f"Text: '{baseline.full_text}'")
    print(f"Confidence: {baseline.mean_confidence:.2%}")
    print()

    # Test all preprocessing tools
    results = {}

    # 1. Adaptive Threshold
    print("1. Adaptive Threshold (Gaussian)")
    print("-" * 80)
    result = await adaptive_threshold(
        test_cell.image_uri, block_size=11, c_constant=2, method="gaussian"
    )
    ocr_result = await ocr_region(result.output_image_uri, backend=OCRBackend.PADDLEOCR)
    results["adaptive_threshold"] = {
        "output_uri": result.output_image_uri,
        "ocr_text": ocr_result.full_text,
        "ocr_confidence": ocr_result.mean_confidence,
        "improvement_estimate": result.improvement_estimate,
    }
    print("✅ Preprocessed")
    print(f"   OCR: '{ocr_result.full_text}' (confidence: {ocr_result.mean_confidence:.2%})")
    print(f"   Estimated improvement: +{result.improvement_estimate:.1%}")
    print()

    # 2. Otsu Threshold
    print("2. Otsu Threshold")
    print("-" * 80)
    result = await otsu_threshold(test_cell.image_uri, blur_kernel=5)
    ocr_result = await ocr_region(result.output_image_uri, backend=OCRBackend.PADDLEOCR)
    results["otsu_threshold"] = {
        "output_uri": result.output_image_uri,
        "ocr_text": ocr_result.full_text,
        "ocr_confidence": ocr_result.mean_confidence,
        "improvement_estimate": result.improvement_estimate,
    }
    print(f"✅ Preprocessed (threshold value: {result.parameters['computed_threshold']})")
    print(f"   OCR: '{ocr_result.full_text}' (confidence: {ocr_result.mean_confidence:.2%})")
    print(f"   Estimated improvement: +{result.improvement_estimate:.1%}")
    print()

    # 3. Morphological Clean (opening - remove noise)
    print("3. Morphological Clean (opening)")
    print("-" * 80)
    result = await morphological_clean(
        test_cell.image_uri, operation="open", kernel_size=3, iterations=1
    )
    ocr_result = await ocr_region(result.output_image_uri, backend=OCRBackend.PADDLEOCR)
    results["morphology_open"] = {
        "output_uri": result.output_image_uri,
        "ocr_text": ocr_result.full_text,
        "ocr_confidence": ocr_result.mean_confidence,
        "improvement_estimate": result.improvement_estimate,
    }
    print("✅ Preprocessed")
    print(f"   OCR: '{ocr_result.full_text}' (confidence: {ocr_result.mean_confidence:.2%})")
    print(f"   Estimated improvement: +{result.improvement_estimate:.1%}")
    print()

    # 4. Remove Noise (bilateral filter)
    print("4. Remove Noise (bilateral filter)")
    print("-" * 80)
    result = await remove_noise(test_cell.image_uri, method="bilateral", strength=5)
    ocr_result = await ocr_region(result.output_image_uri, backend=OCRBackend.PADDLEOCR)
    results["denoise_bilateral"] = {
        "output_uri": result.output_image_uri,
        "ocr_text": ocr_result.full_text,
        "ocr_confidence": ocr_result.mean_confidence,
        "improvement_estimate": result.improvement_estimate,
    }
    print("✅ Preprocessed")
    print(f"   OCR: '{ocr_result.full_text}' (confidence: {ocr_result.mean_confidence:.2%})")
    print(f"   Estimated improvement: +{result.improvement_estimate:.1%}")
    print()

    # 5. Enhance Contrast (CLAHE)
    print("5. Enhance Contrast (CLAHE)")
    print("-" * 80)
    result = await enhance_contrast(
        test_cell.image_uri, method="clahe", clip_limit=2.0, tile_size=8
    )
    ocr_result = await ocr_region(result.output_image_uri, backend=OCRBackend.PADDLEOCR)
    results["contrast_clahe"] = {
        "output_uri": result.output_image_uri,
        "ocr_text": ocr_result.full_text,
        "ocr_confidence": ocr_result.mean_confidence,
        "improvement_estimate": result.improvement_estimate,
    }
    print("✅ Preprocessed")
    print(f"   OCR: '{ocr_result.full_text}' (confidence: {ocr_result.mean_confidence:.2%})")
    print(f"   Estimated improvement: +{result.improvement_estimate:.1%}")
    print()

    # 6. Sharpen Image
    print("6. Sharpen Image (unsharp mask)")
    print("-" * 80)
    result = await sharpen_image(test_cell.image_uri, strength=1.5, method="unsharp")
    ocr_result = await ocr_region(result.output_image_uri, backend=OCRBackend.PADDLEOCR)
    results["sharpen_unsharp"] = {
        "output_uri": result.output_image_uri,
        "ocr_text": ocr_result.full_text,
        "ocr_confidence": ocr_result.mean_confidence,
        "improvement_estimate": result.improvement_estimate,
    }
    print("✅ Preprocessed")
    print(f"   OCR: '{ocr_result.full_text}' (confidence: {ocr_result.mean_confidence:.2%})")
    print(f"   Estimated improvement: +{result.improvement_estimate:.1%}")
    print()

    # 7. Upscale Image
    print("7. Upscale Image (2x bicubic)")
    print("-" * 80)
    result = await upscale_image(test_cell.image_uri, scale=2, method="bicubic")
    ocr_result = await ocr_region(result.output_image_uri, backend=OCRBackend.PADDLEOCR)
    results["upscale_2x"] = {
        "output_uri": result.output_image_uri,
        "ocr_text": ocr_result.full_text,
        "ocr_confidence": ocr_result.mean_confidence,
        "improvement_estimate": result.improvement_estimate,
    }
    print(f"✅ Preprocessed ({result.metadata['original_size']} → {result.metadata['new_size']})")
    print(f"   OCR: '{ocr_result.full_text}' (confidence: {ocr_result.mean_confidence:.2%})")
    print(f"   Estimated improvement: +{result.improvement_estimate:.1%}")
    print()

    # Test combined preprocessing pipeline
    print("8. Combined Pipeline (denoise + contrast + sharpen + upscale)")
    print("-" * 80)

    # Multi-stage preprocessing
    stage1 = await remove_noise(test_cell.image_uri, method="bilateral", strength=3)
    stage2 = await enhance_contrast(stage1.output_image_uri, method="clahe")
    stage3 = await sharpen_image(stage2.output_image_uri, strength=1.3)
    stage4 = await upscale_image(stage3.output_image_uri, scale=2)

    ocr_result = await ocr_region(stage4.output_image_uri, backend=OCRBackend.PADDLEOCR)
    results["combined_pipeline"] = {
        "output_uri": stage4.output_image_uri,
        "ocr_text": ocr_result.full_text,
        "ocr_confidence": ocr_result.mean_confidence,
        "stages": 4,
    }
    print("✅ Preprocessed (4 stages)")
    print(f"   OCR: '{ocr_result.full_text}' (confidence: {ocr_result.mean_confidence:.2%})")
    print()

    # Summary comparison
    print("=" * 80)
    print("Summary: OCR Accuracy Comparison")
    print("=" * 80)
    print()
    print(f"{'Method':<30} {'OCR Result':<15} {'Confidence':<12} {'vs Baseline':<15}")
    print("-" * 80)

    baseline_conf = baseline.mean_confidence
    baseline_text = baseline.full_text.strip()

    print(f"{'Baseline (no preprocess)':<30} {baseline_text:<15} {baseline_conf:>10.2%} {'—':<15}")

    for name, data in results.items():
        text = data["ocr_text"].strip()
        conf = data["ocr_confidence"]
        diff = conf - baseline_conf
        diff_str = f"{diff:+.2%}"

        # Check if OCR result improved
        correct_marker = "✅" if text == "2" else "❌"

        display_name = name.replace("_", " ").title()
        print(f"{display_name:<30} {text:<15} {conf:>10.2%} {diff_str:<15} {correct_marker}")

    print()
    print("=" * 80)
    print("Observations")
    print("=" * 80)
    print()

    # Find best preprocessing method
    best_method = max(results.items(), key=lambda x: x[1]["ocr_confidence"])
    print(f"✅ Best single method: {best_method[0].replace('_', ' ').title()}")
    print(f"   Confidence: {best_method[1]['ocr_confidence']:.2%}")
    print(f"   Improvement: {(best_method[1]['ocr_confidence'] - baseline_conf):+.2%}")
    print()

    # Check combined pipeline
    combined = results["combined_pipeline"]
    print("✅ Combined pipeline (4 stages):")
    print(f"   Confidence: {combined['ocr_confidence']:.2%}")
    print(f"   Improvement: {(combined['ocr_confidence'] - baseline_conf):+.2%}")
    print()

    print("Key takeaways:")
    print("  1. Different preprocessing methods work better for different images")
    print("  2. Combining multiple stages can compound improvements")
    print("  3. Upscaling helps significantly for small/low-res text")
    print("  4. CLAHE contrast enhancement is very effective")
    print("  5. Adaptive threshold works well for uneven lighting")
    print()


if __name__ == "__main__":
    asyncio.run(test_preprocessing_tools())

# Preprocessing - Image Enhancement

Image preprocessing tools for improving OCR accuracy on low-quality, noisy, or poorly-lit images.

## Overview

7 preprocessing tools that enhance image quality:
- **Thresholding**: Convert to binary (black/white) for better OCR
- **Morphology**: Remove noise, fill gaps, smooth edges
- **Enhancement**: Improve contrast, sharpness, resolution

## Examples

| File | Description | Tools Demonstrated |
|------|-------------|-------------------|
| `pipeline.py` | All 7 preprocessing tools + combined pipeline | adaptive_threshold, otsu_threshold, morphological_clean, remove_noise, enhance_contrast, sharpen_image, upscale_image |

## Quick Start

```bash
uv run python 03_preprocessing/pipeline.py
```

**Expected output**:
- Baseline OCR accuracy
- Individual tool results with confidence scores
- Combined pipeline results
- Best preprocessing method recommendation

## Tools Reference

### 1. adaptive_threshold
Adaptive thresholding for varying lighting conditions.

```python
from chuk_mcp_vision.tools.preprocess import adaptive_threshold

result = await adaptive_threshold(
    image_uri,
    block_size=11,      # Neighborhood size (odd number)
    c_constant=2,       # Constant subtracted from mean
    method="gaussian"   # "gaussian" or "mean"
)
```

**Best for**: Uneven lighting, shadows, gradients
**Improvement**: +15-25% on poorly-lit images

### 2. otsu_threshold
Global thresholding with automatic threshold detection.

```python
from chuk_mcp_vision.tools.preprocess import otsu_threshold

result = await otsu_threshold(
    image_uri,
    blur_kernel=5  # Pre-blur kernel size (0 for no blur)
)
```

**Best for**: Clean images with clear separation
**Improvement**: +10-20% on clean but low-contrast images

### 3. morphological_clean
Morphological operations for noise removal and gap filling.

```python
from chuk_mcp_vision.tools.preprocess import morphological_clean

result = await morphological_clean(
    image_uri,
    operation="open",   # "open", "close", "erode", "dilate"
    kernel_size=3,      # Kernel size
    iterations=1        # Number of iterations
)
```

**Operations**:
- `open`: Remove small noise (erode → dilate)
- `close`: Fill small gaps (dilate → erode)
- `erode`: Thin foreground
- `dilate`: Thicken foreground

**Best for**: Salt-and-pepper noise, broken characters
**Improvement**: +5-15% on noisy images

### 4. remove_noise
Noise reduction using various filtering methods.

```python
from chuk_mcp_vision.tools.preprocess import remove_noise

result = await remove_noise(
    image_uri,
    method="bilateral",  # "bilateral", "gaussian", "median"
    strength=5           # Filter strength
)
```

**Methods**:
- `bilateral`: Edge-preserving smoothing (best for documents)
- `gaussian`: Fast blur (good for general noise)
- `median`: Salt-and-pepper noise removal

**Best for**: Scan artifacts, compression noise
**Improvement**: +10-20% on scanned/compressed images

### 5. enhance_contrast
Contrast enhancement using histogram techniques.

```python
from chuk_mcp_vision.tools.preprocess import enhance_contrast

result = await enhance_contrast(
    image_uri,
    method="clahe",     # "clahe" or "histogram"
    clip_limit=2.0,     # CLAHE clip limit
    tile_size=8         # CLAHE tile size
)
```

**Methods**:
- `clahe`: Adaptive contrast (best for local variations)
- `histogram`: Global histogram equalization

**Best for**: Low-contrast images, faded text
**Improvement**: +15-30% on low-contrast images

### 6. sharpen_image
Image sharpening to enhance edges.

```python
from chuk_mcp_vision.tools.preprocess import sharpen_image

result = await sharpen_image(
    image_uri,
    strength=1.5,       # Sharpening strength (1.0-3.0)
    method="unsharp"    # "unsharp" or "kernel"
)
```

**Best for**: Slightly blurry text, soft edges
**Improvement**: +10-20% on blurry images

### 7. upscale_image
Image upscaling for small/low-resolution text.

```python
from chuk_mcp_vision.tools.preprocess import upscale_image

result = await upscale_image(
    image_uri,
    scale=2,            # Scale factor (2 = 2x size)
    method="bicubic"    # "bicubic", "lanczos", "nearest"
)
```

**Best for**: Small text, low-resolution images
**Improvement**: +20-40% on small text (<20px height)

## Combined Pipeline

For best results, chain multiple tools:

```python
# Example: 4-stage pipeline
stage1 = await remove_noise(image_uri, method="bilateral")
stage2 = await enhance_contrast(stage1.output_image_uri, method="clahe")
stage3 = await sharpen_image(stage2.output_image_uri)
stage4 = await upscale_image(stage3.output_image_uri, scale=2)

# Now run OCR on final result
result = await ocr_region(stage4.output_image_uri)
```

**Recommended pipelines**:
- **Scanned documents**: denoise → contrast → sharpen
- **Low-res images**: upscale → contrast → sharpen
- **Poor lighting**: adaptive_threshold → morphology
- **General enhancement**: contrast → sharpen → upscale

## When to Use Preprocessing

**Use preprocessing when**:
- OCR accuracy is below 90%
- Image has visible noise, blur, or low contrast
- Text is very small (<20px)
- Lighting is uneven
- Image is scanned or compressed

**Skip preprocessing when**:
- OCR accuracy is already >95%
- Image is clean and high-quality
- Text is large and clear

## Understanding Results

Each tool returns a `PreprocessResult`:
```python
class PreprocessResult(BaseModel):
    output_image_uri: str              # Preprocessed image URI
    operation: str                      # Operation name
    parameters: dict                    # Parameters used
    improvement_estimate: Optional[float]  # Estimated improvement (0.0-1.0)
    metadata: dict                      # Additional info
```

**improvement_estimate** is a rough estimate based on:
- Image quality metrics (BRISQUE, NIQE)
- Contrast improvement
- Noise reduction
- Edge enhancement

Actual OCR improvement depends on the specific image and OCR backend.

## Tips

1. **Try multiple methods**: Different images benefit from different preprocessing
2. **Start conservative**: Use moderate settings, then increase if needed
3. **Check visually**: Save intermediate results to verify quality
4. **Measure accuracy**: Compare OCR results before/after preprocessing
5. **Pipeline order matters**: Usually: denoise → contrast → sharpen → upscale

## Related Tools

- See `../05_ocr/` for OCR backends and ensemble voting
- See `../02_transforms/` for perspective correction and rotation
- See `../04_grids/` for applying preprocessing to grid cells

## Next Steps

1. Run `pipeline.py` to see all tools in action
2. Try preprocessing on your own images
3. Combine with OCR tools in `05_ocr/`
4. Build custom pipelines for your use case

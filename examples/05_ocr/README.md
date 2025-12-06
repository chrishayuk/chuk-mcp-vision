# OCR - Optical Character Recognition

Multi-backend OCR with ensemble voting for maximum accuracy.

## Overview

Extract text from images using multiple OCR engines and combine results through voting for higher accuracy than any single backend.

## Examples

| File | Description | Accuracy | Prerequisites |
|------|-------------|----------|---------------|
| `multi_backend.py` | Ensemble OCR with voting | 95%+ | None |

## Quick Start

```bash
uv run python 05_ocr/multi_backend.py
```

**Expected output**:
- Individual backend results (PaddleOCR, Tesseract, EasyOCR)
- Voting consensus with confidence
- Accuracy comparison

## Tools Reference

### ocr_region
Single-backend OCR.

```python
from chuk_mcp_vision.tools.ocr import ocr_region
from chuk_mcp_vision.models import OCRBackend

result = await ocr_region(
    image_uri,
    backend=OCRBackend.PADDLEOCR,  # PADDLEOCR, TESSERACT, EASYOCR
    language="en"                   # Language code
)

print(result.full_text)           # Extracted text
print(result.mean_confidence)     # Average confidence
print(result.words)               # List of detected words
```

**Backends**:
- **PaddleOCR**: Best overall, fast, high accuracy (default)
- **Tesseract**: Good fallback, widely supported
- **EasyOCR**: Good for handwriting and non-Latin scripts

### ocr_multi_backend
Multi-backend OCR with voting.

```python
from chuk_mcp_vision.tools.ocr import ocr_multi_backend
from chuk_mcp_vision.models import OCRBackend, VotingStrategy

result = await ocr_multi_backend(
    image_uri,
    backends=[
        OCRBackend.PADDLEOCR,
        OCRBackend.TESSERACT,
        OCRBackend.EASYOCR
    ],
    voting_strategy=VotingStrategy.CONFIDENCE_WEIGHTED,
    min_confidence=0.7
)

print(result.consensus_text)      # Voted consensus
print(result.consensus_confidence) # Consensus confidence
print(result.individual_results)  # Results from each backend
print(result.voting_details)      # Voting breakdown
```

**Voting strategies**:
- `MAJORITY`: Simple majority vote (2 out of 3)
- `CONFIDENCE_WEIGHTED`: Weight by confidence scores (default)
- `UNANIMOUS`: All backends must agree
- `HIGHEST_CONFIDENCE`: Use result with highest confidence

## Backend Comparison

| Backend | Speed | Accuracy | Best For |
|---------|-------|----------|----------|
| PaddleOCR | Fast | 90-95% | General text, digits |
| Tesseract | Medium | 85-90% | Clean printed text |
| EasyOCR | Slow | 85-95% | Handwriting, Asian scripts |

**Recommendation**: Use PaddleOCR for speed, multi-backend for maximum accuracy.

## Voting Strategies Explained

### MAJORITY
Simple majority wins:
```
PaddleOCR: "7"
Tesseract: "7"
EasyOCR: "1"
→ Consensus: "7" (2 out of 3)
```

### CONFIDENCE_WEIGHTED
Weight votes by confidence:
```
PaddleOCR: "7" (0.99)
Tesseract: "7" (0.85)
EasyOCR: "1" (0.60)
→ Consensus: "7" (weighted score: 0.95)
```

### UNANIMOUS
All must agree:
```
PaddleOCR: "7"
Tesseract: "7"
EasyOCR: "7"
→ Consensus: "7" (unanimous)

# If any differ:
PaddleOCR: "7"
Tesseract: "1"
EasyOCR: "7"
→ Consensus: None (not unanimous)
```

### HIGHEST_CONFIDENCE
Use most confident result:
```
PaddleOCR: "7" (0.99)
Tesseract: "1" (0.85)
EasyOCR: "7" (0.60)
→ Consensus: "7" (highest confidence)
```

## Use Cases

### Single Digit Recognition
```python
# For grid cells, use confidence-weighted voting
result = await ocr_multi_backend(
    cell_uri,
    backends=[OCRBackend.PADDLEOCR, OCRBackend.TESSERACT],
    voting_strategy=VotingStrategy.CONFIDENCE_WEIGHTED
)

if result.consensus_confidence > 0.9:
    digit = result.consensus_text.strip()
```

### Full Text Extraction
```python
# For paragraphs, use single backend (faster)
result = await ocr_region(
    document_uri,
    backend=OCRBackend.PADDLEOCR,
    language="en"
)

text = result.full_text
```

### Handwriting Recognition
```python
# For handwriting, use EasyOCR
result = await ocr_region(
    handwriting_uri,
    backend=OCRBackend.EASYOCR,
    language="en"
)
```

### Critical Data Extraction
```python
# For critical data (forms, checks), use unanimous voting
result = await ocr_multi_backend(
    critical_field_uri,
    backends=[
        OCRBackend.PADDLEOCR,
        OCRBackend.TESSERACT,
        OCRBackend.EASYOCR
    ],
    voting_strategy=VotingStrategy.UNANIMOUS
)

if result.consensus_text is not None:
    # All backends agree - high confidence
    value = result.consensus_text
else:
    # Disagreement - needs manual review
    print("Manual review required")
```

## Improving OCR Accuracy

### 1. Preprocessing
Apply preprocessing before OCR:

```python
from chuk_mcp_vision.tools.preprocess import (
    enhance_contrast,
    sharpen_image,
    upscale_image
)

# Enhance image quality
enhanced = await enhance_contrast(image_uri, method="clahe")
sharpened = await sharpen_image(enhanced.output_image_uri)
upscaled = await upscale_image(sharpened.output_image_uri, scale=2)

# Now run OCR
result = await ocr_region(upscaled.output_image_uri)
```

### 2. Multi-Backend Voting
Use ensemble voting for critical text:

```python
result = await ocr_multi_backend(
    image_uri,
    backends=[
        OCRBackend.PADDLEOCR,
        OCRBackend.TESSERACT,
        OCRBackend.EASYOCR
    ],
    voting_strategy=VotingStrategy.CONFIDENCE_WEIGHTED
)
```

### 3. LLM Verification
Verify OCR results with Vision LLMs:

```python
from chuk_mcp_vision.tools.llm import verify_cell_multi_llm

# Get OCR result
ocr_result = await ocr_region(image_uri)

# Verify with LLMs
llm_result = await verify_cell_multi_llm(
    image_uri,
    providers=["openai", "anthropic", "google"],
    expected_type="digit"
)

# Compare results
if ocr_result.full_text != llm_result["consensus"]:
    print(f"Mismatch: OCR={ocr_result.full_text}, LLM={llm_result['consensus']}")
```

## Language Support

All backends support multiple languages:

```python
# English (default)
result = await ocr_region(image_uri, language="en")

# Spanish
result = await ocr_region(image_uri, language="es")

# Chinese
result = await ocr_region(image_uri, language="ch")

# Japanese
result = await ocr_region(image_uri, language="ja")
```

**PaddleOCR languages**: en, ch, ja, ko, es, fr, de, ru, etc.
**Tesseract languages**: 100+ languages (install language packs)
**EasyOCR languages**: 80+ languages

## Performance Tips

1. **Use PaddleOCR for speed**: Fastest backend, good accuracy
2. **Multi-backend for critical data**: Slower but more accurate
3. **Preprocess small/blurry text**: Upscale before OCR
4. **Batch processing**: Process multiple images concurrently with asyncio
5. **Cache results**: Store OCR results to avoid re-processing

## Related Tools

- See `../03_preprocessing/` for image enhancement before OCR
- See `../06_llm/` for LLM-based verification
- See `../04_grids/` for extracting text from grid cells
- See `../07_complete_workflows/` for complete examples

## Next Steps

1. Run `multi_backend.py` to see voting in action
2. Try different voting strategies
3. Combine with preprocessing for better accuracy
4. Use LLM verification for critical text
5. Build custom OCR pipelines for your use case

# Complete Workflows - End-to-End Examples

Production-ready workflows combining multiple tools for real-world tasks.

## Overview

These examples demonstrate complete pipelines from raw image to extracted data:
- Grid detection + preprocessing + OCR + LLM verification
- Multi-stage enhancement pipelines
- Error handling and validation
- Accuracy measurement and reporting

## Examples

| File | Description | Accuracy | Tools Used |
|------|-------------|----------|------------|
| `sudoku_extraction.py` | Complete Sudoku extraction | 98.8% | Grid, OCR, Preprocessing, LLM |

## Quick Start

```bash
uv run python 07_complete_workflows/sudoku_extraction.py
```

**Expected output**:
- Detected grid structure (9x9)
- 81 cell extractions
- OCR results for each cell
- LLM verification for low-confidence cells
- Final Sudoku grid (98.8% accurate)

## Sudoku Extraction Pipeline

Complete workflow achieving 98.8% accuracy:

### Architecture
```
Input Image
    ↓
1. Preprocessing (optional)
    ↓
2. Grid Detection → 9x9 grid structure
    ↓
3. Cell Extraction → 81 individual cells
    ↓
4. Multi-Backend OCR → Initial digit recognition
    ↓
5. LLM Verification → Verify low-confidence results
    ↓
6. Validation → Ensure valid Sudoku
    ↓
Output: Complete Sudoku grid
```

### Code Overview

```python
# 1. Detect grid
grid = await detect_grid(image_uri)

# 2. Extract cells
cells = await extract_grid_cells(image_uri, grid)

# 3. Process each cell
for cell in cells.cells:
    # Multi-backend OCR
    ocr_result = await ocr_multi_backend(
        cell.image_uri,
        backends=[OCRBackend.PADDLEOCR, OCRBackend.TESSERACT],
        voting_strategy=VotingStrategy.CONFIDENCE_WEIGHTED
    )

    # Verify with LLM if confidence is low
    if ocr_result.consensus_confidence < 0.9:
        llm_result = await verify_cell_multi_llm(
            cell.image_uri,
            expected_type="digit"
        )
        digit = llm_result["consensus"]
    else:
        digit = ocr_result.consensus_text

    sudoku[cell.row][cell.col] = digit
```

### Accuracy Breakdown

| Stage | Accuracy | Improvement |
|-------|----------|-------------|
| Single OCR (PaddleOCR) | 92% | baseline |
| Multi-backend voting | 95% | +3% |
| + Preprocessing | 96% | +1% |
| + LLM verification | **98.8%** | +2.8% |

### Performance Metrics

- **Processing time**: ~3-5 seconds per Sudoku
- **API calls**:
  - Grid detection: 1 call
  - OCR: 81 cells × 2 backends = 162 calls
  - LLM: ~5-10 low-confidence cells × 3 providers = 15-30 calls
- **Cost**: ~$0.05-0.10 per Sudoku (LLM calls)

## Workflow Patterns

### Pattern 1: Preprocessing Pipeline
Multi-stage enhancement before extraction:

```python
# Stage 1: Denoise
denoised = await remove_noise(image_uri, method="bilateral")

# Stage 2: Contrast enhancement
enhanced = await enhance_contrast(
    denoised.output_image_uri,
    method="clahe"
)

# Stage 3: Sharpen
sharpened = await sharpen_image(enhanced.output_image_uri)

# Stage 4: Upscale if needed
if image_too_small:
    upscaled = await upscale_image(sharpened.output_image_uri, scale=2)
    final_uri = upscaled.output_image_uri
else:
    final_uri = sharpened.output_image_uri

# Now extract data
grid = await detect_grid(final_uri)
```

### Pattern 2: Cascading Verification
Start with fast methods, escalate if needed:

```python
async def extract_with_cascading_verification(cell_uri):
    # Level 1: Single OCR (fast)
    ocr = await ocr_region(cell_uri, backend=OCRBackend.PADDLEOCR)

    if ocr.mean_confidence > 0.95:
        return ocr.full_text  # High confidence - done

    # Level 2: Multi-backend voting (more accurate)
    multi_ocr = await ocr_multi_backend(
        cell_uri,
        backends=[OCRBackend.PADDLEOCR, OCRBackend.TESSERACT]
    )

    if multi_ocr.consensus_confidence > 0.9:
        return multi_ocr.consensus_text  # Consensus reached - done

    # Level 3: LLM verification (most accurate)
    llm = await verify_cell_multi_llm(cell_uri, expected_type="digit")

    if llm["confidence"] > 0.8:
        return llm["consensus"]  # LLM consensus - done

    # Level 4: Manual review
    return None  # Needs human review
```

### Pattern 3: Error Recovery
Handle failures gracefully:

```python
async def extract_with_retry(image_uri):
    try:
        # Try normal pipeline
        result = await extract_sudoku(image_uri)
        return result
    except GridDetectionError:
        # Grid detection failed - try preprocessing
        enhanced = await enhance_contrast(image_uri, method="clahe")
        sharpened = await sharpen_image(enhanced.output_image_uri)
        result = await extract_sudoku(sharpened.output_image_uri)
        return result
    except OCRError:
        # OCR failed - try different backend
        result = await extract_sudoku(
            image_uri,
            ocr_backend=OCRBackend.TESSERACT
        )
        return result
```

### Pattern 4: Validation and Quality Checks
Ensure output is valid:

```python
async def extract_and_validate(image_uri):
    # Extract data
    grid = await detect_grid(image_uri)
    cells = await extract_grid_cells(image_uri, grid)

    # Validate grid dimensions
    if grid.rows != 9 or grid.cols != 9:
        raise ValidationError("Invalid grid dimensions")

    # Extract digits
    sudoku = extract_digits_from_cells(cells)

    # Validate Sudoku rules
    errors = validate_sudoku(sudoku)

    if errors:
        # Fix errors using LLM verification
        for error in errors:
            cell_uri = cells.cells[error.row * 9 + error.col].image_uri
            llm_result = await verify_cell_multi_llm(cell_uri)
            sudoku[error.row][error.col] = llm_result["consensus"]

    # Re-validate
    errors = validate_sudoku(sudoku)

    if errors:
        raise ValidationError(f"Could not resolve {len(errors)} errors")

    return sudoku
```

## Future Workflows

### Table Extraction (TODO)
Extract data from complex tables:

```python
# Planned workflow:
# 1. Detect table with img2table
# 2. Extract cells
# 3. Classify cell types (header, data, merged)
# 4. OCR text content
# 5. Reconstruct table structure
# 6. Export to CSV/JSON
```

### Form Processing (TODO)
Extract data from forms with text + checkboxes:

```python
# Planned workflow:
# 1. Detect form fields
# 2. Classify field types (text, checkbox, signature)
# 3. Extract text with OCR
# 4. Detect checkbox states
# 5. Validate against form schema
# 6. Export structured data
```

## Best Practices

### 1. Always Measure Accuracy
Track accuracy at each stage:

```python
results = {
    "total": 81,
    "correct": 0,
    "wrong": 0,
    "empty": 0
}

for cell in cells.cells:
    extracted = await extract_digit(cell.image_uri)
    actual = ground_truth[cell.row][cell.col]

    if extracted == actual:
        results["correct"] += 1
    elif extracted is None:
        results["empty"] += 1
    else:
        results["wrong"] += 1

accuracy = results["correct"] / results["total"]
print(f"Accuracy: {accuracy:.2%}")
```

### 2. Log Failures
Track which cells fail for debugging:

```python
failures = []

for cell in cells.cells:
    try:
        digit = await extract_digit(cell.image_uri)
    except Exception as e:
        failures.append({
            "cell": (cell.row, cell.col),
            "error": str(e),
            "uri": cell.image_uri
        })

if failures:
    print(f"Failed to extract {len(failures)} cells:")
    for failure in failures:
        print(f"  [{failure['cell']}]: {failure['error']}")
```

### 3. Cache Results
Avoid re-processing:

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
async def cached_ocr(image_uri: str):
    return await ocr_region(image_uri)

# Or use persistent cache:
import aiosqlite

async def get_or_compute_ocr(image_uri: str):
    # Check cache
    cache_key = hashlib.md5(image_uri.encode()).hexdigest()
    cached = await get_from_cache(cache_key)

    if cached:
        return cached

    # Compute
    result = await ocr_region(image_uri)

    # Store in cache
    await store_in_cache(cache_key, result)

    return result
```

### 4. Handle Edge Cases
Test edge cases thoroughly:

```python
# Test various image qualities
test_cases = [
    "clean_sudoku.jpg",      # High quality
    "blurry_sudoku.jpg",     # Low quality
    "rotated_sudoku.jpg",    # Rotated
    "partial_sudoku.jpg",    # Partial grid
    "handwritten_sudoku.jpg" # Handwritten
]

for test_case in test_cases:
    try:
        result = await extract_sudoku(test_case)
        accuracy = measure_accuracy(result, ground_truth[test_case])
        print(f"{test_case}: {accuracy:.2%}")
    except Exception as e:
        print(f"{test_case}: FAILED - {e}")
```

## Performance Optimization

### 1. Parallel Processing
Process cells concurrently:

```python
import asyncio

# Extract all cells in parallel
tasks = [extract_digit(cell.image_uri) for cell in cells.cells]
digits = await asyncio.gather(*tasks)

# Build sudoku grid
sudoku = [[None for _ in range(9)] for _ in range(9)]
for i, digit in enumerate(digits):
    row = i // 9
    col = i % 9
    sudoku[row][col] = digit
```

### 2. Batch API Calls
Group LLM calls when possible:

```python
# Collect low-confidence cells
low_confidence_cells = [
    cell for cell in cells.cells
    if cell.ocr_confidence < 0.9
]

# Batch verify (if provider supports batching)
results = await verify_batch_multi_llm(
    [cell.image_uri for cell in low_confidence_cells]
)
```

### 3. Optimize Image Sizes
Resize images before processing:

```python
# Resize to optimal size (not too large, not too small)
optimal_width = 1000
if image.shape[1] > optimal_width:
    scale = optimal_width / image.shape[1]
    resized = cv2.resize(image, None, fx=scale, fy=scale)
    image_uri = await save_image_as_artifact(resized)
```

## Related Tools

- See `../03_preprocessing/` for image enhancement
- See `../04_grids/` for grid detection
- See `../05_ocr/` for OCR backends
- See `../06_llm/` for LLM verification

## Next Steps

1. Run `sudoku_extraction.py` to see complete workflow
2. Measure accuracy on your own Sudoku images
3. Adapt workflow for tables or forms
4. Add custom validation logic
5. Optimize for your specific use case

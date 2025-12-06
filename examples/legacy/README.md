# Legacy Examples

Historical examples kept for reference. These examples demonstrate the evolution of the Sudoku extraction pipeline from 46% to 98.8% accuracy.

## Why Legacy?

These examples are **not recommended for new projects** because:
1. **Sudoku-specific**: Hard-coded for Sudoku, not reusable
2. **Lower accuracy**: 46-88% vs 98.8% with new tools
3. **Hard to maintain**: Monolithic code, difficult to debug
4. **Superseded**: Replaced by generic, composable tools

**Use instead**: Examples in `03_preprocessing/`, `04_grids/`, `05_ocr/`, `06_llm/`, `07_complete_workflows/`

## Evolution Timeline

### Phase 1: Basic OCR (46% accuracy)
**01_basic_image_pipeline.py**
- Approach: Direct OCR on raw image
- Result: 46% accuracy
- Problem: No preprocessing, single OCR backend
- Lesson: Raw OCR is insufficient for complex images

### Phase 2: Grid Detection (60-70% accuracy)
**02_sudoku_pipeline.py**, **03_real_sudoku.py**
- Approach: Detect grid, extract cells, OCR each cell
- Result: 60-70% accuracy
- Problem: Poor grid detection, no preprocessing
- Lesson: Grid structure helps but detection must be robust

### Phase 3: Improved Preprocessing (82% accuracy)
**04_sudoku_accurate.py**
- Approach: Thresholding + morphology + grid detection + OCR
- Result: 82% accuracy
- Problem: Still single OCR backend, hard-coded thresholds
- Lesson: Preprocessing significantly improves accuracy

### Phase 4: Vision LLM Experiments (75-80% accuracy)
**05_sudoku_vision_llm.py**, **06_sudoku_hybrid.py**
- Approach: Use Vision LLMs for verification
- Result: 75-80% accuracy (lower than expected)
- Problem: LLMs alone are not better than good OCR
- Lesson: LLMs best for verification, not primary extraction

### Phase 5: Line-Based Detection (85% accuracy)
**07_sudoku_line_based.py**, **08_sudoku_perfect.py**
- Approach: Hough Line Transform for robust grid detection
- Result: 85-88% accuracy
- Problem: Still single OCR backend
- Lesson: Robust grid detection is critical

### Phase 6: Agentic AI (experimental)
**09_agentic_sudoku.py**
- Approach: Give AI agent access to all tools, let it decide
- Result: Experimental, variable accuracy
- Problem: Non-deterministic, hard to debug
- Lesson: Pipelines need structure, not full autonomy

### Phase 7: Modern Architecture (98.8% accuracy)
**→ See 07_complete_workflows/sudoku_extraction.py**
- Approach: Grid detection + multi-backend OCR + LLM verification
- Result: **98.8% accuracy**
- Key improvements:
  1. Generic, reusable tools
  2. Multi-backend ensemble voting
  3. LLM verification for low-confidence cells
  4. Composable preprocessing pipeline
  5. Robust error handling

## Key Insights from Journey

### 1. Don't Hard-Code Domain Logic
❌ Bad (Sudoku-specific):
```python
def extract_sudoku(image):
    # Hard-coded for 9x9 Sudoku
    grid = detect_9x9_grid(image)
    for row in range(9):
        for col in range(9):
            cell = extract_cell(grid, row, col)
            digit = ocr_digit(cell)
```

✅ Good (Generic):
```python
async def extract_grid_data(image_uri, expected_rows=None, expected_cols=None):
    # Works for any grid structure
    grid = await detect_grid(image_uri)
    cells = await extract_grid_cells(image_uri, grid)

    for cell in cells.cells:
        content = await classify_cell_content(cell.image_uri)
```

### 2. Ensemble > Single Method
Single OCR: 92% accuracy
Multi-backend voting: 95% accuracy (+3%)
+ LLM verification: 98.8% accuracy (+3.8%)

### 3. Preprocessing Matters
No preprocessing: 46% accuracy
Basic preprocessing: 82% accuracy (+36%)
Optimized preprocessing: 96% accuracy (+14%)

### 4. LLMs for Verification, Not Extraction
LLM-only extraction: 75-80% accuracy
OCR + LLM verification: 98.8% accuracy

### 5. Tools Over Pipelines
Monolithic pipelines: Hard to debug, maintain, extend
Composable tools: Easy to debug, maintain, extend

## Files Reference

| File | Description | Accuracy | Key Technique |
|------|-------------|----------|---------------|
| 01_basic_image_pipeline.py | Raw OCR | 46% | Direct OCR |
| 02_sudoku_pipeline.py | Basic grid detection | 60% | Grid + OCR |
| 03_real_sudoku.py | Improved grid | 70% | Better grid detection |
| 04_sudoku_accurate.py | Preprocessing | 82% | Threshold + morphology |
| 05_sudoku_vision_llm.py | LLM extraction | 75% | Vision LLMs |
| 06_sudoku_hybrid.py | Hybrid OCR+LLM | 80% | OCR + LLM verification |
| 07_sudoku_line_based.py | Hough lines | 85% | Line detection |
| 08_sudoku_perfect.py | Refined pipeline | 88% | Multi-stage |
| 09_agentic_sudoku.py | AI agent | Variable | Tool autonomy |

## Can I Still Run These?

Yes, but not recommended:
```bash
# These may have import errors or outdated dependencies
uv run python legacy/01_basic_image_pipeline.py
uv run python legacy/04_sudoku_accurate.py
# etc.
```

**Better approach**: Study the code for historical context, but use modern tools in `04_grids/`, `05_ocr/`, `06_llm/`, `07_complete_workflows/`.

## What to Learn from Legacy Code

1. **Evolution of techniques**: See how each iteration improved
2. **Common mistakes**: What approaches didn't work and why
3. **Hard-won lessons**: Insights that led to modern architecture
4. **Historical context**: Why tools were designed the way they are

## Migration Guide

If you have code based on legacy examples:

### Migrate from 01_basic_image_pipeline.py
```python
# OLD (legacy)
result = ocr_image(image)

# NEW (modern)
from chuk_mcp_vision.tools.ocr import ocr_region
from chuk_mcp_vision.models import OCRBackend

result = await ocr_region(image_uri, backend=OCRBackend.PADDLEOCR)
```

### Migrate from 04_sudoku_accurate.py
```python
# OLD (legacy)
preprocessed = preprocess_sudoku(image)
grid = detect_sudoku_grid(preprocessed)

# NEW (modern)
from chuk_mcp_vision.tools.preprocess import enhance_contrast, sharpen_image
from chuk_mcp_vision.tools.grid import detect_grid

enhanced = await enhance_contrast(image_uri, method="clahe")
sharpened = await sharpen_image(enhanced.output_image_uri)
grid = await detect_grid(sharpened.output_image_uri)
```

### Migrate from 08_sudoku_perfect.py
```python
# OLD (legacy)
sudoku = extract_complete_sudoku(image)

# NEW (modern)
# See 07_complete_workflows/sudoku_extraction.py for complete example
from chuk_mcp_vision.tools.grid import detect_grid, extract_grid_cells
from chuk_mcp_vision.tools.ocr import ocr_multi_backend

grid = await detect_grid(image_uri)
cells = await extract_grid_cells(image_uri, grid)

for cell in cells.cells:
    result = await ocr_multi_backend(
        cell.image_uri,
        backends=[OCRBackend.PADDLEOCR, OCRBackend.TESSERACT]
    )
```

## Next Steps

1. **Don't start here**: Use modern examples in other directories
2. **Study for context**: Understand the evolution
3. **Migrate old code**: Replace legacy with modern tools
4. **Build on modern tools**: Start with `04_grids/`, `05_ocr/`, etc.

## Questions?

If you're working with legacy code:
1. Check if there's a modern equivalent in `../04_grids/`, `../05_ocr/`, etc.
2. Read the migration guide above
3. Consult `../07_complete_workflows/sudoku_extraction.py` for the modern approach
4. See `../README.md` for learning paths

# Vision Tools Examples

Comprehensive examples demonstrating CHUK Vision MCP tools from basics to complete workflows.

## Quick Start

```bash
# Install dependencies
uv sync

# Run your first example - grid extraction
uv run python 04_grids/full_pipeline.py

# Try preprocessing tools
uv run python 03_preprocessing/pipeline.py

# Complete Sudoku extraction (98.8% accuracy)
uv run python 07_complete_workflows/sudoku_extraction.py
```

## Categories

### 01_basics/ - Tool Fundamentals
**Status**: ✅ READY

Learn individual tools:
- `detect_lines.py` - Line detection with Hough Transform
- `detect_contours.py` - Contour and shape detection
- `ocr_basic.py` - Basic OCR with single backend
- `embeddings.py` - Image vector embeddings

**Demonstrates**: Core vision primitives (lines, shapes, text, embeddings)

### 02_transforms/ - Image Transformations
**Status**: ✅ READY

Image manipulation tools:
- `perspective_transform.py` - Correct skewed/angled images
- `rotation.py` - Rotation-based classification for 6/9, etc.
- `crop_extract.py` - Extract specific regions

**Demonstrates**: Geometric transformations for better processing

### 03_preprocessing/ - Image Enhancement
**Status**: ✅ READY

Preprocessing tools for better OCR:
- `pipeline.py` - All 7 preprocessing tools demo

**Tools covered**:
- Adaptive thresholding (Gaussian/Mean)
- Otsu thresholding
- Morphological operations (open/close/erode/dilate)
- Noise removal (bilateral/gaussian/median)
- Contrast enhancement (CLAHE/histogram)
- Image sharpening (unsharp mask/kernel)
- Image upscaling (bicubic/lanczos/nearest)

**Expected improvement**: +10-30% OCR accuracy on low-quality images

### 04_grids/ - Grid Detection & Extraction
**Status**: ✅ READY

Generic grid extraction tools:
- `full_pipeline.py` - Complete grid detection and cell extraction

**Workflow**:
1. Detect grid structure (rows/cols)
2. Extract individual cells
3. Classify cell content (empty/filled/text)

**Use cases**: Sudoku, tables, forms, spreadsheets, crosswords

### 05_ocr/ - Optical Character Recognition
**Status**: ✅ READY

OCR with multiple backends:
- `multi_backend.py` - Ensemble voting with multiple OCR engines

**Backends**:
- PaddleOCR (primary)
- Tesseract (backup)
- EasyOCR (backup)
- Vision LLMs (verification)

**Accuracy**: 95%+ with ensemble voting

### 06_llm/ - Vision LLM Tools
**Status**: ✅ READY

Vision LLM verification and consensus:
- `single_llm.py` - Single Vision LLM verification
- `multi_llm_consensus.py` - Multi-provider voting (OpenAI, Anthropic, Google)

**Use cases**: Handwriting, ambiguous characters (6 vs 9), quality verification

### 07_complete_workflows/ - End-to-End Examples
**Status**: ✅ READY (partial)

Production-ready workflows:
- `sudoku_extraction.py` - Complete Sudoku extraction (98.8% accuracy)

**TODO**:
- `table_extraction.py` - Complex table extraction with img2table
- `form_processing.py` - Form extraction with text + checkboxes

### legacy/ - Old Examples
**Status**: 📦 ARCHIVED

Historical examples (reference only):
- 01-08: Old Sudoku pipelines (46-88% accuracy)
- 09: Agentic AI demo
- sudoku_vision_llm_05.py: Duplicate

**Why legacy?**: Replaced by generic, reusable tools

## Learning Paths

### Path 1: Quick Start (30 minutes)
For users who want to get started quickly:

1. **04_grids/full_pipeline.py** - See grid extraction in action
2. **03_preprocessing/pipeline.py** - Learn preprocessing tools
3. **07_complete_workflows/sudoku_extraction.py** - Complete workflow

### Path 2: Deep Dive (2-3 hours)
For users building custom vision pipelines:

1. **01_basics/** - Learn individual tools
2. **02_transforms/** - Image transformations
3. **03_preprocessing/** - Enhancement techniques
4. **04_grids/** - Grid detection
5. **05_ocr/** - OCR backends and voting
6. **06_llm/** - Vision LLM verification
7. **07_complete_workflows/** - Combine everything

### Path 3: Specific Use Case
Jump to relevant category:

- **Tables/Forms** → 04_grids/, 07_complete_workflows/
- **OCR improvement** → 03_preprocessing/, 05_ocr/
- **Handwriting** → 06_llm/, 02_transforms/
- **Document scanning** → 02_transforms/, 03_preprocessing/

## Accuracy Progression

| Example | Accuracy | Key Techniques |
|---------|----------|----------------|
| legacy/01_basic_image_pipeline.py | 46% | Basic OCR only |
| legacy/04_sudoku_accurate.py | 82% | Grid detection + preprocessing |
| legacy/08_sudoku_perfect.py | 88% | Multi-backend OCR |
| 05_ocr/multi_backend.py | 95% | Ensemble voting |
| 07_complete_workflows/sudoku_extraction.py | **98.8%** | Grid + preprocessing + multi-backend + LLM verification |

## Tool Architecture

All tools follow CHUK principles:
- **Async-native**: All tools use async/await
- **Pydantic-native**: Typed models, no dictionaries
- **No magic strings**: Enums for all options
- **Generic**: Reusable across domains
- **Composable**: Tools chain together

## Common Patterns

### Pattern 1: Preprocessing → OCR
```python
# Enhance image quality before OCR
enhanced = await enhance_contrast(image_uri, method="clahe")
upscaled = await upscale_image(enhanced.output_image_uri, scale=2)
result = await ocr_region(upscaled.output_image_uri)
```

### Pattern 2: Multi-Backend Voting
```python
# Get consensus from multiple OCR engines
result = await ocr_multi_backend(
    image_uri,
    backends=[OCRBackend.PADDLEOCR, OCRBackend.TESSERACT],
    voting_strategy=VotingStrategy.CONFIDENCE_WEIGHTED
)
```

### Pattern 3: Grid Extraction
```python
# Extract all cells from a grid
grid = await detect_grid(image_uri)
cells = await extract_grid_cells(image_uri, grid)
for cell in cells.cells:
    content = await classify_cell_content(cell.image_uri)
```

### Pattern 4: LLM Verification
```python
# Verify ambiguous content with multiple LLMs
result = await verify_cell_multi_llm(
    cell_uri,
    providers=["openai", "anthropic", "google"],
    expected_type="digit"
)
```

## Status Legend

- ✅ READY - Fully implemented and tested
- 📝 TODO - Planned but not yet created
- 📦 ARCHIVED - Legacy/reference only

## Tips

1. **Start simple**: Run examples in order (03 → 04 → 05 → 07)
2. **Check assets**: All examples use `assets/sudoku.jpeg`
3. **Read docstrings**: Each tool has comprehensive documentation
4. **Chain tools**: Combine preprocessing + OCR + LLM for best accuracy
5. **Adapt workflows**: Examples are templates - customize for your use case

## Contributing

When adding new examples:
1. Follow category structure (01-07)
2. Use async-native code
3. Return Pydantic models
4. Add comprehensive docstrings
5. Update this README

## Next Steps

See individual category READMEs for detailed usage:
- `03_preprocessing/README.md`
- `04_grids/README.md`
- `05_ocr/README.md`
- `06_llm/README.md`
- `07_complete_workflows/README.md`

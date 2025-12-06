# chuk-mcp-vision

**Generic computer vision MCP server with atomic, composable tools for image analysis.**

Part of the CHUK ecosystem - designed to work with `chuk-ai-planner` for intelligent orchestration of vision primitives.

## Philosophy

This MCP server provides **atomic, generic vision tools** - not domain-specific solutions. The intelligence of "how to parse a Sudoku" or "how to read a chess board" lives in the planner/playbook layer, not in the tools themselves.

Each tool does one thing well:
- `detect_lines` - find straight lines in an image
- `detect_contours` - find shapes and boundaries
- `detect_grid` - find grid structures (grids, tables)
- `extract_grid_cells` - extract individual cells from grids
- `perspective_transform` - correct skew/angle
- `extract_region` - crop a region of interest
- `ocr_region` - read text/digits from a region
- `ocr_multi_backend` - ensemble OCR with voting
- `verify_cell_multi_llm` - multi-LLM consensus verification
- `verify_digit_rotation_consensus` - rotation-based disambiguation
- `embed_image` - generate vector embeddings
- `analyze` - high-level scene understanding (uses LLM vision)

## Accuracy & Results

**Complete Workflow Achievement: 98.8% accuracy** on Sudoku extraction using:
- Grid detection + OCR (93.8% baseline)
- Domain validation (Sudoku rules)
- Multi-LLM consensus (OpenAI + Anthropic + Google voting)
- Smart rotation testing (6 vs 9 disambiguation)
- Cost-optimized (only 8.6% of cells needed LLM verification)

See `examples/07_complete_workflows/sudoku_extraction.py` for the full implementation.

## Key Features

### 🎯 Multi-LLM Consensus
Combine multiple Vision LLMs for higher accuracy:
- **OpenAI GPT-4o**: Fast, excellent for printed text
- **Anthropic Claude 3.5 Sonnet**: Superior reasoning
- **Google Gemini 2.0 Flash**: Cost-effective

Use voting consensus to reduce model bias and increase accuracy.

### 🔄 Rotation Testing
Automatically disambiguate orientation-dependent characters:
- **6 vs 9** (180° rotation)
- **b vs q, d vs p** (letters)
- **Directional symbols**

### 🧩 Grid Processing
Specialized tools for grid-based content:
- **Auto-detect grid structure** (rows, cols, cell boundaries)
- **Extract all cells** with automatic preprocessing
- **Classify cell content** (empty, digit, text, checkbox)
- **Domain validation** (Sudoku, crossword, table constraints)

### 📊 Multi-Backend OCR
Ensemble voting across OCR engines:
- **PaddleOCR**: Best accuracy, GPU-accelerated
- **Tesseract**: CPU-friendly fallback

Voting improves accuracy by combining strengths of different engines.

## Installation

```bash
uv pip install chuk-mcp-vision
```

Or with development tools:
```bash
uv pip install chuk-mcp-vision[dev]
```

## Usage

### With Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "vision": {
      "command": "uv",
      "args": ["run", "chuk-mcp-vision"]
    }
  }
}
```

### Standalone

```bash
# Run in STDIO mode (default)
uv run chuk-mcp-vision

# Run in HTTP mode for web integration
uv run chuk-mcp-vision http
```

## Example Workflows

### 1. Basic Grid Extraction (93.8% accuracy)
```python
from chuk_mcp_vision.tools.grid import detect_grid, extract_grid_cells, classify_cell_content

# Detect grid structure
grid = await detect_grid(image_uri)

# Extract all cells
cells = await extract_grid_cells(image_uri, grid, preprocess=True)

# Classify each cell
for cell in cells.cells:
    content = await classify_cell_content(cell.image_uri, expected_type="digit")
    print(f"Cell [{cell.row},{cell.col}]: {content.value}")
```

### 2. Multi-Backend OCR (95% accuracy)
```python
from chuk_mcp_vision.tools.ocr import ocr_multi_backend
from chuk_mcp_vision.models import OCRBackend

# Run multiple OCR engines and vote
result = await ocr_multi_backend(
    image_uri,
    backends=[OCRBackend.PADDLEOCR, OCRBackend.TESSERACT]
)

print(f"Consensus: {result.consensus_text}")
print(f"Confidence: {result.consensus_confidence}")
```

### 3. Multi-LLM Consensus (98.8% accuracy)
```python
from chuk_mcp_vision.tools.llm.consensus import verify_cell_multi_llm
from chuk_mcp_vision.models import VisionLLMProvider

# Query multiple Vision LLMs and vote
result = await verify_cell_multi_llm(
    cell_uri,
    providers=[
        VisionLLMProvider.OPENAI,
        VisionLLMProvider.ANTHROPIC,
        VisionLLMProvider.GOOGLE
    ],
    expected_type="digit"
)

print(f"Consensus: {result['consensus_value']}")
print(f"Confidence: {result['consensus_confidence']}")
```

### 4. Complete Workflow with Rotation (99%+ accuracy)
```python
from chuk_mcp_vision.tools.grid import detect_grid, extract_grid_cells
from chuk_mcp_vision.tools.llm.consensus import verify_digit_rotation_consensus

# Detect and extract grid
grid = await detect_grid(image_uri)
cells = await extract_grid_cells(image_uri, grid)

# For suspicious cells, use rotation + multi-LLM consensus
result = await verify_digit_rotation_consensus(
    cell_uri,
    angles=[0, 180],  # Test 6 vs 9
    providers=["openai", "anthropic", "google"]
)

print(f"Value: {result['recommended_value']}")
print(f"Orientation: {result['recommended_angle']}°")
print(f"Confidence: {result['confidence']}")
```

## Tools Reference

### Grid Tools

**`detect_grid(image_uri)`**
- Detects grid structure (rows, cols, cell boundaries)
- Returns: `GridStructure` with boundary, row/col dividers

**`extract_grid_cells(image_uri, grid, margin=5, preprocess=True)`**
- Extracts all cells from detected grid
- Returns: `ExtractGridCellsResponse` with cell images and locations

**`classify_cell_content(cell_uri, expected_type="auto")`**
- Classifies cell content (empty, digit, text, checkbox)
- Returns: `CellContent` with type, value, confidence

### OCR Tools

**`ocr_region(image_uri, backend=OCRBackend.PADDLEOCR, language="en")`**
- Single-backend OCR
- Returns: `OCRRegionResponse` with text, lines, words, confidence

**`ocr_multi_backend(image_uri, backends=None, language="en")`**
- Multi-backend ensemble OCR with voting
- Returns: `MultiBackendOCRResponse` with consensus results

### Vision LLM Tools

**`verify_cell_multi_llm(cell_uri, providers=None, expected_type="digit")`**
- Multi-LLM consensus verification
- Returns: consensus value, confidence, voting details

**`verify_digit_rotation_consensus(cell_uri, angles=[0, 180], providers=None)`**
- Rotation testing + multi-LLM consensus
- Returns: recommended value, angle, confidence

### Transform Tools

**`perspective_transform(image_uri, x1, y1, x2, y2, x3, y3, x4, y4, width, height)`**
- Correct skew and perspective distortion
- Returns: transformed image URI

**`extract_region(image_uri, x, y, width, height)`**
- Crop and extract region of interest
- Returns: cropped image URI

**`rotate_and_classify(image_uri, angles=None, classifier_callback=None)`**
- Test content at multiple rotations
- Returns: results by angle, recommended orientation

### Detection Tools

**`detect_lines(image_uri, min_line_length=50, threshold=150)`**
- Detect straight lines using Hough transform
- Returns: line segments with coordinates

**`detect_contours(image_uri, threshold=127, min_area=100)`**
- Detect shapes and boundaries
- Returns: contours with points

### Analysis Tools

**`analyze(image_uri, use_vision_model=False)`**
- High-level scene understanding
- Returns: scene type, structure, content hints, suggested tools

**`embed_image(image_uri, backend=EmbeddingBackend.EVA02)`**
- Generate vector embeddings for similarity search
- Returns: embedding vector

## Configuration

### No Magic Strings
All models and backends use enums:

```python
from chuk_mcp_vision.models import (
    VisionLLMProvider,  # OPENAI, ANTHROPIC, GOOGLE
    VisionModel,        # GPT_4O, CLAUDE_3_5_SONNET, GEMINI_2_0_FLASH
    OCRBackend,         # PADDLEOCR, TESSERACT
    EmbeddingBackend    # EVA02, JINA_CLIP
)

# Configurable models
await verify_cell_multi_llm(
    cell_uri,
    providers=[VisionLLMProvider.OPENAI],
    model_config={
        "openai": VisionModel.GPT_4O_MINI.value  # Use cheaper model
    }
)
```

### Environment Variables
```bash
# Vision LLM API Keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."
```

## Examples

Comprehensive examples in `examples/` directory:

- **`01_basics/`**: Core vision primitives (lines, contours, OCR, embeddings)
- **`02_transforms/`**: Image transformations (crop, rotate, perspective)
- **`03_preprocessing/`**: Image enhancement pipelines
- **`04_grids/`**: Grid detection and cell extraction (93.8% accuracy)
- **`05_ocr/`**: Multi-backend OCR with voting (95% accuracy)
- **`06_llm/`**: Vision LLM consensus and rotation testing
- **`07_complete_workflows/`**: Full Sudoku extraction (98.8% accuracy)

Run examples:
```bash
# Basic examples
uv run python examples/01_basics/detect_lines.py
uv run python examples/01_basics/ocr_basic.py

# Grid processing
uv run python examples/04_grids/full_pipeline.py

# Multi-LLM consensus
uv run python examples/06_llm/single_llm.py
uv run python examples/06_llm/multi_llm_consensus.py

# Complete workflow
uv run python examples/07_complete_workflows/sudoku_extraction.py
```

## Integration with CHUK Ecosystem

- **chuk-mcp-server**: Base MCP framework with @tool decorator
- **chuk-artifacts**: Stores images, embeddings, and OCR results
- **chuk-ai-planner**: Orchestrates vision tools based on scene analysis
- **chuk-mcp-solver**: Consumes structured data (Sudoku grids, etc.)

## Development

```bash
# Install development dependencies
make dev-install

# Run tests
make test

# Run all checks (lint, typecheck, security, tests)
make check

# Format code
make format

# Build
make build
```

## Code Quality

- ✅ **Linting**: Ruff (all checks passing)
- ✅ **Type checking**: MyPy (all checks passing)
- ✅ **Security**: Bandit (no issues found)
- ✅ **Tests**: Pytest (all passing)
- ✅ **Formatting**: Black/Ruff (consistent style)

## License

MIT License - see LICENSE file for details.

## Links

- **GitHub**: https://github.com/chrishayuk/chuk-mcp-vision
- **PyPI**: https://pypi.org/project/chuk-mcp-vision/
- **Model Context Protocol**: https://modelcontextprotocol.io

---

**Built for the CHUK ecosystem - composable AI tools for complex workflows**

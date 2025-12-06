# Grids - Grid Detection & Cell Extraction

Generic grid detection and cell extraction tools for tables, forms, Sudoku, spreadsheets, and any grid-based layout.

## Overview

Extract structured data from grid-based images:
1. **Detect grid structure** (rows, columns, cell boundaries)
2. **Extract individual cells** as separate images
3. **Classify cell content** (empty, filled, text, checkbox, etc.)

## Examples

| File | Description | Accuracy | Prerequisites |
|------|-------------|----------|---------------|
| `full_pipeline.py` | Complete grid detection and cell extraction | 98.8% | None |

## Quick Start

```bash
uv run python 04_grids/full_pipeline.py
```

**Expected output**:
- Grid dimensions (9x9 for Sudoku)
- Cell coordinates and bounding boxes
- Cell content classification
- Extracted cell images saved as artifacts

## Tools Reference

### 1. detect_grid
Detect grid structure using line detection.

```python
from chuk_mcp_vision.tools.grid import detect_grid

grid = await detect_grid(image_uri)

print(grid.rows)          # Number of rows
print(grid.cols)          # Number of columns
print(grid.cell_width)    # Average cell width
print(grid.cell_height)   # Average cell height
```

**How it works**:
1. Detect horizontal and vertical lines using Hough Line Transform
2. Find line intersections to determine grid structure
3. Calculate cell boundaries
4. Return grid metadata

**Works with**:
- Sudoku grids
- Tables with borders
- Spreadsheets
- Forms with grid layout
- Crossword puzzles
- Bingo cards

### 2. extract_grid_cells
Extract individual cells from detected grid.

```python
from chuk_mcp_vision.tools.grid import extract_grid_cells

cells = await extract_grid_cells(image_uri, grid)

for cell in cells.cells:
    print(f"Cell [{cell.row},{cell.col}]: {cell.image_uri}")
```

**Returns**:
- `cells`: List of GridCell objects
- Each cell contains:
  - `row`, `col`: Position in grid
  - `x`, `y`, `width`, `height`: Bounding box
  - `image_uri`: Extracted cell image

**Best practices**:
- Apply preprocessing to source image first
- Cells are extracted with small padding
- Cell images saved as artifacts

### 3. classify_cell_content
Classify what's in each cell.

```python
from chuk_mcp_vision.tools.grid import classify_cell_content

classification = await classify_cell_content(cell.image_uri)

print(classification.content_type)  # "digit", "text", "empty", "checkbox", etc.
print(classification.value)         # OCR'd text if applicable
print(classification.confidence)    # Classification confidence
```

**Content types**:
- `empty`: Blank cell
- `digit`: Single digit (0-9)
- `text`: Text content
- `checkbox`: Checkbox (checked/unchecked)
- `image`: Non-text content
- `mixed`: Multiple content types

**Uses**:
- OCR for text detection
- Contour analysis for empty vs filled
- Edge detection for checkboxes
- Pixel intensity for blank detection

## Full Pipeline Example

```python
from chuk_mcp_vision.tools.grid import (
    detect_grid,
    extract_grid_cells,
    classify_cell_content
)

# 1. Detect grid
grid = await detect_grid(image_uri)
print(f"Detected {grid.rows}x{grid.cols} grid")

# 2. Extract cells
cells = await extract_grid_cells(image_uri, grid)
print(f"Extracted {len(cells.cells)} cells")

# 3. Classify each cell
for cell in cells.cells:
    classification = await classify_cell_content(cell.image_uri)

    if classification.content_type != "empty":
        print(f"[{cell.row},{cell.col}]: {classification.value}")
```

## Use Cases

### Sudoku Extraction
```python
# Extract all digits from Sudoku grid
grid = await detect_grid(sudoku_image_uri)
cells = await extract_grid_cells(sudoku_image_uri, grid)

sudoku_data = [[None for _ in range(9)] for _ in range(9)]

for cell in cells.cells:
    classification = await classify_cell_content(cell.image_uri)
    if classification.content_type == "digit":
        sudoku_data[cell.row][cell.col] = classification.value
```

### Table Extraction
```python
# Extract data from table
grid = await detect_grid(table_image_uri)
cells = await extract_grid_cells(table_image_uri, grid)

table_data = []
for row in range(grid.rows):
    row_data = []
    for col in range(grid.cols):
        cell = cells.cells[row * grid.cols + col]
        classification = await classify_cell_content(cell.image_uri)
        row_data.append(classification.value)
    table_data.append(row_data)
```

### Form Processing
```python
# Extract checkbox states from form
grid = await detect_grid(form_image_uri)
cells = await extract_grid_cells(form_image_uri, grid)

checkboxes = {}
for cell in cells.cells:
    classification = await classify_cell_content(cell.image_uri)
    if classification.content_type == "checkbox":
        checkboxes[f"{cell.row},{cell.col}"] = classification.value == "checked"
```

## Advanced Techniques

### Preprocessing Before Grid Detection
For better accuracy, preprocess images first:

```python
from chuk_mcp_vision.tools.preprocess import (
    enhance_contrast,
    sharpen_image
)

# Enhance grid lines
enhanced = await enhance_contrast(image_uri, method="clahe")
sharpened = await sharpen_image(enhanced.output_image_uri)

# Now detect grid
grid = await detect_grid(sharpened.output_image_uri)
```

### Multi-Backend OCR for Cells
Use OCR voting for higher accuracy:

```python
from chuk_mcp_vision.tools.ocr import ocr_multi_backend
from chuk_mcp_vision.models import OCRBackend, VotingStrategy

for cell in cells.cells:
    result = await ocr_multi_backend(
        cell.image_uri,
        backends=[
            OCRBackend.PADDLEOCR,
            OCRBackend.TESSERACT,
            OCRBackend.EASYOCR
        ],
        voting_strategy=VotingStrategy.CONFIDENCE_WEIGHTED
    )
    print(f"Cell [{cell.row},{cell.col}]: {result.consensus_text}")
```

### LLM Verification
Verify ambiguous cells with Vision LLMs:

```python
from chuk_mcp_vision.tools.llm import verify_cell_multi_llm

for cell in cells.cells:
    # Get initial classification
    classification = await classify_cell_content(cell.image_uri)

    # Verify with LLMs if confidence is low
    if classification.confidence < 0.9:
        llm_result = await verify_cell_multi_llm(
            cell.image_uri,
            providers=["openai", "anthropic", "google"],
            expected_type="digit"
        )
        print(f"LLM consensus: {llm_result['consensus']}")
```

## Troubleshooting

**Grid not detected**:
- Check if image has clear grid lines
- Try preprocessing (contrast, sharpen)
- Manually adjust Hough Line parameters

**Cells extracted incorrectly**:
- Grid lines may be too faint
- Try thresholding before detection
- Check for warped/skewed image (use perspective_transform)

**Classification errors**:
- Preprocess cells before classification
- Use multi-backend OCR voting
- Verify with LLM for ambiguous content

## Related Tools

- See `../03_preprocessing/` for image enhancement
- See `../05_ocr/` for better text recognition
- See `../06_llm/` for verification of ambiguous content
- See `../07_complete_workflows/` for complete examples

## Next Steps

1. Run `full_pipeline.py` to see grid extraction in action
2. Try on your own grid-based images
3. Combine with preprocessing for better accuracy
4. Add OCR voting for text extraction
5. Use LLM verification for critical data

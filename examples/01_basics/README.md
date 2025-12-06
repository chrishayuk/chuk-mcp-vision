# Basics - Tool Fundamentals

Learn individual vision tools by seeing them in action on simple examples.

## Overview

Each example demonstrates a single tool in isolation, showing:
- What the tool does
- How to use it
- What the output looks like
- Common use cases

## Examples

| File | Tool | Description |
|------|------|-------------|
| `detect_lines.py` | detect_lines | Hough Line Transform for line detection |
| `detect_contours.py` | detect_contours | Contour detection for shapes and boundaries |
| `ocr_basic.py` | ocr_region | Single-backend OCR text extraction |
| `embeddings.py` | embed_image | Generate vector embeddings for similarity |

## Quick Start

```bash
# Run all examples in order
uv run python 01_basics/detect_lines.py
uv run python 01_basics/detect_contours.py
uv run python 01_basics/ocr_basic.py
uv run python 01_basics/embeddings.py
```

## Tool Reference

### detect_lines
**Purpose**: Detect straight lines in images using Hough Line Transform

```python
from chuk_mcp_vision.tools.detection.lines import detect_lines

result = await detect_lines(image_uri)

print(f"Horizontal lines: {len(result.horizontal_lines)}")
print(f"Vertical lines: {len(result.vertical_lines)}")
```

**Use cases**:
- Grid structure detection (tables, Sudoku)
- Document layout analysis
- Border detection
- Horizon/vanishing point detection

### detect_contours
**Purpose**: Find boundaries and shapes in images

```python
from chuk_mcp_vision.tools.detection.contours import detect_contours

result = await detect_contours(
    image_uri,
    min_area=100,  # Filter small contours
    mode="external"  # Only external boundaries
)

print(f"Contours found: {result.count}")
for contour in result.contours:
    print(f"Area: {contour.area}, Bbox: {contour.bbox}")
```

**Use cases**:
- Object segmentation
- Shape detection
- Component counting
- Region of interest extraction

### ocr_region
**Purpose**: Extract text from images

```python
from chuk_mcp_vision.tools.ocr import ocr_region
from chuk_mcp_vision.models import OCRBackend

result = await ocr_region(
    image_uri,
    backend=OCRBackend.PADDLEOCR,  # or TESSERACT, EASYOCR
    language="en"
)

print(f"Text: {result.full_text}")
print(f"Confidence: {result.mean_confidence:.2%}")
print(f"Words: {len(result.words)}")
```

**Backends**:
- **PaddleOCR** (recommended): Best accuracy and speed
- **Tesseract**: Good for clean printed text
- **EasyOCR**: Best for handwriting and Asian scripts

**Use cases**:
- Document digitization
- Receipt/invoice processing
- License plate recognition
- Text extraction from images

### embed_image
**Purpose**: Generate vector embeddings for image similarity

```python
from chuk_mcp_vision.tools.embeddings import embed_image

result = await embed_image(image_uri)

print(f"Dimensions: {result.dimension}")
print(f"Embedding: {result.embedding}")  # List of floats
```

**Note**: Currently returns placeholder vectors (TODO: implement EVA-02/Jina CLIP)

**Use cases**:
- Image similarity search
- Duplicate detection
- Image clustering
- Visual search engines
- Content-based retrieval

## Learning Path

1. **Start with detect_lines.py** - Understand line detection basics
2. **Move to detect_contours.py** - Learn about shape detection
3. **Try ocr_basic.py** - See text extraction in action
4. **Explore embeddings.py** - Understand vector representations

## Combining Tools

These basic tools are building blocks for more complex workflows:

```python
# Example: Find text in specific regions
# 1. Detect contours to find regions
contours = await detect_contours(image_uri)

# 2. Extract largest contour (main document)
largest = max(contours.contours, key=lambda c: c.area)

# 3. Extract that region
from chuk_mcp_vision.tools.transform import extract_region
region = await extract_region(
    image_uri,
    x=largest.bbox.x,
    y=largest.bbox.y,
    width=largest.bbox.width,
    height=largest.bbox.height
)

# 4. OCR the extracted region
text = await ocr_region(region.output_image_uri)
```

## Tips

1. **Preprocessing helps**: Enhance images before processing (see `03_preprocessing/`)
2. **Start simple**: Test on clean images first
3. **Adjust parameters**: Each tool has tunable parameters
4. **Check confidence**: OCR and other tools return confidence scores
5. **Chain tools**: Combine multiple tools for better results

## Related Examples

- See `../02_transforms/` for image transformations
- See `../03_preprocessing/` for image enhancement
- See `../04_grids/` for grid-specific workflows
- See `../05_ocr/` for advanced OCR with voting

## Next Steps

After mastering these basics:
1. Learn image transforms in `02_transforms/`
2. Improve accuracy with preprocessing in `03_preprocessing/`
3. Build complete workflows in `04_grids/` and `07_complete_workflows/`

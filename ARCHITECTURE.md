# chuk-mcp-vision Architecture

## Design Philosophy

**Atomic, Generic Tools** - This MCP server provides primitive vision operations, not domain-specific solutions. The intelligence of "how to solve Sudoku" or "how to scan a receipt" lives in the planner/orchestration layer (chuk-ai-planner), not in the tools.

**One Tool, One Job** - Each tool does exactly one thing well. Complex workflows emerge from composition, not from fat multi-purpose tools.

**Backend Flexibility** - Where appropriate (OCR, embeddings), tools support multiple backend implementations via a `backend` parameter, allowing users to choose speed vs accuracy tradeoffs.

## Project Structure

```
chuk-mcp-vision/
├── src/chuk_mcp_vision/
│   ├── __init__.py
│   ├── server.py              # Main MCP server entry point
│   ├── models.py              # Pydantic models (no dictionaries!)
│   └── tools/                 # One folder per tool category
│       ├── __init__.py        # Exports all tools
│       ├── common.py          # Shared utilities (load_image, save_image)
│       ├── analyze/           # High-level scene analysis
│       │   └── tool.py
│       ├── detect/            # Detection primitives
│       │   ├── lines.py       # Hough line detection
│       │   └── contours.py    # Contour/shape detection
│       ├── transform/         # Geometric transforms
│       │   └── perspective.py # Perspective correction
│       ├── extract/           # Region extraction
│       │   └── region.py      # Simple cropping
│       ├── ocr/               # Text recognition
│       │   └── tool.py        # Multi-backend OCR
│       └── embedding/         # Vector embeddings
│           └── tool.py        # Multi-backend embedding
├── tests/
│   ├── __init__.py
│   └── test_models.py         # Basic model tests
├── Makefile                   # Standard CHUK Makefile
├── pyproject.toml
├── README.md
└── LICENSE
```

## Tool Categories

### 1. analyze
**Purpose:** High-level scene understanding using vision models or CV heuristics.

**Output:** Scene type, structure, content hints, suggested next tools.

**When to use:** First call when encountering a new image.

### 2. detect
**Purpose:** Find low-level geometric features (lines, contours, shapes).

**Tools:**
- `detect_lines` - Hough line transform for grids, tables, edges
- `detect_contours` - Shape/boundary detection for objects, documents

**When to use:** After analyze suggests grid/table/document structure.

### 3. transform
**Purpose:** Geometric transformations to normalize images.

**Tools:**
- `perspective_transform` - Correct skew/angle using 4 corner points

**When to use:** After detecting document/grid boundaries, before extraction.

### 4. extract
**Purpose:** Crop regions of interest for focused processing.

**Tools:**
- `extract_region` - Simple rectangular crop

**When to use:** Extract individual cells, fields, or sub-regions for OCR/analysis.

### 5. ocr
**Purpose:** Text and digit recognition.

**Backends:**
- `paddleocr` - High accuracy, GPU-accelerated, complex layouts
- `tesseract` - CPU-friendly, simple printed text

**When to use:** After extracting regions with text/digits.

### 6. embedding
**Purpose:** Generate vector representations for similarity search.

**Backends:**
- `eva02` - Pure vision model for image-image similarity
- `jina-clip` - Multimodal for text-image cross-modal search

**When to use:** For visual search, RAG, clustering, recommendations.

## Example: Sudoku Solver Workflow

The planner orchestrates tools to solve a Sudoku puzzle:

```
1. analyze(image)
   → scene_type: PUZZLE
   → structure: "grid_9x9"
   → suggested_tools: ["detect_lines", "extract_region", "ocr_region"]

2. detect_lines(image, min_line_length=100)
   → horizontal_lines: 10
   → vertical_lines: 10
   → lines: [...coordinates...]

3. Calculate grid cell boundaries from line intersections

4. For each of 81 cells:
   extract_region(image, x, y, width, height)
   → cell_image_uri

5. For each cell_image_uri:
   ocr_region(cell_image_uri, backend="paddleocr")
   → full_text: "5" or "" (empty cell)

6. Assemble 9x9 array of digits

7. Pass to chuk-mcp-solver for constraint solving
```

Note: The Sudoku-specific logic (calculating intersections, assembling array) lives in the **planner**, not in vision tools.

## Integration Points

### chuk-ai-planner
- Calls `analyze` first to understand the image
- Selects which tools to use based on suggested_tools
- Orchestrates multi-step pipelines
- Handles tool failures and retries

### chuk-virtual-fs
- Stores intermediate images (extracted regions, transformed images)
- Provides image_uri references between tool calls

### chuk-artifacts
- Stores OCR output as structured data
- Stores embeddings for search

### Qdrant
- Vector search over image embeddings
- Multimodal RAG (text queries → image results)

## Backend Implementation Notes

### OpenCV (cv2)
Used for all classical computer vision:
- Line detection (Hough transform)
- Contour detection
- Perspective transformation
- Edge detection (Canny)
- Image I/O and manipulation

### PaddleOCR
- High-accuracy OCR engine
- Good for complex layouts, handwriting
- GPU-accelerated
- Heavier dependency

### Tesseract (pytesseract)
- Classic OCR engine
- CPU-friendly
- Lighter dependency
- Best for simple printed text

### EVA-02 (TODO)
- Next-gen vision transformer
- Pure image embeddings
- ~1024 dimensions
- Best for image-to-image similarity

### Jina CLIP (TODO)
- Multimodal (text + image)
- ~768 dimensions
- Best for text-to-image search

## Testing Strategy

1. **Unit tests** - Test Pydantic models in isolation
2. **Integration tests** - Test each tool with real images
3. **Pipeline tests** - Test multi-tool workflows (Sudoku, receipt scanning)
4. **Backend tests** - Test backend selection and fallback

## Future Enhancements

1. **Vision model integration** - Replace CV heuristics in `analyze` with GPT-4.1-mini/Claude vision
2. **Implement embedding models** - Add actual EVA-02 and Jina CLIP model loading
3. **Virtual-fs integration** - Store intermediate images in virtual-fs instead of temp
4. **Cloud storage** - Support URIs beyond local file paths
5. **Additional tools**:
   - `detect_text_regions` - Find text bounding boxes
   - `classify_image` - Image classification (for chess pieces, etc.)
   - `detect_objects` - YOLO-style object detection
6. **Streaming** - Support video frame processing

## Dependencies

### Core
- `chuk-mcp-server` - MCP framework
- `pydantic` - Type-safe models
- `opencv-python` - Computer vision primitives
- `numpy` - Array operations
- `pillow` - Image I/O

### OCR
- `paddleocr` - High-accuracy OCR (optional, heavy)
- `pytesseract` - Classic OCR (optional, requires Tesseract binary)

### Embeddings (TODO)
- `transformers` - For loading EVA-02, Jina CLIP
- `torch` - Model inference

## Performance Considerations

- Line/contour detection: ~10-50ms per image (CPU)
- OCR (PaddleOCR): ~100-500ms per region (GPU recommended)
- OCR (Tesseract): ~50-200ms per region (CPU-friendly)
- Embeddings: ~10-100ms per image (GPU recommended)
- Perspective transform: ~5-20ms per image (CPU)

For Sudoku (81 cells):
- Total pipeline: ~5-10 seconds with PaddleOCR on GPU
- Total pipeline: ~10-20 seconds with Tesseract on CPU

## License

MIT - See LICENSE file

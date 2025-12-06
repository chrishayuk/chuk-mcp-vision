# Transforms - Image Transformations

Image transformation tools for geometric manipulations: perspective correction, rotation, and region extraction.

## Overview

Transform tools modify images geometrically while preserving content:
- **Perspective Transform**: Correct skewed/angled photos
- **Rotation**: Handle orientation-dependent content
- **Region Extraction**: Crop specific areas

## Examples

| File | Description | Use Case |
|------|-------------|----------|
| `perspective_transform.py` | Correct perspective distortion | Document scanning, deskewing |
| `rotation.py` | Rotation-based classification | 6/9 disambiguation, orientation |
| `crop_extract.py` | Extract specific regions | Focus on ROI, grid cells |

## Quick Start

```bash
# Run all examples
uv run python 02_transforms/perspective_transform.py
uv run python 02_transforms/rotation.py
uv run python 02_transforms/crop_extract.py
```

## Tool Reference

### perspective_transform
**Purpose**: Correct perspective distortion (deskewing)

```python
from chuk_mcp_vision.tools.transform import perspective_transform

result = await perspective_transform(
    image_uri,
    source_points=[
        [x1, y1],  # Top-left
        [x2, y2],  # Top-right
        [x3, y3],  # Bottom-right
        [x4, y4]   # Bottom-left
    ],
    width=800,
    height=600
)
```

**Use cases**:
- Document scanning (photos of paper)
- Whiteboard capture
- License plate straightening
- Book/magazine scanning

### rotate_and_classify
**Purpose**: Test multiple rotations to find correct orientation

```python
from chuk_mcp_vision.tools.transform import rotate_and_classify

result = await rotate_and_classify(
    image_uri,
    angles=[0, 90, 180, 270]
)

print(f"Recommended: {result.recommended_value} at {result.recommended_angle}°")
```

**Use cases**:
- 6 vs 9 disambiguation
- b/q, d/p character orientation
- Directional symbols
- Auto-rotation of scanned documents

### extract_region
**Purpose**: Crop specific rectangular regions

```python
from chuk_mcp_vision.tools.transform import extract_region

result = await extract_region(
    image_uri,
    x=100,
    y=100,
    width=200,
    height=200
)
```

**Use cases**:
- Focus on region of interest
- Extract form fields
- Grid cell extraction
- Crop to content

## Combining Transforms

Transforms often work together in pipelines:

### Pipeline 1: Document Scanning
```python
# 1. Detect document boundary
contours = await detect_contours(image_uri)
largest = max(contours.contours, key=lambda c: c.area)

# 2. Get corner points
corners = get_corners(largest)  # Helper function

# 3. Apply perspective correction
corrected = await perspective_transform(
    image_uri,
    source_points=corners,
    width=2480,  # A4 at 300 DPI
    height=3508
)

# 4. Extract content region (remove margins)
final = await extract_region(
    corrected.output_image_uri,
    x=100, y=100,
    width=2280, height=3308
)
```

### Pipeline 2: Grid Cell Processing
```python
# 1. Detect grid
grid = await detect_grid(image_uri)

# 2. Extract each cell
for row in range(grid.rows):
    for col in range(grid.cols):
        cell_x = grid.boundary.x + col * grid.cell_width
        cell_y = grid.boundary.y + row * grid.cell_height

        # 3. Extract cell
        cell = await extract_region(
            image_uri,
            x=int(cell_x),
            y=int(cell_y),
            width=int(grid.cell_width),
            height=int(grid.cell_height)
        )

        # 4. Test rotation if needed
        if ambiguous:
            result = await rotate_and_classify(cell.output_image_uri)
```

## Tips

1. **Perspective Transform**:
   - Accurate corner detection is critical
   - Order points correctly (TL, TR, BR, BL)
   - Use contour detection to find boundaries
   - Preprocessing helps (edge detection, thresholding)

2. **Rotation**:
   - Test common angles first (0°, 90°, 180°, 270°)
   - Use geometric analysis (loop position) as fallback
   - Combine with OCR/LLM for verification
   - Cache results to avoid re-processing

3. **Region Extraction**:
   - Validate coordinates are within bounds
   - Add padding to avoid clipping
   - Use contour detection for automatic boundary finding
   - Process extracted regions independently

## Related Examples

- See `../01_basics/` for contour detection (finding regions)
- See `../03_preprocessing/` for enhancing extracted regions
- See `../04_grids/` for grid-specific extraction
- See `../07_complete_workflows/` for complete pipelines

## Next Steps

1. Master perspective correction for document scanning
2. Learn rotation techniques for orientation handling
3. Practice region extraction for focused processing
4. Combine transforms in multi-stage pipelines

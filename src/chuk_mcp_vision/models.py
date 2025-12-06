"""Pydantic models for vision tool inputs and outputs.

All tool inputs/outputs are properly typed using Pydantic models for type safety,
validation, and better IDE support. No dictionaries, no magic strings.
"""

from enum import Enum
from typing import Optional, Union

from pydantic import BaseModel, Field


# Enums for backend selection
class OCRBackend(str, Enum):
    """OCR engine backend options."""

    PADDLEOCR = "paddleocr"
    TESSERACT = "tesseract"


class EmbeddingBackend(str, Enum):
    """Image embedding model backend options."""

    EVA02 = "eva02"
    JINA_CLIP = "jina-clip"


class SceneType(str, Enum):
    """Types of scenes that can be detected."""

    PUZZLE = "puzzle"
    DOCUMENT = "document"
    PHOTO = "photo"
    SCREENSHOT = "screenshot"
    CHART = "chart"
    TABLE = "table"
    FORM = "form"
    UI = "ui"
    CODE = "code"
    UNKNOWN = "unknown"


class CellContentType(str, Enum):
    """Types of content that can be found in grid cells."""

    EMPTY = "empty"
    DIGIT = "digit"
    TEXT = "text"
    CHECKBOX = "checkbox"
    IMAGE = "image"
    UNKNOWN = "unknown"


class ContentType(str, Enum):
    """Types of content that can be detected in images."""

    DIGITS = "digits"
    TEXT = "text"
    HANDWRITTEN = "handwritten"
    PRINTED = "printed"
    SYMBOLS = "symbols"
    SHAPES = "shapes"
    GRID = "grid"


class VisionLLMProvider(str, Enum):
    """Vision LLM provider options."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


class VisionModel(str, Enum):
    """Vision model options (latest recommended models)."""

    # OpenAI models
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"

    # Anthropic models
    CLAUDE_3_5_SONNET = "claude-3-5-sonnet-20241022"
    CLAUDE_3_OPUS = "claude-3-opus-20240229"
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"

    # Google models
    GEMINI_2_0_FLASH = "gemini-2.0-flash-exp"
    GEMINI_1_5_PRO = "gemini-1.5-pro"


# Line detection models
class LineSegment(BaseModel):
    """A detected line segment with start and end points."""

    x1: float = Field(..., description="X coordinate of start point")
    y1: float = Field(..., description="Y coordinate of start point")
    x2: float = Field(..., description="X coordinate of end point")
    y2: float = Field(..., description="Y coordinate of end point")
    angle: Optional[float] = Field(None, description="Line angle in degrees (0-180)")
    length: Optional[float] = Field(None, description="Line length in pixels")


class DetectLinesResponse(BaseModel):
    """Response from detect_lines tool."""

    lines: list[LineSegment] = Field(..., description="Detected line segments")
    image_width: int = Field(..., description="Original image width")
    image_height: int = Field(..., description="Original image height")
    horizontal_lines: int = Field(..., description="Count of horizontal lines")
    vertical_lines: int = Field(..., description="Count of vertical lines")


# Contour detection models
class Point(BaseModel):
    """A 2D point."""

    x: float
    y: float


class Contour(BaseModel):
    """A detected contour (shape boundary)."""

    points: list[Point] = Field(..., description="Points defining the contour")
    area: float = Field(..., description="Area enclosed by contour in square pixels")
    perimeter: float = Field(..., description="Perimeter length in pixels")
    bounding_box: dict[str, float] = Field(..., description="Bounding box {x, y, width, height}")
    is_closed: bool = Field(True, description="Whether the contour is closed")


class DetectContoursResponse(BaseModel):
    """Response from detect_contours tool."""

    contours: list[Contour] = Field(..., description="Detected contours")
    image_width: int
    image_height: int
    total_contours: int = Field(..., description="Total number of contours found")


# Perspective transform models
class PerspectiveTransformResponse(BaseModel):
    """Response from perspective_transform tool."""

    output_image_uri: str = Field(
        ..., description="URI of transformed image (in virtual-fs or temp storage)"
    )
    output_width: int = Field(..., description="Width of output image")
    output_height: int = Field(..., description="Height of output image")
    transform_matrix: Optional[list[list[float]]] = Field(
        None, description="3x3 transformation matrix (for debugging)"
    )


# Region extraction models
class BoundingBox(BaseModel):
    """A rectangular bounding box."""

    x: float = Field(..., description="Top-left X coordinate")
    y: float = Field(..., description="Top-left Y coordinate")
    width: float = Field(..., description="Width of box")
    height: float = Field(..., description="Height of box")


class ExtractRegionResponse(BaseModel):
    """Response from extract_region tool."""

    output_image_uri: str = Field(..., description="URI of extracted region image (in virtual-fs)")
    region: BoundingBox = Field(..., description="The extracted region coordinates")
    output_width: int
    output_height: int


# OCR models
class OCRWord(BaseModel):
    """A single word/text detection from OCR."""

    text: str = Field(..., description="Detected text content")
    confidence: float = Field(..., description="Detection confidence (0.0-1.0)")
    bounding_box: BoundingBox = Field(..., description="Location of text in image")


class OCRLine(BaseModel):
    """A line of text from OCR."""

    text: str = Field(..., description="Full line text")
    words: list[OCRWord] = Field(..., description="Individual words in line")
    bounding_box: BoundingBox


class OCRRegionResponse(BaseModel):
    """Response from ocr_region tool."""

    full_text: str = Field(..., description="All detected text concatenated")
    lines: list[OCRLine] = Field(..., description="Lines of text with positions")
    backend_used: OCRBackend = Field(..., description="OCR backend that was used")
    language: str = Field(default="en", description="Detected or specified language")
    mean_confidence: float = Field(..., description="Average confidence across all detections")


class OCRBackendResult(BaseModel):
    """Result from a single OCR backend."""

    backend: str = Field(..., description="Backend name")
    text: str = Field(..., description="Extracted text")
    confidence: float = Field(..., description="Backend confidence score")
    success: bool = Field(..., description="Whether OCR succeeded")
    error: Optional[str] = Field(None, description="Error message if failed")


class MultiBackendOCRResponse(BaseModel):
    """Response from ocr_multi_backend tool."""

    results: list[OCRBackendResult] = Field(..., description="Results from each backend")
    consensus_text: str = Field(..., description="Text agreed upon by majority vote")
    consensus_confidence: float = Field(..., description="Confidence in consensus result")
    voting_details: dict = Field(..., description="Breakdown of voting results")
    disagreements: list[dict] = Field(..., description="Where backends disagree")
    total_backends: int = Field(..., description="Total backends attempted")
    successful_backends: int = Field(..., description="Number of successful backends")


# Image embedding models
class EmbedImageResponse(BaseModel):
    """Response from embed_image tool."""

    embedding: list[float] = Field(..., description="High-dimensional vector embedding")
    dimension: int = Field(..., description="Embedding dimension (e.g., 1024, 768)")
    backend_used: EmbeddingBackend = Field(..., description="Embedding model used")
    model_name: str = Field(..., description="Specific model name/version")


# Scene analysis models
class RegionOfInterest(BaseModel):
    """A region of interest detected in the image."""

    label: str = Field(..., description="What this region contains")
    bounding_box: BoundingBox
    confidence: float = Field(..., description="Detection confidence (0.0-1.0)")


class AnalyzeResponse(BaseModel):
    """Response from analyze tool (high-level scene understanding)."""

    scene_type: SceneType = Field(..., description="Type of scene detected")
    structure: Optional[str] = Field(
        None, description="Structure description (e.g., 'grid_9x9', 'table_3x4')"
    )
    content_hints: list[ContentType] = Field(
        default_factory=list, description="Types of content detected"
    )
    suggested_tools: list[str] = Field(
        default_factory=list, description="Recommended vision tools to use next"
    )
    regions_of_interest: list[RegionOfInterest] = Field(
        default_factory=list, description="Important regions to focus on"
    )
    description: str = Field(..., description="Natural language description of the image")
    confidence: float = Field(..., description="Overall analysis confidence (0.0-1.0)")


# Grid detection models
class GridStructure(BaseModel):
    """Detected grid structure with row/column divisions."""

    boundary: BoundingBox = Field(..., description="Outer grid boundary")
    rows: int = Field(..., description="Number of rows")
    cols: int = Field(..., description="Number of columns")
    row_dividers: list[float] = Field(..., description="Y-coordinates of horizontal dividing lines")
    col_dividers: list[float] = Field(..., description="X-coordinates of vertical dividing lines")
    cell_width: float = Field(..., description="Average cell width in pixels")
    cell_height: float = Field(..., description="Average cell height in pixels")
    is_regular: bool = Field(..., description="Whether grid has uniform cell sizes")
    confidence: float = Field(..., description="Detection confidence (0.0-1.0)")


class CellLocation(BaseModel):
    """Location and image URI of a grid cell."""

    row: int = Field(..., description="Row index (0-based)")
    col: int = Field(..., description="Column index (0-based)")
    image_uri: str = Field(..., description="URI to cell image artifact")
    x: float = Field(..., description="X coordinate in original image")
    y: float = Field(..., description="Y coordinate in original image")
    width: float = Field(..., description="Cell width")
    height: float = Field(..., description="Cell height")


class ExtractGridCellsResponse(BaseModel):
    """Response from extract_grid_cells tool."""

    cells: list[CellLocation] = Field(..., description="List of all extracted cells")
    total_cells: int = Field(..., description="Total number of cells")
    grid_size: tuple[int, int] = Field(..., description="Grid dimensions (rows, cols)")


class CellContent(BaseModel):
    """Classified content of a grid cell."""

    content_type: CellContentType = Field(..., description="Type of content detected")
    value: Optional[Union[int, str, bool]] = Field(
        None, description="Extracted value (digit, text, or boolean)"
    )
    confidence: float = Field(..., description="Classification confidence (0.0-1.0)")
    is_empty: bool = Field(..., description="Whether cell is empty")
    raw_text: Optional[str] = Field(None, description="Raw OCR text before processing")
    metadata: Optional[dict] = Field(default_factory=dict, description="Additional metadata")  # type: ignore[arg-type]


class DigitClassificationResult(BaseModel):
    """Result from KNN digit classification."""

    digit: Optional[int] = Field(None, description="Predicted digit (1-9) or None if empty")
    confidence: float = Field(..., description="Classification confidence (0.0-1.0)")
    method: str = Field(..., description="Classification method used")
    is_empty: bool = Field(..., description="Whether cell appears empty")
    neighbors_votes: Optional[dict] = Field(None, description="Voting breakdown from k neighbors")
    distances: Optional[list[float]] = Field(None, description="Distances to k nearest neighbors")


# Rotation and orientation models
class OrientationResult(BaseModel):
    """Classification result at a specific rotation angle."""

    angle: int = Field(..., description="Rotation angle in degrees")
    rotated_image_uri: str = Field(..., description="URI to rotated image")
    classification_value: Optional[str] = Field(None, description="Classified value at this angle")
    confidence: float = Field(..., description="Classification confidence")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class RotateAndClassifyResponse(BaseModel):
    """Response from rotate_and_classify tool."""

    results_by_angle: list[OrientationResult] = Field(
        ..., description="Results at each rotation angle"
    )
    recommended_angle: int = Field(..., description="Most likely correct orientation (degrees)")
    recommended_value: Optional[str] = Field(None, description="Most likely correct value")
    confidence: float = Field(..., description="Overall confidence in recommendation")
    disambiguation: str = Field(
        ..., description="What orientations disambiguate (e.g., '6↔9 at 180°')"
    )


# Preprocessing models
class PreprocessResult(BaseModel):
    """Result from preprocessing operation."""

    output_image_uri: str = Field(..., description="URI of preprocessed image")
    operation: str = Field(..., description="Preprocessing operation applied")
    parameters: dict = Field(default_factory=dict, description="Parameters used")
    improvement_estimate: Optional[float] = Field(
        None, description="Estimated quality improvement (0.0-1.0)"
    )
    metadata: dict = Field(default_factory=dict, description="Additional metadata")

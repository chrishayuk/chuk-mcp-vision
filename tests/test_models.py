"""Test Pydantic models."""

from chuk_mcp_vision.models import (
    BoundingBox,
    DetectLinesResponse,
    LineSegment,
    OCRBackend,
    SceneType,
)


def test_line_segment():
    """Test LineSegment model."""
    line = LineSegment(x1=0.0, y1=0.0, x2=100.0, y2=100.0, angle=45.0, length=141.4)
    assert line.x1 == 0.0
    assert line.angle == 45.0


def test_detect_lines_response():
    """Test DetectLinesResponse model."""
    line = LineSegment(x1=0.0, y1=0.0, x2=100.0, y2=0.0)
    response = DetectLinesResponse(
        lines=[line], image_width=800, image_height=600, horizontal_lines=1, vertical_lines=0
    )
    assert len(response.lines) == 1
    assert response.horizontal_lines == 1


def test_bounding_box():
    """Test BoundingBox model."""
    bbox = BoundingBox(x=10.0, y=20.0, width=100.0, height=50.0)
    assert bbox.x == 10.0
    assert bbox.width == 100.0


def test_ocr_backend_enum():
    """Test OCRBackend enum."""
    assert OCRBackend.PADDLEOCR == "paddleocr"
    assert OCRBackend.TESSERACT == "tesseract"


def test_scene_type_enum():
    """Test SceneType enum."""
    assert SceneType.PUZZLE == "puzzle"
    assert SceneType.DOCUMENT == "document"

"""Image analysis tool - high-level scene understanding."""

import math
import os

import cv2
import numpy as np
from chuk_mcp_server import tool

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # dotenv is optional

from ...models import AnalyzeResponse, ContentType, SceneType
from ..common import load_image_from_artifact


async def _analyze_with_vision_llm(image_uri: str) -> AnalyzeResponse:
    """Analyze image using vision LLM (GPT-4o-mini or Claude Haiku).

    This provides much better scene understanding than CV heuristics.
    For Sudoku, a vision LLM can directly read all digits with 99%+ accuracy.
    """
    import base64
    import json
    import httpx

    # Load and encode image
    img = await load_image_from_artifact(image_uri)
    success, buffer = cv2.imencode(".jpg", img)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    prompt = """Analyze this image and determine what type of content it contains.

Return a JSON object with:
{
  "scene_type": "puzzle" | "document" | "photo" | "table" | "unknown",
  "structure": "description of structure (e.g., 'grid_9x9', 'table', 'text_document')",
  "content_hints": ["list", "of", "content", "types"],  // e.g., ["digits", "text", "grid"]
  "description": "brief description of what you see",
  "confidence": 0.95  // 0-1
}

If this is a Sudoku puzzle, also include:
{
  "sudoku_grid": [[...], [...], ...]  // 9x9 grid with integers 1-9 or null for empty
}
"""

    # Try OpenAI first, then Anthropic
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    try:
        if os.environ.get("OPENAI_API_KEY"):
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{img_base64}"
                                        },
                                    },
                                ],
                            }
                        ],
                        "max_tokens": 2000,
                    },
                )
                result = response.json()
                content = result["choices"][0]["message"]["content"]
        else:
            # Anthropic
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                    },
                    json={
                        "model": "claude-3-haiku-20240307",
                        "max_tokens": 2000,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": "image/jpeg",
                                            "data": img_base64,
                                        },
                                    },
                                    {"type": "text", "text": prompt},
                                ],
                            }
                        ],
                    },
                )
                result = response.json()
                content = result["content"][0]["text"]

        # Parse JSON
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        data = json.loads(content)

        # Convert to AnalyzeResponse
        scene_type = SceneType[data.get("scene_type", "unknown").upper()]
        content_hints = [ContentType[h.upper()] for h in data.get("content_hints", [])]

        return AnalyzeResponse(
            scene_type=scene_type,
            structure=data.get("structure"),
            content_hints=content_hints,
            suggested_tools=["detect_lines", "extract_region", "ocr_region"],
            regions_of_interest=[],
            description=data.get("description", ""),
            confidence=data.get("confidence", 0.9),
        )

    except Exception as e:
        print(f"Vision LLM analysis failed: {e}")
        return None


@tool
async def analyze(
    image_uri: str,
    use_vision_model: bool = True,
) -> AnalyzeResponse:
    """High-level scene understanding - determines what's in an image and suggests next steps.

    This is typically the FIRST tool to call when encountering a new image.
    It uses vision models (GPT-4.1-mini or Claude) to understand the scene,
    then suggests which atomic vision tools to use next.

    Args:
        image_uri: URI or file path to the image to analyze
        use_vision_model: If True, use LLM vision API for analysis.
            If False, use classical CV heuristics (faster but less accurate).

    Returns:
        AnalyzeResponse: Contains:
            - scene_type: What kind of image (puzzle, document, photo, etc.)
            - structure: Detected structure (e.g., "grid_9x9", "table")
            - content_hints: What's in it (digits, text, handwritten, etc.)
            - suggested_tools: Which vision tools to call next
            - regions_of_interest: Important areas to focus on
            - description: Natural language description
            - confidence: How confident the analysis is

    Example:
        result = await analyze("sudoku.jpg")
        # result.scene_type = SceneType.PUZZLE
        # result.structure = "grid_9x9"
        # result.content_hints = [ContentType.DIGITS, ContentType.GRID]
        # result.suggested_tools = ["detect_lines", "extract_region", "ocr_region"]

    Tips for planners:
        - Use this first to understand what you're working with
        - Follow the suggested_tools in the response
        - Use regions_of_interest to focus extraction
        - If confidence is low, may need human input
    """
    if use_vision_model:
        # Call GPT-4o-mini or Claude Haiku vision API for analysis
        # This provides much better scene understanding than CV heuristics
        try:
            vision_result = await _analyze_with_vision_llm(image_uri)
            if vision_result:
                return vision_result
        except Exception as e:
            # Fall back to heuristics if vision API fails
            print(f"Vision API failed: {e}, falling back to CV heuristics")
            pass

    # Heuristic analysis using classical CV
    img = await load_image_from_artifact(image_uri)
    height, width = img.shape[:2]

    # Try to detect grid structure
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)

    # Count horizontal and vertical lines
    h_lines = 0
    v_lines = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180
            if angle < 10 or angle > 170:
                h_lines += 1
            elif 80 < angle < 100:
                v_lines += 1

    # Determine scene type based on heuristics
    scene_type = SceneType.UNKNOWN
    structure = None
    content_hints = []
    suggested_tools = []

    if h_lines >= 8 and v_lines >= 8:
        # Likely a grid
        scene_type = SceneType.PUZZLE
        if h_lines == 10 and v_lines == 10:
            structure = "grid_9x9"  # Sudoku-like
        else:
            structure = f"grid_{v_lines - 1}x{h_lines - 1}"

        content_hints = [ContentType.GRID, ContentType.DIGITS]
        suggested_tools = ["detect_lines", "extract_region", "ocr_region"]

    elif h_lines >= 3 or v_lines >= 3:
        # Table or document
        scene_type = SceneType.TABLE
        structure = "table"
        content_hints = [ContentType.TEXT]
        suggested_tools = ["detect_lines", "ocr_region"]

    else:
        # Generic photo or document
        scene_type = SceneType.DOCUMENT
        content_hints = [ContentType.TEXT]
        suggested_tools = ["ocr_region"]

    description = f"Image appears to be a {scene_type.value}"
    if structure:
        description += f" with {structure} structure"

    return AnalyzeResponse(
        scene_type=scene_type,
        structure=structure,
        content_hints=content_hints,
        suggested_tools=suggested_tools,
        regions_of_interest=[],  # TODO: Implement region detection
        description=description,
        confidence=0.7,  # Heuristic confidence
    )

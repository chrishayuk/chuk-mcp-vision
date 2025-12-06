#!/usr/bin/env python3
"""Sudoku extraction using Vision LLM (GPT-4V or Claude).

This example demonstrates using a vision-capable LLM to directly read
the Sudoku grid, which is much more accurate than traditional OCR.
"""

import asyncio
import sys
from pathlib import Path
import base64
import json
import os

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2

from chuk_mcp_vision.tools.common import load_image_from_artifact


async def extract_sudoku_with_vision_llm(image_path: str, provider: str = "openai"):
    """Extract Sudoku grid using a vision LLM.

    Args:
        image_path: Path to Sudoku image
        provider: "openai" for GPT-4V or "anthropic" for Claude

    Returns:
        9x9 grid as list of lists, with None for empty cells
    """
    # Load image
    img = await load_image_from_artifact(image_path)

    # Encode image to base64
    success, buffer = cv2.imencode(".jpg", img)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    # Prompt for LLM
    prompt = """Look at this Sudoku puzzle image and extract the grid VERY CAREFULLY.

CRITICAL: Read the grid from TOP TO BOTTOM, LEFT TO RIGHT.
Row 0 = TOP row, Row 8 = BOTTOM row
Column 0 = LEFT column, Column 8 = RIGHT column

Return ONLY a valid JSON object with this exact structure:
{
  "grid": [
    [row0col0, row0col1, row0col2, row0col3, row0col4, row0col5, row0col6, row0col7, row0col8],
    [row1col0, row1col1, row1col2, row1col3, row1col4, row1col5, row1col6, row1col7, row1col8],
    [row2col0, row2col1, row2col2, row2col3, row2col4, row2col5, row2col6, row2col7, row2col8],
    [row3col0, row3col1, row3col2, row3col3, row3col4, row3col5, row3col6, row3col7, row3col8],
    [row4col0, row4col1, row4col2, row4col3, row4col4, row4col5, row4col6, row4col7, row4col8],
    [row5col0, row5col1, row5col2, row5col3, row5col4, row5col5, row5col6, row5col7, row5col8],
    [row6col0, row6col1, row6col2, row6col3, row6col4, row6col5, row6col6, row6col7, row6col8],
    [row7col0, row7col1, row7col2, row7col3, row7col4, row7col5, row7col6, row7col7, row7col8],
    [row8col0, row8col1, row8col2, row8col3, row8col4, row8col5, row8col6, row8col7, row8col8]
  ]
}

Rules:
- Use null for empty cells
- Use integers 1-9 for filled cells
- Must be exactly 9 rows and 9 columns
- The VERY FIRST value is the TOP-LEFT cell
- The VERY LAST value is the BOTTOM-RIGHT cell
- Read each row LEFT to RIGHT before moving to next row
- TRIPLE-CHECK the position of each digit
- Return ONLY the JSON object, no other text or explanation

Example: If the top row shows "7 _ _ 5 2 _ _ _ 8" (where _ is empty), the first array should be:
[7, null, null, 5, 2, null, null, null, 8]
"""

    if provider == "openai":
        return await _call_openai_vision(img_base64, prompt)
    elif provider == "anthropic":
        return await _call_anthropic_vision(img_base64, prompt)
    else:
        raise ValueError(f"Unknown provider: {provider}")


async def _call_openai_vision(img_base64: str, prompt: str):
    """Call OpenAI GPT-4 Vision API."""
    import httpx

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o",  # or gpt-4-turbo with vision
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                            },
                        ],
                    }
                ],
                "max_tokens": 1000,
            },
        )

        result = response.json()

        if "error" in result:
            raise ValueError(f"OpenAI API error: {result['error']}")

        # Extract JSON from response
        content = result["choices"][0]["message"]["content"]

        # Parse JSON (handle markdown code blocks if present)
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        data = json.loads(content)
        return data["grid"]


async def _call_anthropic_vision(img_base64: str, prompt: str):
    """Call Anthropic Claude Vision API."""
    import httpx

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 1000,
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

        if "error" in result:
            raise ValueError(f"Anthropic API error: {result['error']}")

        # Extract JSON from response
        content = result["content"][0]["text"]

        # Parse JSON (handle markdown code blocks if present)
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        data = json.loads(content)
        return data["grid"]


def display_grid(grid):
    """Display Sudoku grid in a nice format."""
    print("   ", "┌" + "───┬" * 8 + "───┐")
    for i, row in enumerate(grid):
        print("   ", "│", end="")
        for j, cell in enumerate(row):
            val = str(cell) if cell is not None else "."
            print(f" {val} ", end="")
            if j < 8:
                print("│" if (j + 1) % 3 == 0 else "│", end="")
        print("│")

        if i < 8:
            if (i + 1) % 3 == 0:
                print("   ", "├" + "───┼" * 8 + "───┤")
            else:
                print("   ", "├" + "───┼" * 8 + "───┤")
    print("   ", "└" + "───┴" * 8 + "───┘")


async def main():
    """Extract Sudoku using Vision LLM."""
    print("=" * 70)
    print("CHUK Vision - Sudoku Extraction with Vision LLM")
    print("=" * 70)
    print()

    sudoku_path = Path(__file__).parent.parent / "assets" / "sudoku.jpeg"
    print(f"📸 Loading: {sudoku_path.name}")
    print()

    # Determine which provider to use
    if os.environ.get("OPENAI_API_KEY"):
        provider = "openai"
        model = "GPT-4 Vision"
    elif os.environ.get("ANTHROPIC_API_KEY"):
        provider = "anthropic"
        model = "Claude 3.5 Sonnet"
    else:
        print("❌ Error: No API key found!")
        print()
        print("Please set one of:")
        print("  export OPENAI_API_KEY='your-key-here'")
        print("  export ANTHROPIC_API_KEY='your-key-here'")
        return

    print(f"🤖 Using: {model}")
    print()
    print("Extracting Sudoku grid...")

    try:
        grid = await extract_sudoku_with_vision_llm(str(sudoku_path), provider)

        print("✅ Extraction complete!")
        print()

        # Count filled cells
        filled = sum(1 for row in grid for cell in row if cell is not None)

        print("📊 Statistics:")
        print("   Total cells: 81")
        print(f"   Filled cells: {filled}")
        print(f"   Empty cells: {81 - filled}")
        print(f"   Fill rate: {filled / 81 * 100:.1f}%")
        print()

        print("🔢 Extracted Sudoku Grid:")
        print()
        display_grid(grid)
        print()

        print("✅ Ready to solve!")
        print()
        print("Next steps:")
        print("  1. Feed this grid to chuk-mcp-solver")
        print("  2. Verify accuracy (vision LLMs are typically 99%+ accurate)")
        print("  3. Create MCP tool that wraps this functionality")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

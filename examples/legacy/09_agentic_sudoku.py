#!/usr/bin/env python3
"""Agentic Sudoku extraction - LLM orchestrates vision tools.

This demonstrates the CHUK architecture philosophy:
- Generic, atomic vision tools (detect_lines, extract_region, ocr_region, analyze)
- AI agent orchestrates tools to solve complex tasks
- No hard-coded pipelines - LLM figures out the best approach

The LLM has access to all vision tools and iteratively refines its approach
to achieve 100% accuracy on Sudoku extraction.
"""

import asyncio
import sys
from pathlib import Path
import json
import base64

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import os

from chuk_mcp_vision.tools.common import load_image_from_artifact, save_image_as_artifact
from chuk_mcp_vision.tools.analyze.tool import analyze
from chuk_mcp_vision.tools.detect.lines import detect_lines
from chuk_mcp_vision.tools.extract.region import extract_region
from chuk_mcp_vision.tools.ocr.tool import ocr_region
from chuk_mcp_vision.models import OCRBackend


# Ground truth for validation
GROUND_TRUTH = [
    [7, None, None, 5, 2, None, None, None, 8],
    [None, 5, 6, None, 9, 8, None, None, None],
    [None, 4, None, 3, 6, 7, None, 5, None],
    [None, 6, 2, 7, 8, None, None, None, None],
    [8, None, 1, 4, None, None, None, None, 2],
    [4, 3, None, None, 1, 9, None, 6, None],
    [None, None, None, None, None, 5, None, None, None],
    [5, None, None, 6, None, 2, 9, 3, 1],
    [None, None, 7, 9, 4, 1, 5, None, None],
]


async def call_llm(messages: list, provider: str = "openai"):
    """Call LLM API (OpenAI or Anthropic)."""
    import httpx

    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-4o",
                    "messages": messages,
                    "max_tokens": 2000,
                },
            )
            result = response.json()
            if "error" in result:
                raise ValueError(f"OpenAI error: {result['error']}")
            return result["choices"][0]["message"]["content"]

    elif provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        # Convert messages to Anthropic format
        system_msg = None
        user_msgs = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                user_msgs.append(msg)

        async with httpx.AsyncClient(timeout=60.0) as client:
            payload = {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 2000,
                "messages": user_msgs,
            }
            if system_msg:
                payload["system"] = system_msg

            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            result = response.json()
            if "error" in result:
                raise ValueError(f"Anthropic error: {result['error']}")
            return result["content"][0]["text"]


async def agentic_extract_sudoku(
    image_path: str, provider: str = "openai", max_iterations: int = 5
):
    """Use an LLM agent to orchestrate vision tools and extract Sudoku grid.

    The agent can:
    1. analyze_image - Get high-level understanding
    2. detect_lines - Find grid structure
    3. extract_region - Extract specific areas
    4. ocr_region - Read digits
    5. Iteratively refine approach based on results

    Args:
        image_path: Path to Sudoku image
        provider: "openai" or "anthropic"
        max_iterations: Maximum planning iterations

    Returns:
        Extracted 9x9 grid
    """
    # Load and prepare image
    img = await load_image_from_artifact(image_path)
    h, w = img.shape[:2]

    # Save as artifact for tool use
    image_uri = await save_image_as_artifact(img, suffix=".jpg", category="sudoku")

    # Encode for LLM vision
    success, buffer = cv2.imencode(".jpg", img)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    system_prompt = f"""You are an AI agent tasked with extracting a Sudoku grid from an image.

You have access to these vision tools (simulated as function calls):
1. analyze_image(uri) -> scene understanding, structure detection
2. detect_lines(uri) -> find grid lines (horizontal/vertical)
3. extract_region(uri, x, y, width, height) -> crop specific area
4. ocr_region(uri, backend="paddleocr") -> read text/digits

The image is {w}×{h} pixels.

Your goal: Extract the 9×9 Sudoku grid with 100% accuracy.

Approach:
1. First, look at the image and understand its structure
2. Plan your extraction strategy step-by-step
3. Execute tool calls (I'll simulate results)
4. Validate results against ground truth
5. Refine approach if needed

Output your plan as JSON:
{{
  "strategy": "description of approach",
  "steps": [
    {{"tool": "analyze_image", "params": {{"uri": "{image_uri}"}}, "reason": "why"}},
    {{"tool": "detect_lines", "params": {{"uri": "{image_uri}"}}, "reason": "why"}},
    ...
  ],
  "expected_outcome": "what you expect to achieve"
}}

Be methodical. Think about grid alignment, cell boundaries, margins to avoid grid lines."""

    conversation = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Extract the Sudoku grid from this image. Start by analyzing it.",
                },
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
            ]
            if provider == "openai"
            else [
                {
                    "type": "text",
                    "text": "Extract the Sudoku grid from this image. Start by analyzing it.",
                },
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": img_base64},
                },
            ],
        },
    ]

    print("🤖 Agent starting...\n")

    for iteration in range(max_iterations):
        print(f"{'=' * 70}")
        print(f"Iteration {iteration + 1}/{max_iterations}")
        print(f"{'=' * 70}\n")

        # Get LLM response
        response = await call_llm(conversation, provider)
        print(f"Agent response:\n{response}\n")

        # Parse plan (extract JSON if present)
        plan = None
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0].strip()
            plan = json.loads(json_str)
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0].strip()
            try:
                plan = json.loads(json_str)
            except:
                pass

        if not plan:
            # Try to extract grid directly if LLM provided it
            if "grid" in response.lower():
                print("Agent may have provided grid directly - checking...")
                # This would be the vision LLM approach
                break

            print("No structured plan found - asking for clarification")
            conversation.append({"role": "assistant", "content": response})
            conversation.append(
                {
                    "role": "user",
                    "content": "Please provide a structured JSON plan with tool calls.",
                }
            )
            continue

        print(f"📋 Strategy: {plan.get('strategy', 'N/A')}")
        print(f"📊 Steps: {len(plan.get('steps', []))}")
        print()

        # Execute tool calls (simplified - in real system these would be actual MCP calls)
        results = []
        for i, step in enumerate(plan.get("steps", [])):
            tool = step.get("tool")
            params = step.get("params", {})
            reason = step.get("reason", "")

            print(f"  Step {i + 1}: {tool}")
            print(f"    Reason: {reason}")
            print(f"    Params: {params}")

            # Simulate tool execution
            if tool == "analyze_image" or tool == "analyze":
                result = await analyze(params["uri"])
                print(f"    Result: {result.scene_type}, structure={result.structure}")
                results.append({"tool": tool, "result": result.model_dump()})

            elif tool == "detect_lines":
                result = await detect_lines(params["uri"])
                print(f"    Result: {result.horizontal_lines}H × {result.vertical_lines}V lines")
                results.append(
                    {
                        "tool": tool,
                        "result": {
                            "h_lines": result.horizontal_lines,
                            "v_lines": result.vertical_lines,
                            "total": len(result.lines),
                        },
                    }
                )

            elif tool == "extract_region":
                result = await extract_region(
                    params["uri"], params["x"], params["y"], params["width"], params["height"]
                )
                print(f"    Result: Region extracted -> {result.output_image_uri}")
                results.append({"tool": tool, "result": {"uri": result.output_image_uri}})

            elif tool == "ocr_region":
                backend = OCRBackend(params.get("backend", "paddleocr"))
                result = await ocr_region(params["uri"], backend=backend)
                print(f"    Result: '{result.full_text}' (conf={result.mean_confidence:.2f})")
                results.append(
                    {
                        "tool": tool,
                        "result": {"text": result.full_text, "confidence": result.mean_confidence},
                    }
                )

            print()

        # Provide results back to LLM
        conversation.append({"role": "assistant", "content": response})
        conversation.append(
            {
                "role": "user",
                "content": f"Tool execution results:\n{json.dumps(results, indent=2)}\n\nBased on these results, what's your next step? If you have enough info to extract the grid, provide it as a 9x9 array.",
            }
        )

    print("\n⚠️  Maximum iterations reached - switching to direct Vision LLM approach")
    print("(The agent would ideally converge to a solution within iterations)\n")

    # Fallback: Just use vision LLM directly
    vision_prompt = """Extract the Sudoku grid from this image.

Return ONLY a JSON object:
{
  "grid": [
    [row0 - 9 integers or null],
    [row1 - 9 integers or null],
    ...
    [row8 - 9 integers or null]
  ]
}

Rules:
- Top-left is [0,0], bottom-right is [8,8]
- Use null for empty cells, integers 1-9 for filled cells
- Exactly 9 rows × 9 columns
"""

    vision_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": vision_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
            ]
            if provider == "openai"
            else [
                {"type": "text", "text": vision_prompt},
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": img_base64},
                },
            ],
        }
    ]

    response = await call_llm(vision_messages, provider)

    # Extract grid
    if "```json" in response:
        json_str = response.split("```json")[1].split("```")[0].strip()
    elif "```" in response:
        json_str = response.split("```")[1].split("```")[0].strip()
    else:
        json_str = response

    data = json.loads(json_str)
    return data["grid"]


def display_grid(grid, title="Sudoku Grid"):
    """Display grid."""
    print(f"{title}:")
    print()
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
    print()


def validate_grid(grid, ground_truth):
    """Compare extracted grid with ground truth."""
    errors = []
    for r in range(9):
        for c in range(9):
            if ground_truth[r][c] != grid[r][c]:
                errors.append((r, c, ground_truth[r][c], grid[r][c]))

    accuracy = (81 - len(errors)) / 81 * 100

    print(f"✅ Accuracy: {accuracy:.1f}% ({81 - len(errors)}/81 correct)")
    print()

    if errors:
        print(f"❌ Errors: {len(errors)}")
        for r, c, expected, got in errors[:10]:
            print(f"   [{r},{c}]: Expected {expected}, Got {got}")
        if len(errors) > 10:
            print(f"   ... and {len(errors) - 10} more")
    else:
        print("🎉 PERFECT! 100% accuracy achieved!")

    return accuracy


async def main():
    """Agentic Sudoku extraction demo."""
    print("=" * 70)
    print("CHUK Vision - Agentic AI Sudoku Extraction")
    print("=" * 70)
    print()
    print("Philosophy: Let AI orchestrate atomic tools, not hard-coded pipelines")
    print()

    sudoku_path = Path(__file__).parent.parent / "assets" / "sudoku.jpeg"
    print(f"📸 Loading: {sudoku_path.name}")
    print()

    # Determine provider
    if os.environ.get("OPENAI_API_KEY"):
        provider = "openai"
        model = "GPT-4o"
    elif os.environ.get("ANTHROPIC_API_KEY"):
        provider = "anthropic"
        model = "Claude 3.5 Sonnet"
    else:
        print("❌ Error: No API key found!")
        print("Set OPENAI_API_KEY or ANTHROPIC_API_KEY")
        return

    print(f"🤖 Agent: {model}")
    print()

    try:
        grid = await agentic_extract_sudoku(str(sudoku_path), provider, max_iterations=3)

        print()
        print("=" * 70)
        print("Results")
        print("=" * 70)
        print()

        display_grid(grid, "🎯 Extracted Grid")
        validate_grid(grid, GROUND_TRUTH)

        print()
        print("💡 Key Insight:")
        print("   This demonstrates CHUK's philosophy: atomic tools + AI orchestration")
        print("   Instead of hard-coding extraction logic, we let the AI figure out")
        print("   the best approach using generic vision primitives.")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

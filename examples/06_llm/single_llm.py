#!/usr/bin/env python3
"""Example: Single Vision LLM - Use one Vision LLM for verification.

This example demonstrates using a single Vision LLM (OpenAI, Anthropic, or Google)
to verify OCR results or extract visual information.

Tools used:
- detect_grid, extract_grid_cells: Extract cells
- verify_cell_multi_llm: Single-provider LLM verification
"""

import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from chuk_mcp_vision.tools.grid import detect_grid, extract_grid_cells
from chuk_mcp_vision.tools.llm.consensus import verify_cell_multi_llm


async def demo_single_llm():
    """Demonstrate single Vision LLM verification."""
    print("=" * 70)
    print("Single Vision LLM Demo")
    print("=" * 70)
    print()

    # Check which LLM providers are available
    providers_available = []
    if os.environ.get("OPENAI_API_KEY"):
        providers_available.append(("openai", "GPT-4o"))
    if os.environ.get("ANTHROPIC_API_KEY"):
        providers_available.append(("anthropic", "Claude 3.5 Sonnet"))
    if os.environ.get("GOOGLE_API_KEY"):
        providers_available.append(("google", "Gemini 2.0 Flash"))

    if not providers_available:
        print("❌ No LLM API keys found in environment")
        print("   Set one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY")
        return

    provider, model_name = providers_available[0]
    print(f"Using provider: {provider} ({model_name})")
    print()

    # Load image
    image_path = Path(__file__).parent.parent.parent / "assets" / "sudoku.jpeg"
    if not image_path.exists():
        print(f"❌ Error: {image_path} not found")
        return

    print(f"📸 Image: {image_path.name}")
    print()

    # Extract grid and cells
    print("Step 1: Extract Grid Cells")
    print("-" * 70)

    grid = await detect_grid(str(image_path))
    cells = await extract_grid_cells(str(image_path), grid)

    print(f"✅ Grid extracted: {grid.rows}×{grid.cols} = {len(cells.cells)} cells")
    print()

    # Test on a few cells
    print("Step 2: Verify Cells with Vision LLM")
    print("-" * 70)

    test_cells = [
        (0, 0, 7),  # Should be 7
        (0, 3, 5),  # Should be 5
        (1, 1, 5),  # Should be 5
        (2, 1, 4),  # Should be 4
    ]

    results = []
    for row, col, expected in test_cells:
        cell_idx = row * 9 + col
        cell = cells.cells[cell_idx]

        print(f"\nCell [{row},{col}] (expected: {expected}):")

        # Single provider verification
        result = await verify_cell_multi_llm(
            cell.image_uri,
            providers=[provider],  # Single provider only
            expected_type="digit",
        )

        llm_value = result["consensus_value"]
        confidence = result["consensus_confidence"]
        correct = llm_value == expected

        marker = "✅" if correct else "❌"
        print(f"   LLM result: '{llm_value}' (confidence: {confidence:.2%}) {marker}")

        results.append(
            {
                "row": row,
                "col": col,
                "expected": expected,
                "llm_value": llm_value,
                "confidence": confidence,
                "correct": correct,
            }
        )

    print()

    # Summary
    print("Summary:")
    print("-" * 70)
    correct_count = sum(1 for r in results if r["correct"])
    total_count = len(results)
    accuracy = correct_count / total_count if total_count > 0 else 0

    print(f"Accuracy: {correct_count}/{total_count} = {accuracy:.1%}")
    print()

    if accuracy < 1.0:
        print("❌ Errors found:")
        for r in results:
            if not r["correct"]:
                print(
                    f"   Cell [{r['row']},{r['col']}]: "
                    f"Expected '{r['expected']}', got '{r['llm_value']}'"
                )
        print()

    print("💡 When to use single LLM:")
    print("-" * 70)
    print("  • Cost-sensitive applications")
    print("  • When one provider is trusted")
    print("  • Quick prototyping")
    print("  • Non-critical accuracy requirements")
    print()

    print("💡 When to use multi-LLM consensus:")
    print("-" * 70)
    print("  • Critical accuracy needed")
    print("  • Ambiguous or challenging images")
    print("  • When cost is not primary concern")
    print("  • Production systems requiring validation")
    print()

    print(f"Provider used: {provider} ({model_name})")
    print(f"Cost estimate: ~${len(results) * 0.003:.4f} ({len(results)} images)")
    print()


if __name__ == "__main__":
    asyncio.run(demo_single_llm())

"""Example 13: Multi-LLM Consensus - Query multiple Vision LLMs and vote

Goal: Achieve 100% accuracy by combining multiple Vision LLMs
- Use chuk-llm to query OpenAI (GPT-4o), Anthropic (Claude), Google (Gemini)
- Vote with confidence weighting
- Combine with rotation testing for 6/9 disambiguation

Tools used:
- detect_grid: Find grid boundary
- extract_grid_cells: Extract all cells
- verify_cell_multi_llm: Multi-provider vision verification
- verify_digit_rotation_consensus: Rotation + multi-LLM

Expected result: Better than 98.8% (fixes remaining error)
"""

import asyncio
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from chuk_mcp_vision.tools.grid.detect import detect_grid
from chuk_mcp_vision.tools.grid.extract_cells import extract_grid_cells
from chuk_mcp_vision.tools.llm.consensus import (
    verify_cell_multi_llm,
    verify_digit_rotation_consensus,
)


async def main():
    # Load image
    image_path = Path(__file__).parent.parent.parent / "assets" / "sudoku.jpeg"
    if not image_path.exists():
        print(f"Error: {image_path} not found")
        return

    # Use image path directly
    image_uri = str(image_path)

    print("=" * 80)
    print("Example 13: Multi-LLM Consensus (OpenAI + Anthropic + Google)")
    print("=" * 80)
    print()

    # Step 1: Detect grid
    print("Step 1: Detecting grid...")
    grid = await detect_grid(image_uri)
    print(f"  Grid: {grid.rows}x{grid.cols} cells")
    print()

    # Step 2: Extract cells
    print("Step 2: Extracting cells...")
    cells_response = await extract_grid_cells(image_uri, grid)
    cells = cells_response.cells
    print(f"  Extracted {len(cells)} cells")
    print()

    # Step 3: Test multi-LLM consensus on suspicious cell [2,4]
    # This is the cell that was wrong in Example 12 (OCR + single LLM said "9", truth is "6")
    print("Step 3: Testing multi-LLM consensus on suspicious cell [2,4]...")
    print()

    cell_2_4 = cells[2 * 9 + 4]  # Row 2, Col 4 (0-indexed)

    print("  Method 1: Multi-LLM consensus (no rotation)")
    result_consensus = await verify_cell_multi_llm(
        cell_2_4.image_uri,
        providers=["openai", "anthropic", "google"],
        expected_type="digit",
        context={"grid_position": [2, 4], "use_case": "sudoku"},
    )

    print(f"  Consensus value: {result_consensus['consensus_value']}")
    print(f"  Consensus confidence: {result_consensus['consensus_confidence']:.2f}")
    print(
        f"  Successful providers: {result_consensus['successful_providers']}/{result_consensus['total_providers']}"
    )
    print()
    print("  Individual provider results:")
    for r in result_consensus["results"]:
        if r["success"]:
            print(
                f"    {r['provider']:12} ({r['model']:30}): {r['value']} (conf: {r['confidence']:.2f})"
            )
        else:
            print(f"    {r['provider']:12}: FAILED - {r['error']}")
    print()

    if result_consensus["disagreements"]:
        print("  ⚠️  Disagreements detected:")
        for d in result_consensus["disagreements"]:
            print(f"    {d}")
        print()

    # Step 4: Test rotation + multi-LLM consensus
    print("  Method 2: Rotation + Multi-LLM consensus")
    result_rotation = await verify_digit_rotation_consensus(
        cell_2_4.image_uri,
        angles=[0, 180],  # Test 6 vs 9
        providers=["openai", "anthropic", "google"],
    )

    print(f"  Recommended value: {result_rotation['recommended_value']}")
    print(f"  Recommended angle: {result_rotation['recommended_angle']}°")
    print(f"  Confidence: {result_rotation['confidence']:.2f}")
    print(f"  Orientation dependency: {result_rotation['orientation_dependency']}")
    print()
    print("  Results by angle:")
    for r in result_rotation["results_by_angle"]:
        print(f"    {r['angle']:3}°: {r['value']} (conf: {r['confidence']:.2f})")
    print()

    # Step 5: Apply multi-LLM consensus to all cells
    print("Step 4: Applying multi-LLM consensus to ALL cells...")
    print("(Note: This will be expensive - querying 81 cells × 3 providers = 243 API calls)")
    print()

    user_input = input("Continue with full grid verification? (y/n): ")
    if user_input.lower() != "y":
        print("Skipping full grid verification.")
        print()
        print("=" * 80)
        print("Summary")
        print("=" * 80)
        print()
        print("Multi-LLM consensus provides:")
        print("  ✅ Reduced bias - Different models trained on different data")
        print("  ✅ Higher accuracy - Majority vote more reliable than single model")
        print("  ✅ Confidence scores - Know when models agree vs disagree")
        print("  ✅ Rotation testing - Disambiguates 6 vs 9, b vs q, etc.")
        print()
        print("Trade-offs:")
        print("  ⚠️  Higher cost - 3x API calls")
        print("  ⚠️  Slower - Sequential queries to different providers")
        print()
        print("Recommended strategy:")
        print("  1. Use grid tools + domain validation (93.8%, cheap, fast)")
        print("  2. Flag suspicious cells with validation")
        print("  3. Use multi-LLM consensus ONLY on flagged cells")
        print("  4. Expected accuracy: 99%+")
        return

    # Full grid verification (expensive!)
    grid_values = [[None for _ in range(9)] for _ in range(9)]

    for idx, cell in enumerate(cells):
        row = idx // 9
        col = idx % 9

        print(f"  Cell [{row},{col}]...", end=" ", flush=True)

        result = await verify_cell_multi_llm(
            cell.image_uri, providers=["openai", "anthropic", "google"], expected_type="digit"
        )

        value = result["consensus_value"]
        confidence = result["consensus_confidence"]

        grid_values[row][col] = value
        print(f"{value} (conf: {confidence:.2f})")

    print()
    print("=" * 80)
    print("Extracted Grid (Multi-LLM Consensus)")
    print("=" * 80)
    print()

    # Print grid
    for row in grid_values:
        print("  " + " ".join(str(v) if v is not None else "." for v in row))
    print()

    # Load ground truth
    ground_truth_path = Path(__file__).parent / "easy-sudoku-solution.json"
    if ground_truth_path.exists():
        with open(ground_truth_path) as f:
            ground_truth = json.load(f)

        # Compare
        correct = 0
        total = 0
        errors = []

        for row in range(9):
            for col in range(9):
                if ground_truth[row][col] is not None:
                    total += 1
                    predicted = grid_values[row][col]
                    actual = ground_truth[row][col]

                    if predicted == actual:
                        correct += 1
                    else:
                        errors.append(
                            {"position": [row, col], "predicted": predicted, "actual": actual}
                        )

        accuracy = (correct / total * 100) if total > 0 else 0

        print("=" * 80)
        print("Results")
        print("=" * 80)
        print()
        print(f"Accuracy: {correct}/{total} = {accuracy:.1f}%")
        print()

        if errors:
            print(f"Errors ({len(errors)}):")
            for err in errors:
                print(
                    f"  Cell {err['position']}: predicted {err['predicted']}, actual {err['actual']}"
                )
        else:
            print("🎉 100% ACCURACY ACHIEVED! 🎉")
        print()

        # Compare with previous results
        print("=" * 80)
        print("Comparison with Previous Methods")
        print("=" * 80)
        print()
        print("| Method | Accuracy | Errors | Cost | Speed |")
        print("|--------|----------|--------|------|-------|")
        print("| Grid tools (Example 10) | 93.8% | 5 | Low | Fast |")
        print("| Multi-backend OCR (Example 11) | 92.6% | 6 | Medium | Medium |")
        print("| Grid + Single LLM (Example 12) | 98.8% | 1 | Low | Fast |")
        print(
            f"| **Multi-LLM Consensus** | **{accuracy:.1f}%** | **{len(errors)}** | **High** | **Slow** |"
        )
        print()


if __name__ == "__main__":
    asyncio.run(main())

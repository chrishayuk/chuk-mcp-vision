"""Multi-LLM consensus tool - uses chuk-llm for vision verification across providers."""

import base64
import os
from typing import Dict, List, Optional

from chuk_mcp_server import tool

from ...models import VisionLLMProvider, VisionModel
from ..common import load_image_from_artifact


async def _call_openai_vision(prompt: str, image_base64: str, model: Optional[str] = None) -> str:
    """Call OpenAI vision API."""
    import openai

    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Use provided model or default from VisionModel enum
    model_name = model or VisionModel.GPT_4O.value

    response = await client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                    },
                ],
            }
        ],
        max_tokens=50,
        temperature=0.0,
    )

    return response.choices[0].message.content


async def _call_anthropic_vision(
    prompt: str, image_base64: str, model: Optional[str] = None
) -> str:
    """Call Anthropic Claude vision API."""
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Use provided model or default from VisionModel enum
    model_name = model or VisionModel.CLAUDE_3_5_SONNET.value

    response = await client.messages.create(
        model=model_name,
        max_tokens=50,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_base64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )

    return response.content[0].text


async def _call_google_vision(prompt: str, image_base64: str, model: Optional[str] = None) -> str:
    """Call Google Gemini vision API."""
    import google.generativeai as genai
    from PIL import Image
    import io

    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    # Use provided model or default from VisionModel enum
    model_name = model or VisionModel.GEMINI_2_0_FLASH.value
    gemini_model = genai.GenerativeModel(model_name)

    # Convert base64 to PIL Image
    image_data = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_data))

    response = await gemini_model.generate_content_async([prompt, image])
    return response.text


@tool
async def verify_cell_multi_llm(
    cell_uri: str,
    providers: Optional[List[str]] = None,
    expected_type: str = "digit",
    context: Optional[Dict] = None,
    model_config: Optional[Dict[str, str]] = None,
) -> Dict:
    """Verify cell content using multiple Vision LLMs with voting consensus.

    Generic multi-provider vision verification:
    - Works for: digits, text, checkboxes, symbols
    - Uses chuk-llm to query multiple providers (OpenAI, Anthropic, Google)
    - Combines results with confidence-weighted voting
    - Identifies disagreements and low-confidence results

    Strategy:
    1. Load cell image and convert to base64
    2. Query each provider's vision model
    3. Parse and normalize responses
    4. Vote with confidence weighting
    5. Return consensus result + voting details

    Args:
        cell_uri: URI to cell image artifact
        providers: List of providers to query (default: ["openai", "anthropic", "google"])
        expected_type: Content type hint ("digit", "text", "checkbox", "auto")
        context: Optional context (e.g., "sudoku_cell", "grid_position": [2,4])
        model_config: Optional dict mapping provider to model name
                     e.g., {"openai": "gpt-4o", "anthropic": "claude-3-5-sonnet-20241022"}

    Returns:
        Dict with:
        - consensus_value: Most likely content (digit, text, or boolean)
        - consensus_confidence: Confidence in consensus (0.0-1.0)
        - results: List of individual provider results
        - voting_details: Breakdown of voting
        - disagreements: Where providers disagree
        - total_providers: Number of providers attempted
        - successful_providers: Number of successful responses
    """
    # Default providers: OpenAI (GPT-4o), Anthropic (Claude 3.5), Google (Gemini)
    if providers is None:
        providers = [
            VisionLLMProvider.OPENAI.value,
            VisionLLMProvider.ANTHROPIC.value,
            VisionLLMProvider.GOOGLE.value,
        ]

    # Default model config if not provided
    if model_config is None:
        model_config = {}

    # Load image and convert to base64
    img = await load_image_from_artifact(cell_uri)
    import cv2

    _, buffer = cv2.imencode(".png", img)
    image_base64 = base64.b64encode(buffer).decode("utf-8")

    # Build prompt based on expected type
    if expected_type == "digit":
        prompt = """Analyze this image of a grid cell.

What single digit (0-9) do you see? If empty, respond with "empty".

Respond ONLY with:
- A single digit (1-9 for Sudoku, 0-9 otherwise)
- OR the word "empty"
- OR "uncertain" if unclear

Do not explain. Just the value."""
    elif expected_type == "text":
        prompt = """Analyze this image of a grid cell.

What text do you see? If empty, respond with "empty".

Respond ONLY with the text content, nothing else."""
    elif expected_type == "checkbox":
        prompt = """Analyze this image of a checkbox or grid cell.

Is the checkbox checked?

Respond ONLY with:
- "checked" if checked
- "unchecked" if not checked
- "empty" if no checkbox present"""
    else:  # auto
        prompt = """Analyze this image of a grid cell.

What do you see in this cell?

Respond ONLY with the content value (digit, text, or "empty"). Nothing else."""

    # Add context if provided
    if context:
        context_str = ", ".join(f"{k}={v}" for k, v in context.items())
        prompt += f"\n\nContext: {context_str}"

    results = []

    # Query each provider
    for provider in providers:
        try:
            # Get model for this provider (use config or default)
            provider_model = model_config.get(provider)

            # Call provider-specific vision API
            if provider == VisionLLMProvider.OPENAI.value:
                response = await _call_openai_vision(prompt, image_base64, model=provider_model)
                model = provider_model or VisionModel.GPT_4O.value
            elif provider == VisionLLMProvider.ANTHROPIC.value:
                response = await _call_anthropic_vision(prompt, image_base64, model=provider_model)
                model = provider_model or VisionModel.CLAUDE_3_5_SONNET.value
            elif provider == VisionLLMProvider.GOOGLE.value:
                response = await _call_google_vision(prompt, image_base64, model=provider_model)
                model = provider_model or VisionModel.GEMINI_2_0_FLASH.value
            else:
                raise ValueError(f"Unsupported provider: {provider}")

            # Parse response
            # chuk-llm returns a string response
            text = response.strip().lower()

            # Normalize value
            if text in ["empty", "uncertain", "unclear", "none"]:
                value = None
                confidence = 0.5 if text == "uncertain" else 0.9
            elif text in ["checked", "unchecked"]:
                value = text == "checked"
                confidence = 0.95
            elif text.isdigit() and len(text) == 1:
                value = int(text)
                confidence = 0.95
            else:
                # Text content
                value = text
                confidence = 0.85

            results.append(
                {
                    "provider": provider,
                    "model": model,
                    "value": value,
                    "confidence": confidence,
                    "raw_response": response,
                    "success": True,
                    "error": None,
                }
            )

        except Exception as e:
            results.append(
                {
                    "provider": provider,
                    "model": None,
                    "value": None,
                    "confidence": 0.0,
                    "raw_response": None,
                    "success": False,
                    "error": str(e),
                }
            )

    # Analyze results and vote
    analysis = _analyze_consensus(results)

    return {
        "consensus_value": analysis["consensus_value"],
        "consensus_confidence": analysis["consensus_confidence"],
        "results": results,
        "voting_details": analysis["voting_details"],
        "disagreements": analysis["disagreements"],
        "total_providers": len(providers),
        "successful_providers": sum(1 for r in results if r["success"]),
    }


def _analyze_consensus(results: List[Dict]) -> Dict:
    """Analyze multi-provider results to determine consensus.

    Strategy:
    1. Group by value
    2. Weight votes by confidence
    3. Detect disagreements
    4. Return consensus with confidence
    """
    # Filter successful results
    successful = [r for r in results if r["success"]]

    if not successful:
        return {
            "consensus_value": None,
            "consensus_confidence": 0.0,
            "voting_details": {},
            "disagreements": ["All providers failed"],
        }

    # Group by value
    value_votes = {}
    for result in successful:
        value = result["value"]
        # Convert to hashable type for grouping
        if value is None:
            key = "empty"
        elif isinstance(value, bool):
            key = "checked" if value else "unchecked"
        else:
            key = str(value)

        if key not in value_votes:
            value_votes[key] = {
                "value": value,
                "weighted_confidence": 0.0,
                "count": 0,
                "providers": [],
            }

        value_votes[key]["weighted_confidence"] += result["confidence"]
        value_votes[key]["count"] += 1
        value_votes[key]["providers"].append(result["provider"])

    # Find consensus (highest weighted confidence)
    if not value_votes:
        return {
            "consensus_value": None,
            "consensus_confidence": 0.0,
            "voting_details": {},
            "disagreements": ["No valid values detected"],
        }

    consensus_key = max(value_votes.items(), key=lambda x: x[1]["weighted_confidence"])[0]
    consensus = value_votes[consensus_key]

    # Calculate consensus confidence
    total_confidence = sum(v["weighted_confidence"] for v in value_votes.values())
    consensus_confidence = (
        consensus["weighted_confidence"] / total_confidence if total_confidence > 0 else 0.0
    )

    # Detect disagreements
    disagreements = []
    if len(value_votes) > 1:
        # Multiple different values
        for key, votes in value_votes.items():
            if key != consensus_key:
                disagreements.append(
                    {
                        "value": votes["value"],
                        "providers": votes["providers"],
                        "confidence": votes["weighted_confidence"] / total_confidence,
                    }
                )

    # Check for low agreement (close vote)
    if consensus_confidence < 0.6:
        disagreements.append(
            {"issue": "Low consensus", "consensus_confidence": consensus_confidence}
        )

    return {
        "consensus_value": consensus["value"],
        "consensus_confidence": float(consensus_confidence),
        "voting_details": {
            k: {
                "value": v["value"],
                "count": v["count"],
                "providers": v["providers"],
                "share": v["weighted_confidence"] / total_confidence,
            }
            for k, v in value_votes.items()
        },
        "disagreements": disagreements,
    }


@tool
async def verify_digit_rotation_consensus(
    cell_uri: str, angles: Optional[List[int]] = None, providers: Optional[List[str]] = None
) -> Dict:
    """Combine rotation testing with multi-LLM consensus for ambiguous digits.

    Use case: Disambiguate 6 vs 9, b vs q, d vs p, etc.

    Strategy:
    1. Rotate image to each angle
    2. Query each provider at each angle
    3. Analyze for orientation-dependent content
    4. Return most likely value + orientation

    Args:
        cell_uri: URI to cell image
        angles: Rotation angles to test (default: [0, 180])
        providers: LLM providers to query

    Returns:
        Dict with:
        - recommended_value: Most likely correct value
        - recommended_angle: Most likely correct orientation (degrees)
        - confidence: Overall confidence
        - results_by_angle: Consensus at each angle
        - orientation_dependency: Whether content is orientation-dependent
    """
    from ..transform.rotate import rotate_and_classify

    # Set defaults
    if angles is None:
        angles = [0, 180]
    if providers is None:
        providers = ["openai", "anthropic", "google"]

    # Custom classifier callback that uses multi-LLM consensus
    async def multi_llm_classifier(image_uri: str) -> Dict:
        result = await verify_cell_multi_llm(image_uri, providers=providers, expected_type="digit")
        # Convert value to string for compatibility with rotation tool
        value = result["consensus_value"]
        if value is None:
            value_str = ""
        elif isinstance(value, bool):
            value_str = str(value).lower()
        else:
            value_str = str(value)

        return {
            "value": value_str,
            "confidence": result["consensus_confidence"],
            "metadata": result,
        }

    # Use rotation tool with multi-LLM classifier
    rotation_result = await rotate_and_classify(
        cell_uri, angles=angles, classifier_callback=multi_llm_classifier
    )

    return {
        "recommended_value": rotation_result.recommended_value,
        "recommended_angle": rotation_result.recommended_angle,
        "confidence": rotation_result.confidence,
        "results_by_angle": [
            {
                "angle": r.angle,
                "value": r.classification_value,
                "confidence": r.confidence,
                "image_uri": r.rotated_image_uri,
            }
            for r in rotation_result.results_by_angle
        ],
        "orientation_dependency": rotation_result.disambiguation,
        "method": "rotation + multi-LLM consensus",
    }

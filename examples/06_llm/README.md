# LLM - Vision LLM Tools

Vision Language Model tools for verification, consensus, and handling ambiguous visual content.

## Overview

Use multiple Vision LLMs (OpenAI GPT-4o, Anthropic Claude 3.5, Google Gemini 2.0) to:
- Verify OCR results
- Handle ambiguous characters (6 vs 9, b vs q, etc.)
- Achieve consensus through multi-provider voting
- Extract complex visual information

## Examples

| File | Description | Accuracy | Prerequisites |
|------|-------------|----------|---------------|
| `single_llm.py` | Single Vision LLM verification | 95-99% | One API key in .env |
| `multi_llm_consensus.py` | Multi-provider voting | 99%+ | Multiple API keys in .env |

## Quick Start

```bash
# Set up API keys in .env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-...
GOOGLE_API_KEY=...

# Run example
uv run python 06_llm/multi_llm_consensus.py
```

**Expected output**:
- Individual LLM responses
- Voting consensus
- Confidence scores
- Disagreement analysis

## Tools Reference

### verify_cell_multi_llm
Query multiple Vision LLMs and vote on consensus.

```python
from chuk_mcp_vision.tools.llm import verify_cell_multi_llm

result = await verify_cell_multi_llm(
    cell_uri,
    providers=["openai", "anthropic", "google"],  # Optional, uses all available
    expected_type="digit",                         # "digit", "letter", "text", "checkbox"
    context=None                                   # Optional context dict
)

print(result["consensus"])           # Voted consensus value
print(result["confidence"])          # Consensus confidence (0.0-1.0)
print(result["individual_responses"]) # Responses from each LLM
print(result["voting_details"])      # Voting breakdown
```

**How it works**:
1. Encodes image as base64
2. Sends to each Vision LLM with specialized prompt
3. Collects responses and confidence scores
4. Votes using confidence-weighted voting
5. Returns consensus with voting details

**Providers**:
- **openai**: GPT-4o (gpt-4o)
- **anthropic**: Claude 3.5 (claude-3-5-sonnet-20241022)
- **google**: Gemini 2.0 (gemini-2.0-flash-exp)

### verify_digit_rotation_consensus
Combine rotation testing with multi-LLM consensus for ambiguous digits.

```python
from chuk_mcp_vision.tools.llm import verify_digit_rotation_consensus

result = await verify_digit_rotation_consensus(
    cell_uri,
    providers=["openai", "anthropic", "google"]
)

print(result["recommended_digit"])   # Most likely digit
print(result["recommended_angle"])   # Correct rotation angle
print(result["confidence"])          # Overall confidence
print(result["rotation_results"])    # Results at each angle
```

**Use cases**:
- Disambiguating 6 vs 9
- Handling rotated images
- Verifying orientation-dependent characters

## Integration with chuk-llm

These tools use the `chuk-llm` library for provider abstraction:

```python
from chuk_llm import ask

# Single provider
response = await ask(
    prompt="What digit is in this image?",
    images=[image_base64],
    provider="openai"  # or "anthropic", "google"
)

# Multi-provider consensus (automatic)
response = await ask(
    prompt="What digit is in this image?",
    images=[image_base64],
    providers=["openai", "anthropic", "google"]
)
```

**Benefits**:
- Automatic provider discovery
- Unified API across providers
- Built-in error handling
- Confidence scoring
- Voting logic

## When to Use LLMs

### ✅ Use LLMs when:
- OCR confidence is low (<90%)
- Characters are ambiguous (6/9, b/q, d/p)
- Image is rotated or skewed
- Handwriting recognition needed
- Context is important
- Verification is critical

### ❌ Don't use LLMs when:
- OCR is already >95% accurate
- Text is clean and clear
- Speed is critical (LLMs are slower)
- Cost is a concern (LLMs more expensive than OCR)

## Voting Strategies

### Confidence-Weighted Voting
Default strategy - weight votes by confidence:

```
OpenAI:    "7" (0.98)
Anthropic: "7" (0.95)
Google:    "1" (0.60)
→ Consensus: "7" (weighted score: 0.96)
```

### Unanimous Voting
All LLMs must agree:

```
OpenAI:    "7"
Anthropic: "7"
Google:    "7"
→ Consensus: "7" (unanimous)
```

### Majority Voting
Simple majority:

```
OpenAI:    "7"
Anthropic: "7"
Google:    "1"
→ Consensus: "7" (2 out of 3)
```

## Use Cases

### OCR Verification
Verify critical OCR results:

```python
# Get OCR result
ocr_result = await ocr_region(image_uri)

# Verify with LLMs if confidence is low
if ocr_result.mean_confidence < 0.9:
    llm_result = await verify_cell_multi_llm(
        image_uri,
        expected_type="digit"
    )

    if ocr_result.full_text != llm_result["consensus"]:
        print(f"Mismatch detected: OCR={ocr_result.full_text}, LLM={llm_result['consensus']}")
        # Use LLM result (higher confidence)
        value = llm_result["consensus"]
```

### Ambiguous Character Resolution
Handle 6 vs 9, b vs q, etc.:

```python
result = await verify_digit_rotation_consensus(
    cell_uri,
    providers=["openai", "anthropic", "google"]
)

print(f"Digit: {result['recommended_digit']}")
print(f"Rotation: {result['recommended_angle']}°")
```

### Handwriting Recognition
LLMs excel at handwriting:

```python
result = await verify_cell_multi_llm(
    handwriting_uri,
    expected_type="text",
    context={"note": "Handwritten text, may be cursive"}
)
```

### Checkbox Detection
Detect checkbox states:

```python
result = await verify_cell_multi_llm(
    checkbox_uri,
    expected_type="checkbox"
)

is_checked = result["consensus"] == "checked"
```

## Advanced Techniques

### Cascading Verification
Start with OCR, escalate to LLM if needed:

```python
async def extract_with_verification(image_uri):
    # Try OCR first (fast, cheap)
    ocr_result = await ocr_region(image_uri)

    if ocr_result.mean_confidence > 0.95:
        # High confidence - use OCR
        return ocr_result.full_text

    # Low confidence - verify with LLMs
    llm_result = await verify_cell_multi_llm(image_uri)

    if llm_result["confidence"] > 0.9:
        # LLM consensus reached
        return llm_result["consensus"]

    # Still unclear - manual review
    return None
```

### Context-Aware Verification
Provide context to improve accuracy:

```python
result = await verify_cell_multi_llm(
    cell_uri,
    expected_type="digit",
    context={
        "valid_values": ["1", "2", "3", "4", "5", "6", "7", "8", "9"],
        "note": "This is a Sudoku cell, should be a single digit 1-9",
        "surrounding_values": {"top": "5", "left": "3"}
    }
)
```

### Multi-Angle Analysis
Test multiple rotations and vote:

```python
from chuk_mcp_vision.tools.transform import rotate_and_classify

# Test at multiple angles
rotation_result = await rotate_and_classify(
    image_uri,
    angles=[0, 90, 180, 270],
    classifier_callback=lambda uri: verify_cell_multi_llm(uri, expected_type="digit")
)

print(f"Recommended value: {rotation_result.recommended_value}")
print(f"Recommended angle: {rotation_result.recommended_angle}")
```

## Cost Considerations

Vision LLM API costs (approximate):
- **OpenAI GPT-4o**: $0.005 per image
- **Anthropic Claude 3.5**: $0.003 per image
- **Google Gemini 2.0**: $0.002 per image

**Cost optimization**:
1. Use OCR first, LLM only for verification
2. Cache LLM results
3. Use single provider when possible
4. Batch requests when supported
5. Set confidence thresholds to avoid unnecessary calls

## Performance Tips

1. **Cache results**: Store LLM responses to avoid duplicate calls
2. **Concurrent requests**: Process multiple images in parallel with asyncio
3. **Failover logic**: Handle API errors gracefully
4. **Timeout handling**: Set timeouts for API calls
5. **Rate limiting**: Respect provider rate limits

## Error Handling

```python
from chuk_llm import ask, LLMError

try:
    result = await verify_cell_multi_llm(
        image_uri,
        providers=["openai", "anthropic", "google"]
    )
except LLMError as e:
    print(f"LLM error: {e}")
    # Fallback to OCR
    result = await ocr_region(image_uri)
```

## Related Tools

- See `../05_ocr/` for OCR backends
- See `../02_transforms/` for rotation tools
- See `../03_preprocessing/` for image enhancement
- See `../07_complete_workflows/` for complete examples

## Next Steps

1. Run `multi_llm_consensus.py` to see voting in action
2. Set up API keys for all providers
3. Try verification on ambiguous images
4. Combine with OCR for cascading verification
5. Build custom LLM pipelines for your use case

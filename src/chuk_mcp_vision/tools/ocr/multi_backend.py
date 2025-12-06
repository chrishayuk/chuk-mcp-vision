"""Multi-backend OCR - ensemble voting for higher accuracy."""

from typing import List, Optional
from collections import Counter
from chuk_mcp_server import tool

from ...models import OCRBackend, MultiBackendOCRResponse, OCRBackendResult
from .tool import ocr_region


@tool
async def ocr_multi_backend(
    image_uri: str, backends: Optional[List[str]] = None, language: str = "en"
) -> MultiBackendOCRResponse:
    """Run multiple OCR backends and combine results with voting.

    This tool increases accuracy by running several OCR engines and using
    consensus voting. Different OCR engines have different strengths, and
    ensemble methods typically outperform any single engine.

    Use cases:
    - Low-confidence OCR results that need validation
    - Critical text where accuracy is paramount
    - Ambiguous characters (6 vs 9, O vs 0, 1 vs l vs I)

    Args:
        image_uri: URI to the image
        backends: List of backends to use. If None, uses ["paddleocr", "tesseract"]
        language: Language for OCR (default "en")

    Returns:
        MultiBackendOCRResponse with:
        - results: Individual results from each backend
        - consensus_text: Text agreed upon by majority
        - consensus_confidence: Confidence in consensus
        - voting_details: Breakdown of votes per character/word
        - disagreements: Where backends disagree
    """
    if backends is None:
        backends = ["paddleocr", "tesseract"]

    # Run all backends
    results = []
    for backend_name in backends:
        try:
            backend_enum = OCRBackend(backend_name.lower())
            ocr_result = await ocr_region(image_uri, backend=backend_enum, language=language)

            results.append(
                OCRBackendResult(
                    backend=backend_name,
                    text=ocr_result.full_text,
                    confidence=ocr_result.mean_confidence,
                    success=True,
                )
            )
        except Exception as e:
            results.append(
                OCRBackendResult(
                    backend=backend_name, text="", confidence=0.0, success=False, error=str(e)
                )
            )

    # Filter successful results
    successful = [r for r in results if r.success]

    if not successful:
        return MultiBackendOCRResponse(
            results=results,
            consensus_text="",
            consensus_confidence=0.0,
            voting_details={},
            disagreements=[],
            total_backends=len(backends),
            successful_backends=0,
        )

    # Voting logic
    texts = [r.text.strip() for r in successful]

    # Simple majority voting for full text
    text_counts = Counter(texts)
    consensus_text, consensus_count = text_counts.most_common(1)[0]

    # Calculate consensus confidence
    # Weight by: (vote_proportion * average_confidence_of_agreeing_backends)
    vote_proportion = consensus_count / len(successful)
    agreeing_confidences = [r.confidence for r in successful if r.text.strip() == consensus_text]
    avg_agreeing_conf = sum(agreeing_confidences) / len(agreeing_confidences)
    consensus_confidence = vote_proportion * avg_agreeing_conf

    # Find disagreements
    disagreements = []
    for i, result in enumerate(successful):
        if result.text.strip() != consensus_text:
            disagreements.append(
                {
                    "backend": result.backend,
                    "text": result.text,
                    "confidence": result.confidence,
                    "differs_from_consensus": True,
                }
            )

    # Voting details
    voting_details = {
        "text_votes": dict(text_counts),
        "consensus_votes": consensus_count,
        "total_votes": len(successful),
        "agreement_rate": vote_proportion,
    }

    return MultiBackendOCRResponse(
        results=results,
        consensus_text=consensus_text,
        consensus_confidence=consensus_confidence,
        voting_details=voting_details,
        disagreements=disagreements,
        total_backends=len(backends),
        successful_backends=len(successful),
    )

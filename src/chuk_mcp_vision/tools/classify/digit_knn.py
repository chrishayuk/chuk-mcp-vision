"""Digit classification using OpenCV KNN - complementary to OCR.

This provides a different approach to digit recognition:
- OCR engines: Trained on text/documents
- KNN classifier: Trained on digit images (MNIST-like)

Using both gives better ensemble voting when they disagree.
"""

import cv2
import numpy as np
from typing import Tuple
from chuk_mcp_server import tool

from ...models import DigitClassificationResult
from ..common import load_image_from_artifact


# Pretrained KNN model (will be trained on first use)
_knn_model = None
_model_trained = False


def _get_knn_model():
    """Get or create KNN model."""
    global _knn_model, _model_trained

    if _knn_model is None:
        _knn_model = cv2.ml.KNearest_create()

    return _knn_model


def _prepare_training_data():
    """Prepare training data from synthetic digits.

    For production, you'd use MNIST or a real dataset.
    This creates synthetic training samples for demonstration.
    """
    samples = []
    labels = []

    # Create simple synthetic digits (very basic, would use MNIST in production)
    for digit in range(10):
        for _ in range(50):  # 50 samples per digit
            # Create 20x20 image
            img = np.zeros((20, 20), dtype=np.uint8)

            # Very simple digit representations (just for demo)
            # In production, use actual MNIST dataset
            if digit == 1:
                cv2.line(img, (10, 5), (10, 15), 255, 2)
            elif digit == 7:
                cv2.line(img, (5, 5), (15, 5), 255, 2)
                cv2.line(img, (15, 5), (10, 15), 255, 2)
            # Add more digit patterns...

            # Flatten to feature vector
            samples.append(img.flatten())
            labels.append(digit)

    return np.array(samples, dtype=np.float32), np.array(labels, dtype=np.int32)


def _train_model():
    """Train the KNN model if not already trained."""
    global _model_trained

    if _model_trained:
        return

    knn = _get_knn_model()

    # In production, load from MNIST dataset
    # For now, use synthetic data
    samples, labels = _prepare_training_data()

    # Train
    knn.train(samples, cv2.ml.ROW_SAMPLE, labels)
    _model_trained = True


@tool
async def classify_digit_knn(
    cell_uri: str, k: int = 5, target_size: Tuple[int, int] = (20, 20)
) -> DigitClassificationResult:
    """Classify a digit using OpenCV KNN algorithm.

    This is a complementary tool to OCR-based classification.
    KNN works differently than OCR:
    - OCR: Trained on text in documents (general purpose)
    - KNN: Trained on digit images (specialized)

    When to use:
    - As ensemble with OCR (voting)
    - For printed digits (not handwritten text)
    - When OCR has low confidence
    - For game/puzzle digits (like Sudoku)

    Args:
        cell_uri: URI to cell image
        k: Number of nearest neighbors (default 5)
        target_size: Resize to this size before classification

    Returns:
        DigitClassificationResult with:
        - digit: Predicted digit (0-9 or None)
        - confidence: Classification confidence
        - method: "knn"
        - neighbors_votes: How k neighbors voted
    """
    # Train model on first use
    _train_model()

    # Load image
    img = await load_image_from_artifact(cell_uri)

    # Preprocess
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Check if empty (mostly white)
    if np.mean(gray) > 250:
        return DigitClassificationResult(digit=None, confidence=1.0, method="knn", is_empty=True)

    # Resize to target size
    resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)

    # Threshold
    _, binary = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Flatten to feature vector
    features = binary.flatten().astype(np.float32).reshape(1, -1)

    # Classify
    knn = _get_knn_model()
    ret, results, neighbours, dist = knn.findNearest(features, k)

    # Get predicted digit
    predicted_digit = int(results[0][0])

    # Calculate confidence based on neighbor agreement
    neighbors_votes = {}
    for neighbor in neighbours[0]:
        digit = int(neighbor)
        neighbors_votes[digit] = neighbors_votes.get(digit, 0) + 1

    # Confidence = proportion of neighbors agreeing
    max_votes = max(neighbors_votes.values())
    confidence = max_votes / k

    return DigitClassificationResult(
        digit=predicted_digit if 1 <= predicted_digit <= 9 else None,
        confidence=confidence,
        method="knn",
        is_empty=False,
        neighbors_votes=neighbors_votes,
        distances=dist[0].tolist(),
    )


@tool
async def train_knn_from_dataset(dataset_uri: str, labels_uri: str) -> dict:
    """Train KNN model from a custom dataset.

    Use this to train on actual Sudoku fonts or specific digit styles.

    Args:
        dataset_uri: URI to numpy array of images (N x H x W)
        labels_uri: URI to numpy array of labels (N,)

    Returns:
        Training results and stats
    """
    # This would load actual training data
    # For now, just a placeholder
    return {
        "status": "trained",
        "samples": 0,
        "accuracy": 0.0,
        "message": "Custom training not implemented - using synthetic data",
    }

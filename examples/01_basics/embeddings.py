#!/usr/bin/env python3
"""Example: Image Embeddings - Generate vector embeddings for image similarity.

This example demonstrates the embed_image tool which converts images into
vector embeddings. These embeddings can be used for:
- Image similarity search
- Duplicate detection
- Image clustering
- Visual search
- Content-based retrieval

Tools used:
- embed_image: Generate image embeddings
"""

import asyncio
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from chuk_mcp_vision.tools.embeddings.tool import embed_image


def cosine_similarity(vec1: list, vec2: list) -> float:
    """Calculate cosine similarity between two vectors."""
    a = np.array(vec1)
    b = np.array(vec2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


async def demo_embeddings():
    """Demonstrate image embeddings and similarity."""
    print("=" * 70)
    print("Image Embeddings Demo")
    print("=" * 70)
    print()
    print("⚠️  NOTE: Embedding tool currently returns placeholder vectors.")
    print("    Real embedding models (EVA-02, Jina CLIP) are TODO.")
    print()
    print("Generating vector embeddings for images...")
    print()

    # Load image
    image_path = Path(__file__).parent.parent.parent / "assets" / "sudoku.jpeg"
    if not image_path.exists():
        print(f"❌ Error: {image_path} not found")
        return

    print(f"📸 Image: {image_path.name}")
    print()

    # Generate embedding
    print("Step 1: Generate Embedding")
    print("-" * 70)
    result = await embed_image(str(image_path))

    print("✅ Embedding generated:")
    print(f"   Dimensions: {result.dimension}")
    print(f"   Model: {result.model_name}")
    print(f"   Backend: {result.backend_used}")
    print(f"   Vector length: {len(result.embedding)}")
    print()

    # Show first few values
    print("First 10 embedding values:")
    print(f"   {result.embedding[:10]}")
    print()

    # Generate embedding again (should be identical)
    print("Step 2: Verify Consistency")
    print("-" * 70)
    result2 = await embed_image(str(image_path))

    similarity = cosine_similarity(result.embedding, result2.embedding)
    print("✅ Same image embedded twice:")
    print(f"   Cosine similarity: {similarity:.6f}")
    print("   Expected: 1.000000 (identical)")
    print()

    # Vector properties
    print("Step 3: Vector Analysis")
    print("-" * 70)
    vec = np.array(result.embedding)
    print("Vector statistics:")
    print(f"   Mean: {vec.mean():.6f}")
    print(f"   Std dev: {vec.std():.6f}")
    print(f"   Min: {vec.min():.6f}")
    print(f"   Max: {vec.max():.6f}")
    print(f"   Norm (L2): {np.linalg.norm(vec):.6f}")
    print()

    print("💡 What are embeddings?")
    print("-" * 70)
    print("Embeddings are dense vector representations of images that capture")
    print("semantic meaning. Similar images have similar embeddings (measured")
    print("by cosine similarity or Euclidean distance).")
    print()

    print("Use cases for image embeddings:")
    print("  • Reverse image search (find similar images)")
    print("  • Duplicate detection (identify near-duplicates)")
    print("  • Image clustering (group similar images)")
    print("  • Content-based retrieval (search by visual similarity)")
    print("  • Visual recommendation systems")
    print("  • Anomaly detection (find outliers)")
    print()

    print("Next steps:")
    print("  • Compare embeddings of different images")
    print("  • Build a vector database (Pinecone, Weaviate, ChromaDB)")
    print("  • Implement k-NN search for similarity matching")
    print("  • Use embeddings as features for ML models")
    print()


if __name__ == "__main__":
    asyncio.run(demo_embeddings())

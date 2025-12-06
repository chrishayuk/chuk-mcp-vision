"""Common utilities shared across vision tools using chuk-artifacts.

All intermediate images are stored as BLOBs in chuk-artifacts for pipeline composition.
Images are session-scoped by default (auto-cleanup when session ends).
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from chuk_artifacts import ArtifactStore, NamespaceType, StorageScope

# Global artifact store instance for vision artifacts
# This will be initialized when the server starts
_store: Optional[ArtifactStore] = None


async def init_vision_store() -> ArtifactStore:
    """Initialize the global artifact store for vision artifacts.

    Uses memory provider by default for fast, ephemeral storage.
    Images are stored as SESSION-scoped BLOBs with automatic cleanup.

    Returns:
        The initialized artifact store
    """
    global _store

    if _store is not None:
        return _store

    # Initialize with memory provider (fast, ephemeral)
    _store = ArtifactStore(
        storage_provider="vfs-memory",  # In-memory VFS
        session_provider="memory",  # In-memory sessions
    )

    return _store


async def get_vision_store() -> ArtifactStore:
    """Get the global artifact store, initializing if needed.

    Returns:
        The artifact store instance

    Raises:
        RuntimeError: If store cannot be initialized
    """
    global _store

    if _store is None:
        _store = await init_vision_store()

    return _store


async def load_image_from_artifact(image_ref: str) -> np.ndarray:
    """Load an image from artifact store or local file path.

    Args:
        image_ref: Artifact namespace ID or local file path

    Returns:
        OpenCV image (numpy array in BGR format)

    Raises:
        FileNotFoundError: If image doesn't exist
        ValueError: If image cannot be loaded
    """
    store = await get_vision_store()

    # Check if it's an artifact ID (namespace_id format: "ns-xxxx" or uuid format)
    # Try artifact store first if it looks like an ID (not a file path)
    if not image_ref.startswith("/") and not image_ref.startswith("."):
        try:
            # Try to read from artifact store
            image_bytes = await store.read_namespace(image_ref)

            # Decode bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                raise ValueError(f"Failed to decode image from artifact: {image_ref}")

            return img
        except Exception:
            # If not found in artifacts, try as file path
            pass

    # Try as local file path
    path = Path(image_ref)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {image_ref}")

    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Failed to load image: {image_ref}")

    return img


async def save_image_as_artifact(
    img: np.ndarray, suffix: str = ".jpg", category: str = "output", metadata: dict = None
) -> str:
    """Save an image as a BLOB artifact and return its namespace ID.

    Args:
        img: OpenCV image (numpy array)
        suffix: File suffix (e.g., ".jpg", ".png")
        category: Category tag for metadata (e.g., "output", "transformed", "regions")
        metadata: Additional metadata to store with the image

    Returns:
        Namespace ID (UUID) of the stored image artifact
    """
    store = await get_vision_store()

    # Encode image to bytes
    success, encoded = cv2.imencode(suffix, img)
    if not success:
        raise ValueError(f"Failed to encode image with suffix {suffix}")

    image_bytes = encoded.tobytes()

    # Get image dimensions for metadata
    height, width = img.shape[:2]
    channels = img.shape[2] if len(img.shape) > 2 else 1

    # Build metadata
    artifact_metadata = {
        "category": category,
        "format": suffix.lstrip("."),
        "width": width,
        "height": height,
        "channels": channels,
        "size_bytes": len(image_bytes),
    }

    if metadata:
        artifact_metadata.update(metadata)

    # Create BLOB namespace (single-file artifact)
    namespace = await store.create_namespace(
        type=NamespaceType.BLOB,
        scope=StorageScope.SESSION,  # Ephemeral, auto-cleanup
    )

    # Write image data to the BLOB
    await store.write_namespace(namespace.namespace_id, data=image_bytes)

    # Update metadata
    # TODO: Add metadata support to namespace API
    # For now, metadata is stored but not retrieved

    return namespace.namespace_id

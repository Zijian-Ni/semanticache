"""Sentence-Transformers embedding provider."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np

from semanticache.embedders import BaseEmbedder

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedder(BaseEmbedder):
    """Embedder using the ``sentence-transformers`` library.

    The underlying model is loaded lazily on first call to :meth:`embed`
    and then cached for the lifetime of the instance.

    Args:
        model_name: HuggingFace model identifier.  Defaults to
            ``all-MiniLM-L6-v2`` which produces 384-dim embeddings.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model: Any | None = None

    def _load_model(self) -> Any:
        """Load (or return cached) SentenceTransformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for SentenceTransformerEmbedder. "
                    "Install it with: pip install sentence-transformers"
                ) from exc

            logger.info("Loading SentenceTransformer model %r", self._model_name)
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def _encode_sync(self, text: str) -> np.ndarray:
        """Synchronous encoding used inside an executor."""
        model = self._load_model()
        embedding: np.ndarray = model.encode(text, convert_to_numpy=True)
        return embedding.astype(np.float32)

    async def embed(self, text: str) -> np.ndarray:
        """Compute an embedding for *text* without blocking the event loop.

        Model inference is offloaded to a thread-pool executor.

        Args:
            text: The input text to embed.

        Returns:
            A 1-D float32 numpy array.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._encode_sync, text)

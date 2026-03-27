"""Tests for embedding providers."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# SentenceTransformerEmbedder
# ---------------------------------------------------------------------------

class TestSentenceTransformerEmbedder:
    @pytest.fixture
    def mock_model(self):
        """Create a mock SentenceTransformer model that returns a 384-dim vector."""
        model = MagicMock()
        model.encode.return_value = np.random.default_rng(0).standard_normal(384).astype(
            np.float32
        )
        return model

    @patch("semanticache.embedders.sentence_transformers.SentenceTransformer", create=True)
    async def test_embed_returns_ndarray(self, mock_st_class, mock_model):
        """Embed should return a 1-D float32 numpy array."""
        # Patch the import inside the module
        with patch(
            "semanticache.embedders.sentence_transformers.SentenceTransformer",
            create=True,
        ) as mock_cls:
            mock_cls.return_value = mock_model

            from semanticache.embedders.sentence_transformers import (
                SentenceTransformerEmbedder,
            )

            embedder = SentenceTransformerEmbedder(model_name="test-model")
            # Inject the mock model directly
            embedder._model = mock_model

            result = await embedder.embed("Hello world")

            assert isinstance(result, np.ndarray)
            assert result.dtype == np.float32
            assert result.ndim == 1

    async def test_embed_calls_model_encode(self):
        """Embed should call the model's encode method with the input text."""
        from semanticache.embedders.sentence_transformers import (
            SentenceTransformerEmbedder,
        )

        mock_model = MagicMock()
        mock_model.encode.return_value = np.ones(384, dtype=np.float32)

        embedder = SentenceTransformerEmbedder()
        embedder._model = mock_model

        await embedder.embed("test prompt")

        mock_model.encode.assert_called_once_with("test prompt", convert_to_numpy=True)

    async def test_embed_output_shape(self):
        """Output shape should match the model's embedding dimension."""
        from semanticache.embedders.sentence_transformers import (
            SentenceTransformerEmbedder,
        )

        dim = 768
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros(dim, dtype=np.float32)

        embedder = SentenceTransformerEmbedder()
        embedder._model = mock_model

        result = await embedder.embed("test")
        assert result.shape == (dim,)

    def test_import_error_when_library_missing(self):
        """Should raise ImportError if sentence_transformers is not installed."""
        from semanticache.embedders.sentence_transformers import (
            SentenceTransformerEmbedder,
        )

        embedder = SentenceTransformerEmbedder()
        embedder._model = None  # Force re-load

        with patch.dict("sys.modules", {"sentence_transformers": None}):
            with pytest.raises(ImportError, match="sentence-transformers"):
                embedder._load_model()


# ---------------------------------------------------------------------------
# OpenAIEmbedder (mock openai client)
# ---------------------------------------------------------------------------

class TestOpenAIEmbedder:
    """Tests for an OpenAI-based embedder.

    Since the OpenAIEmbedder module may not exist yet, these tests
    demonstrate the expected interface using a mock implementation.
    """

    def _make_openai_embedder(self, mock_client):
        """Build a minimal OpenAI embedder using the BaseEmbedder interface."""
        from semanticache.embedders import BaseEmbedder

        class _MockOpenAIEmbedder(BaseEmbedder):
            def __init__(self, client):
                self.client = client

            async def embed(self, text: str) -> np.ndarray:
                response = await self.client.embeddings.create(
                    model="text-embedding-3-small",
                    input=text,
                )
                return np.array(response.data[0].embedding, dtype=np.float32)

        return _MockOpenAIEmbedder(mock_client)

    async def test_embed_returns_ndarray(self):
        mock_embedding = MagicMock()
        mock_embedding.embedding = np.random.default_rng(0).standard_normal(1536).tolist()

        mock_response = MagicMock()
        mock_response.data = [mock_embedding]

        mock_client = MagicMock()
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        embedder = self._make_openai_embedder(mock_client)
        result = await embedder.embed("test prompt")

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.ndim == 1

    async def test_embed_output_shape(self):
        dim = 1536
        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.0] * dim

        mock_response = MagicMock()
        mock_response.data = [mock_embedding]

        mock_client = MagicMock()
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        embedder = self._make_openai_embedder(mock_client)
        result = await embedder.embed("test")

        assert result.shape == (dim,)

    async def test_embed_calls_client(self):
        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.1] * 1536

        mock_response = MagicMock()
        mock_response.data = [mock_embedding]

        mock_client = MagicMock()
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        embedder = self._make_openai_embedder(mock_client)
        await embedder.embed("hello world")

        mock_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input="hello world",
        )

    async def test_embed_error_handling(self):
        mock_client = MagicMock()
        mock_client.embeddings.create = AsyncMock(
            side_effect=RuntimeError("API error")
        )

        embedder = self._make_openai_embedder(mock_client)

        with pytest.raises(RuntimeError, match="API error"):
            await embedder.embed("test")

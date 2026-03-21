from unittest.mock import MagicMock, patch

import pytest

from app.retrieval.embeddings import get_embeddings


def test_get_embeddings_openai() -> None:
    with patch("app.retrieval.embeddings.OpenAIEmbeddings") as mock_cls:
        mock_cls.return_value = MagicMock()
        result = get_embeddings("openai", "text-embedding-3-small")
        mock_cls.assert_called_once_with(model="text-embedding-3-small")
        assert result is mock_cls.return_value


def test_get_embeddings_sentence_transformers() -> None:
    with patch("langchain_community.embeddings.HuggingFaceEmbeddings") as mock_cls:
        mock_cls.return_value = MagicMock()
        result = get_embeddings("sentence-transformers", "all-MiniLM-L6-v2")
        mock_cls.assert_called_once_with(model_name="all-MiniLM-L6-v2")
        assert result is mock_cls.return_value


def test_get_embeddings_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported embedding provider"):
        get_embeddings("unknown", "some-model")

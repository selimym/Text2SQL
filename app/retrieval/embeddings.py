from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings


def get_embeddings(provider: str, model: str) -> Embeddings:
    """Return an Embeddings instance for the specified provider and model."""
    if provider == "openai":
        return OpenAIEmbeddings(model=model)
    if provider == "sentence-transformers":
        return HuggingFaceEmbeddings(model_name=model)
    raise ValueError(f"Unsupported embedding provider: {provider!r}")

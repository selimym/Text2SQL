from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI


def get_llm(provider: str, model: str) -> BaseChatModel:
    if provider == "anthropic":
        return ChatAnthropic(model=model)
    if provider == "openai":
        return ChatOpenAI(model=model)
    raise ValueError(f"Unsupported LLM provider: {provider!r}. Use 'anthropic' or 'openai'.")

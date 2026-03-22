from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: str = "anthropic"
    llm_model: str = "claude-3-5-sonnet-20241022"
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    chroma_persist_dir: str = ".chroma"
    spider_data_dir: str = "spider_data"
    log_level: str = "INFO"
    max_result_rows: int = 100
    query_timeout_seconds: int = 30
    postgres_profiler_url: str = "postgresql://text2sql:text2sql@localhost:5433/text2sql_profiler"
    pipeline_variant: Literal["baseline", "deterministic", "agent"] = "baseline"
    max_retries: int = 2
    agent_max_iterations: int = 10
    langsmith_project: str = "text2sql"
    max_schema_docs: int = 10
    max_example_docs: int = 5


def get_settings() -> AppSettings:
    return AppSettings()

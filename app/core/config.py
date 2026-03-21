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


def get_settings() -> AppSettings:
    return AppSettings()

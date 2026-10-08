from pydantic_settings import BaseSettings, SettingsConfigDict
class Settings(BaseSettings):
    app_name: str = "Production RAG AI System"
    environment: str = "development"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    redis_url: str = "redis://localhost:6379/0"
    top_k: int = 4
    chunk_size: int = 900
    chunk_overlap: int = 120
    max_upload_mb: int = 10
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
settings = Settings()

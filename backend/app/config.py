from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Production RAG AI System"
    environment: str = "development"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    redis_url: str = "redis://localhost:6379/0"
    top_k: int = Field(default=4, ge=1, le=20)
    chunk_size: int = Field(default=900, ge=100, le=10000)
    chunk_overlap: int = Field(default=120, ge=0, le=9999)
    max_upload_mb: int = Field(default=10, ge=1, le=100)

    @model_validator(mode="after")
    def validate_chunk_configuration(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

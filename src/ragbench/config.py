from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"),
        env_file_encoding="utf-8",
        env_prefix="RAGBENCH_",
        extra="ignore",
    )

    provider: Literal["demo", "openai"] = "demo"
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    chat_model: str = "gpt-5-mini"
    embedding_model: str = "text-embedding-3-small"
    store_backend: Literal["memory", "qdrant"] = "qdrant"
    retrieval_mode: Literal["vector", "hybrid"] = "vector"
    qdrant_path: str = ".ragbench/qdrant"
    qdrant_collection: str = "ragbench_chunks"
    experiment_db_path: str = ".ragbench/experiments.sqlite3"
    chunk_size: int = Field(default=120, ge=30, le=2_000)
    chunk_overlap: int = Field(default=20, ge=0, le=500)
    top_k: int = Field(default=3, ge=1, le=20)
    max_upload_bytes: int = Field(default=10_485_760, ge=1024, le=104_857_600)

    @model_validator(mode="after")
    def validate_provider(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when RAGBENCH_PROVIDER=openai")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()

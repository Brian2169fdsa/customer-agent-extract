"""Application settings loaded from environment variables via pydantic-settings."""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """All configuration for the Customer Agent Extraction Service."""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # ═══════ ANTHROPIC ═══════
    ANTHROPIC_API_KEY: str = Field(description="Anthropic API key")

    # ═══════ EXTRACTION CONFIG ═══════
    EXTRACTION_MODEL: str = Field(default="claude-sonnet-4-5-20250929")
    VERIFICATION_MODEL: str = Field(default="claude-haiku-4-5-20251001")
    MAX_EXTRACTION_TOKENS: int = Field(default=8000)
    EXTRACTION_TEMPERATURE: float = Field(default=0.0)

    # ═══════ CHUNKING CONFIG ═══════
    CHUNK_MAX_TOKENS: int = Field(default=6000)
    CHUNK_OVERLAP_TOKENS: int = Field(default=500)
    CHUNK_MIN_TOKENS: int = Field(default=500)

    # ═══════ MULTI-PASS CONFIG ═══════
    MULTI_PASS_DEFAULT: bool = Field(default=False)
    MAX_PASSES: int = Field(default=2)

    # ═══════ SERVICE CONFIG ═══════
    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(default=8100)
    SERVICE_API_KEY: str = Field(default="REPLACE_WITH_RANDOM_SECRET")
    MAX_CONCURRENT: int = Field(default=5)


settings = Settings()

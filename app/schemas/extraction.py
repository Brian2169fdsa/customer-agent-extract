"""Request and response models for the /extract endpoint."""

from typing import Literal, Optional

from pydantic import BaseModel, Field

from .fact import Fact


class ExtractionRequest(BaseModel):
    """Request body for the /extract endpoint."""

    content: str = Field(description="Raw interaction content to process")
    content_type: Literal[
        "transcript", "email", "chat", "pipedrive_activity", "pipedrive_deal"
    ] = Field(default="transcript")
    interaction_id: Optional[str] = Field(
        default=None, description="Notion page ID for reference"
    )
    multi_pass: Optional[bool] = Field(
        default=None, description="Override multi-pass setting for this request"
    )
    priority: Literal["normal", "high"] = Field(
        default="normal",
        description="High priority = multi-pass + deeper extraction",
    )


class ExtractionMetadata(BaseModel):
    """Metadata about the extraction process."""

    content_length: int = Field(description="Length of input content in characters")
    token_count: int = Field(description="Approximate token count of input")
    chunks_processed: int = Field(
        description="Number of chunks (1 if no chunking needed)"
    )
    passes_completed: int = Field(
        description="Number of extraction passes (1 or 2)"
    )
    facts_extracted: int
    facts_with_grounding: int = Field(
        description="Facts that have character-level source grounding"
    )
    action_items_extracted: int
    model_used: str
    processing_time_seconds: float
    cost_estimate_usd: float = Field(
        description="Estimated API cost for this extraction"
    )


class ExtractionResponse(BaseModel):
    """Response body from the /extract endpoint."""

    success: bool
    facts: list[Fact]
    action_items: list[dict]  # Serialized ActionItems
    summary: str
    metadata: ExtractionMetadata

"""Models for source grounding results used by the fallback grounder."""

from typing import Optional

from pydantic import BaseModel, Field


class GroundingResult(BaseModel):
    """A single grounding result mapping a fact to its source location."""

    fact_index: int = Field(description="1-based index of the fact being grounded")
    start_offset: Optional[int] = Field(
        default=None, description="Start character offset in source document"
    )
    end_offset: Optional[int] = Field(
        default=None, description="End character offset in source document"
    )
    source_text: Optional[str] = Field(
        default=None, description="The exact text span from the source"
    )

"""Pydantic models for Facts, ActionItems, and ExtractionResult.

These models are the heart of the system. Instructor uses them to guarantee
every extraction matches the schema. If Claude returns an invalid category,
Instructor catches the validation error and retries automatically.
"""

from datetime import date
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class FactCategory(str, Enum):
    """Valid categories for extracted facts."""

    COMMITMENT = "Commitment"
    DECISION = "Decision"
    CONCERN = "Concern"
    FEEDBACK = "Feedback"
    PRIORITY = "Priority"
    PREFERENCE = "Preference"
    CONTEXT = "Context"
    REQUEST = "Request"
    MILESTONE = "Milestone"


class Confidence(str, Enum):
    """Confidence level of an extracted fact."""

    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class Sentiment(str, Enum):
    """Sentiment attached to a fact."""

    POSITIVE = "Positive"
    NEUTRAL = "Neutral"
    NEGATIVE = "Negative"
    URGENT = "Urgent"


class SourceGrounding(BaseModel):
    """Character-level source grounding for a fact."""

    start_offset: int = Field(description="Start character offset in source document")
    end_offset: int = Field(description="End character offset in source document")
    source_text: str = Field(
        description="The exact text span from the source that supports this fact"
    )


class Fact(BaseModel):
    """A single structured fact extracted from an interaction."""

    claim: str = Field(description="Concise factual statement that stands alone")
    category: FactCategory
    relationship: str = Field(
        description="Person or company name this fact belongs to"
    )
    speaker: str = Field(description="Who said or wrote this")
    timestamp: str = Field(
        description=(
            "HH:MM for transcripts, 'paragraph N' for emails, "
            "'activity' for Pipedrive"
        )
    )
    confidence: Confidence
    sentiment: Sentiment
    due_date: Optional[date] = Field(
        default=None, description="For commitments/requests with deadlines"
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Topic tags: pricing, product, timeline, etc.",
    )

    # Source grounding — populated by the grounding pass
    grounding: Optional[SourceGrounding] = Field(
        default=None, description="Character-level source location"
    )

    @field_validator("claim")
    @classmethod
    def claim_not_empty(cls, v: str) -> str:
        """Claim must be at least 5 characters — be specific."""
        if len(v.strip()) < 5:
            raise ValueError("Claim must be at least 5 characters — be specific")
        return v.strip()

    @field_validator("relationship")
    @classmethod
    def relationship_not_empty(cls, v: str) -> str:
        """Relationship must identify a person or company."""
        if len(v.strip()) < 1:
            raise ValueError("Relationship must identify a person or company")
        return v.strip()


class ActionItem(BaseModel):
    """An action item extracted from the interaction."""

    title: str = Field(description="Specific actionable task")
    owner: str = Field(default="Brian", description="Person responsible")
    priority: Literal["P1", "P2", "P3"] = Field(default="P2")
    due_date: Optional[date] = None
    relationship: str = Field(description="Related person or company")


class ExtractionResult(BaseModel):
    """Complete extraction output from a single pass."""

    facts: list[Fact] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    summary: str = Field(description="One paragraph summary, 2-4 sentences")

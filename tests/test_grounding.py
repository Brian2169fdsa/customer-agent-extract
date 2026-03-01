"""Tests for source grounding (mapping facts to character offsets)."""

import pytest

from app.schemas.fact import Confidence, Fact, FactCategory, Sentiment, SourceGrounding
from app.schemas.grounding import GroundingResult


class TestSourceGrounding:
    """Test the SourceGrounding model."""

    def test_valid_grounding(self):
        """Valid grounding with correct offsets."""
        grounding = SourceGrounding(
            start_offset=100,
            end_offset=150,
            source_text="We've approved a budget of $50,000",
        )
        assert grounding.end_offset > grounding.start_offset
        assert len(grounding.source_text) > 0

    def test_grounding_attached_to_fact(self):
        """Grounding should attach correctly to a fact."""
        source = "I'm really worried we won't make the Q2 deadline for this project."
        fact = Fact(
            claim="Customer concerned about Q2 deadline",
            category=FactCategory.CONCERN,
            relationship="Acme Corp",
            speaker="Dave",
            timestamp="14:30",
            confidence=Confidence.HIGH,
            sentiment=Sentiment.NEGATIVE,
            grounding=SourceGrounding(
                start_offset=0,
                end_offset=len(source),
                source_text=source,
            ),
        )
        assert fact.grounding is not None
        assert fact.grounding.source_text == source


class TestGroundingResult:
    """Test the GroundingResult model used by the fallback grounder."""

    def test_valid_result(self):
        """Valid grounding result with all fields."""
        result = GroundingResult(
            fact_index=1,
            start_offset=50,
            end_offset=100,
            source_text="exact quote from source",
        )
        assert result.fact_index == 1

    def test_null_grounding(self):
        """Facts that can't be grounded should have None fields."""
        result = GroundingResult(
            fact_index=3,
            start_offset=None,
            end_offset=None,
            source_text=None,
        )
        assert result.start_offset is None

    def test_grounding_offset_accuracy(self):
        """Grounding offsets should correctly index into the source text."""
        source = "The budget is $50,000 and the timeline is Q2 2026."
        start = source.index("$50,000")
        end = start + len("$50,000")

        result = GroundingResult(
            fact_index=1,
            start_offset=start,
            end_offset=end,
            source_text=source[start:end],
        )
        assert result.source_text == "$50,000"
        assert source[result.start_offset : result.end_offset] == "$50,000"

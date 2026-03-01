"""Tests for the core extraction engine (Instructor + Claude)."""

import pytest

from app.schemas.fact import (
    ActionItem,
    Confidence,
    ExtractionResult,
    Fact,
    FactCategory,
    Sentiment,
    SourceGrounding,
)


class TestFactModel:
    """Test Pydantic model validation for Fact."""

    def test_valid_fact(self):
        """A well-formed fact should pass validation."""
        fact = Fact(
            claim="Customer wants to launch by Q2 2026",
            category=FactCategory.COMMITMENT,
            relationship="Acme Corp",
            speaker="Dave",
            timestamp="14:23",
            confidence=Confidence.HIGH,
            sentiment=Sentiment.POSITIVE,
            tags=["timeline"],
        )
        assert fact.claim == "Customer wants to launch by Q2 2026"
        assert fact.category == FactCategory.COMMITMENT

    def test_claim_too_short_raises(self):
        """Claims under 5 characters should fail validation."""
        with pytest.raises(ValueError, match="at least 5 characters"):
            Fact(
                claim="Hi",
                category=FactCategory.CONTEXT,
                relationship="Acme",
                speaker="Dave",
                timestamp="14:00",
                confidence=Confidence.LOW,
                sentiment=Sentiment.NEUTRAL,
            )

    def test_empty_relationship_raises(self):
        """Empty relationship field should fail validation."""
        with pytest.raises(ValueError, match="identify a person or company"):
            Fact(
                claim="Budget is $50,000 for Phase 1",
                category=FactCategory.CONTEXT,
                relationship="",
                speaker="Dave",
                timestamp="14:00",
                confidence=Confidence.HIGH,
                sentiment=Sentiment.NEUTRAL,
            )

    def test_invalid_category_raises(self):
        """Invalid category should fail Pydantic validation."""
        with pytest.raises(ValueError):
            Fact(
                claim="Customer expressed interest in premium plan",
                category="InvalidCategory",
                relationship="Acme Corp",
                speaker="Dave",
                timestamp="14:00",
                confidence=Confidence.MEDIUM,
                sentiment=Sentiment.POSITIVE,
            )

    def test_fact_with_grounding(self):
        """Facts can have optional source grounding attached."""
        fact = Fact(
            claim="Budget confirmed at $50,000",
            category=FactCategory.DECISION,
            relationship="Acme Corp",
            speaker="Sarah",
            timestamp="15:00",
            confidence=Confidence.HIGH,
            sentiment=Sentiment.POSITIVE,
            grounding=SourceGrounding(
                start_offset=100,
                end_offset=150,
                source_text="We've approved a budget of $50,000",
            ),
        )
        assert fact.grounding is not None
        assert fact.grounding.start_offset == 100

    def test_fact_without_grounding(self):
        """Facts without grounding should default to None."""
        fact = Fact(
            claim="Customer prefers monthly billing",
            category=FactCategory.PREFERENCE,
            relationship="Acme Corp",
            speaker="Dave",
            timestamp="14:30",
            confidence=Confidence.MEDIUM,
            sentiment=Sentiment.NEUTRAL,
        )
        assert fact.grounding is None


class TestActionItemModel:
    """Test Pydantic model validation for ActionItem."""

    def test_valid_action_item(self):
        """A well-formed action item should pass validation."""
        item = ActionItem(
            title="Send revised proposal to Acme Corp",
            owner="Brian",
            priority="P1",
            relationship="Acme Corp",
        )
        assert item.priority == "P1"
        assert item.owner == "Brian"

    def test_default_owner(self):
        """Owner should default to Brian."""
        item = ActionItem(
            title="Follow up on pricing discussion",
            relationship="Acme Corp",
        )
        assert item.owner == "Brian"

    def test_invalid_priority_raises(self):
        """Priority must be P1, P2, or P3."""
        with pytest.raises(ValueError):
            ActionItem(
                title="Send proposal",
                priority="P4",
                relationship="Acme Corp",
            )


class TestExtractionResult:
    """Test the complete ExtractionResult model."""

    def test_empty_result(self):
        """Empty extraction result should be valid."""
        result = ExtractionResult(
            facts=[],
            action_items=[],
            summary="No significant facts found in this interaction.",
        )
        assert len(result.facts) == 0
        assert len(result.action_items) == 0

    def test_serialization_roundtrip(self):
        """ExtractionResult should serialize and deserialize cleanly."""
        result = ExtractionResult(
            facts=[
                Fact(
                    claim="Customer wants to launch by Q2",
                    category=FactCategory.COMMITMENT,
                    relationship="Acme Corp",
                    speaker="Dave",
                    timestamp="14:23",
                    confidence=Confidence.HIGH,
                    sentiment=Sentiment.POSITIVE,
                )
            ],
            action_items=[],
            summary="Brief discussion about launch timeline.",
        )
        data = result.model_dump()
        restored = ExtractionResult.model_validate(data)
        assert restored.facts[0].claim == result.facts[0].claim

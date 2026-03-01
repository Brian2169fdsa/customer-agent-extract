"""Tests for cross-chunk fact deduplication."""

import pytest

from app.extraction.deduplicator import (
    _normalize_claim,
    _text_similarity,
    deduplicate_action_items,
    deduplicate_facts,
)
from app.schemas.fact import (
    ActionItem,
    Confidence,
    Fact,
    FactCategory,
    Sentiment,
    SourceGrounding,
)


class TestNormalization:
    """Test claim normalization for comparison."""

    def test_lowercase(self):
        assert _normalize_claim("Hello World") == "hello world"

    def test_strip_punctuation(self):
        assert _normalize_claim("Hello, World!") == "hello world"

    def test_collapse_whitespace(self):
        assert _normalize_claim("hello   world") == "hello world"

    def test_strip_outer_whitespace(self):
        assert _normalize_claim("  hello  ") == "hello"


class TestTextSimilarity:
    """Test Levenshtein-based text similarity."""

    def test_identical_strings(self):
        assert _text_similarity("hello", "hello") == 1.0

    def test_completely_different(self):
        sim = _text_similarity("abc", "xyz")
        assert sim < 0.5

    def test_similar_strings(self):
        sim = _text_similarity(
            "customer wants to launch by q2",
            "customer wants to launch by q2 2026",
        )
        assert sim > 0.8

    def test_empty_strings(self):
        assert _text_similarity("", "") == 1.0
        assert _text_similarity("hello", "") == 0.0
        assert _text_similarity("", "hello") == 0.0


def _make_fact(claim: str, category=FactCategory.CONTEXT, relationship="Acme Corp", confidence=Confidence.MEDIUM, grounding=None):
    """Helper to create a Fact with sensible defaults."""
    return Fact(
        claim=claim,
        category=category,
        relationship=relationship,
        speaker="Dave",
        timestamp="14:00",
        confidence=confidence,
        sentiment=Sentiment.NEUTRAL,
        grounding=grounding,
    )


class TestDeduplication:
    """Test fact deduplication across chunks."""

    @pytest.mark.asyncio
    async def test_single_chunk_no_dedup(self):
        """Single chunk should pass through without dedup."""
        facts = [_make_fact("Customer wants premium plan")]
        result = await deduplicate_facts([facts])
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_exact_duplicate_removed(self):
        """Exact same claim across chunks should be deduplicated."""
        fact1 = _make_fact("Budget is $50,000 for Phase 1")
        fact2 = _make_fact("Budget is $50,000 for Phase 1")
        result = await deduplicate_facts([[fact1], [fact2]])
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_similar_claims_deduplicated(self):
        """Very similar claims (same relationship + category) should be merged."""
        fact1 = _make_fact("Customer budget is $50,000 for Phase 1")
        fact2 = _make_fact("Customer budget is $50,000 for Phase one")
        result = await deduplicate_facts([[fact1], [fact2]])
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_different_facts_preserved(self):
        """Genuinely different facts should not be deduplicated."""
        fact1 = _make_fact("Budget is $50,000 for Phase 1")
        fact2 = _make_fact("Timeline is Q2 2026 for launch")
        result = await deduplicate_facts([[fact1], [fact2]])
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_merge_keeps_higher_confidence(self):
        """When merging duplicates, keep the higher confidence fact."""
        fact1 = _make_fact("Budget is $50,000", confidence=Confidence.MEDIUM)
        fact2 = _make_fact("Budget is $50,000", confidence=Confidence.HIGH)
        result = await deduplicate_facts([[fact1], [fact2]])
        assert len(result) == 1
        assert result[0].confidence == Confidence.HIGH

    @pytest.mark.asyncio
    async def test_merge_keeps_grounding(self):
        """When merging, prefer the fact with source grounding."""
        grounding = SourceGrounding(start_offset=10, end_offset=50, source_text="budget is $50,000")
        fact1 = _make_fact("Budget is $50,000", grounding=None)
        fact2 = _make_fact("Budget is $50,000", grounding=grounding)
        result = await deduplicate_facts([[fact1], [fact2]])
        assert len(result) == 1
        assert result[0].grounding is not None

    @pytest.mark.asyncio
    async def test_different_relationships_not_merged(self):
        """Same claim but different relationships should NOT be merged."""
        fact1 = _make_fact("Budget is $50,000", relationship="Acme Corp")
        fact2 = _make_fact("Budget is $50,000", relationship="Beta Inc")
        result = await deduplicate_facts([[fact1], [fact2]])
        assert len(result) == 2


class TestActionItemDedup:
    """Test action item deduplication."""

    def test_unique_items_preserved(self):
        items = [
            ActionItem(title="Send proposal to Acme", relationship="Acme Corp"),
            ActionItem(title="Schedule follow-up call with Beta", relationship="Beta Inc"),
        ]
        result = deduplicate_action_items(items)
        assert len(result) == 2

    def test_duplicate_items_removed(self):
        items = [
            ActionItem(title="Send proposal to Acme Corp", relationship="Acme Corp"),
            ActionItem(title="Send proposal to Acme Corp", relationship="Acme Corp"),
        ]
        result = deduplicate_action_items(items)
        assert len(result) == 1

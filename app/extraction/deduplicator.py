"""Cross-chunk fact deduplication.

When a document is chunked with overlap, the same fact may be extracted from
two adjacent chunks. This module merges duplicates by comparing claims using
normalized text matching and fuzzy similarity.
"""

import logging
import re
import unicodedata

from ..schemas.fact import ActionItem, Fact

logger = logging.getLogger(__name__)


async def deduplicate_facts(chunks_facts: list[list[Fact]]) -> list[Fact]:
    """Merge facts from multiple chunks, removing duplicates from overlap regions.

    Two facts are considered duplicates if:
    - Same relationship AND same category AND normalized claim similarity > 0.85
    - OR exact same normalized claim text AND same relationship

    When merging, keeps the fact with higher confidence and better grounding.
    """
    if len(chunks_facts) == 1:
        return chunks_facts[0]

    all_facts: list[Fact] = []

    for chunk_facts in chunks_facts:
        for fact in chunk_facts:
            # Check for duplicates against existing facts
            is_dup, merge_target = _find_duplicate(fact, all_facts)
            if is_dup and merge_target is not None:
                _merge_facts(merge_target, fact)
                logger.debug(f"Merged duplicate fact: {fact.claim[:50]}...")
            else:
                all_facts.append(fact)

    deduped = sum(len(cf) for cf in chunks_facts) - len(all_facts)
    if deduped > 0:
        logger.info(f"Deduplication removed {deduped} duplicate facts")

    return all_facts


def deduplicate_action_items(action_items: list[ActionItem]) -> list[ActionItem]:
    """Remove duplicate action items based on title similarity."""
    if len(action_items) <= 1:
        return action_items

    unique: list[ActionItem] = []
    seen_titles: set[str] = set()

    for item in action_items:
        normalized = _normalize_claim(item.title)
        if normalized not in seen_titles:
            # Also check fuzzy match
            is_dup = any(
                _text_similarity(normalized, _normalize_claim(u.title)) > 0.85
                for u in unique
            )
            if not is_dup:
                unique.append(item)
                seen_titles.add(normalized)

    return unique


def _normalize_claim(text: str) -> str:
    """Normalize a claim for comparison: lowercase, strip punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _text_similarity(a: str, b: str) -> float:
    """Compute normalized Levenshtein similarity between two strings.

    Returns a value between 0.0 (completely different) and 1.0 (identical).
    Uses the simple ratio: 1 - (edit_distance / max_length).
    """
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0

    # Simple Levenshtein distance
    len_a, len_b = len(a), len(b)
    if len_a > len_b:
        a, b = b, a
        len_a, len_b = len_b, len_a

    current_row = list(range(len_a + 1))
    for i in range(1, len_b + 1):
        previous_row = current_row
        current_row = [i] + [0] * len_a
        for j in range(1, len_a + 1):
            add = previous_row[j] + 1
            delete = current_row[j - 1] + 1
            change = previous_row[j - 1] + (0 if b[i - 1] == a[j - 1] else 1)
            current_row[j] = min(add, delete, change)

    distance = current_row[len_a]
    max_len = max(len_a, len_b)
    return 1.0 - (distance / max_len)


def _find_duplicate(
    fact: Fact, existing: list[Fact]
) -> tuple[bool, Fact | None]:
    """Check if a fact is a duplicate of any existing fact.

    A duplicate requires same relationship AND (same category with high
    similarity OR exact normalized match).
    """
    normalized = _normalize_claim(fact.claim)

    for existing_fact in existing:
        # Must have same relationship for any dedup
        if fact.relationship.lower() != existing_fact.relationship.lower():
            continue

        existing_normalized = _normalize_claim(existing_fact.claim)

        # Exact normalized match
        if normalized == existing_normalized:
            return True, existing_fact

        # Fuzzy match requires same category too
        if fact.category == existing_fact.category:
            similarity = _text_similarity(normalized, existing_normalized)
            if similarity > 0.85:
                return True, existing_fact

    return False, None


def _merge_facts(target: Fact, source: Fact) -> None:
    """Merge a duplicate fact into the target, keeping the better data.

    Prefers: higher confidence, better grounding, earlier timestamp.
    """
    confidence_rank = {"High": 3, "Medium": 2, "Low": 1}

    # Keep higher confidence
    if confidence_rank.get(source.confidence.value, 0) > confidence_rank.get(
        target.confidence.value, 0
    ):
        target.confidence = source.confidence

    # Keep grounding if target doesn't have it
    if target.grounding is None and source.grounding is not None:
        target.grounding = source.grounding

    # Merge tags (union)
    target.tags = list(set(target.tags) | set(source.tags))

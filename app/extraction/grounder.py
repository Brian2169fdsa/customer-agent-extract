"""Source grounding — maps extracted facts back to exact character offsets in the source.

Tries LangExtract first for character-level offset grounding. If LangExtract's
Anthropic provider isn't available, falls back to a direct Claude API call that
finds source spans for each fact. The fallback adds ~$0.01 and ~2s per extraction.
"""

import logging
from typing import Optional

import instructor
from anthropic import Anthropic

from ..config import settings
from ..prompts.grounding import get_grounding_prompt
from ..schemas.fact import Fact, SourceGrounding
from ..schemas.grounding import GroundingResult

logger = logging.getLogger(__name__)

_anthropic_client = Anthropic()
_instructor_client = instructor.from_anthropic(_anthropic_client)


async def ground_facts(
    facts: list[Fact],
    source_text: str,
    content_type: str,
) -> list[Fact]:
    """Map each extracted fact back to its exact location in the source text.

    Tries LangExtract first, falls back to direct Claude grounding.
    """
    if not facts:
        return facts

    try:
        return await _ground_with_langextract(facts, source_text, content_type)
    except Exception as e:
        logger.info(f"LangExtract grounding unavailable ({e}), using fallback")
        return await _ground_with_claude_fallback(facts, source_text)


async def _ground_with_langextract(
    facts: list[Fact],
    source_text: str,
    content_type: str,
) -> list[Fact]:
    """Use LangExtract for character-level offset grounding."""
    import langextract as lx

    prompt = (
        "For each claim provided, find the exact passage in the source text "
        "that supports this claim. Return the verbatim text span. "
        "If the claim is a synthesis of multiple passages, return the most "
        "relevant single passage."
    )

    claims = [f.claim for f in facts]

    result = lx.extract(
        text_or_documents=source_text,
        prompt_description=prompt,
        examples=_build_grounding_examples(content_type),
        model_id="anthropic/claude-sonnet-4-5-20250929",
    )

    # Map LangExtract results back to facts
    for fact in facts:
        match = _find_best_langextract_match(fact.claim, result, source_text)
        if match:
            fact.grounding = SourceGrounding(
                start_offset=match["start"],
                end_offset=match["end"],
                source_text=source_text[match["start"] : match["end"]],
            )

    grounded = sum(1 for f in facts if f.grounding is not None)
    logger.info(f"LangExtract grounded {grounded}/{len(facts)} facts")
    return facts


async def _ground_with_claude_fallback(
    facts: list[Fact],
    source_text: str,
) -> list[Fact]:
    """Fallback: use Claude directly to find source locations for each fact.

    Batches all facts into a single Claude call for efficiency.
    """
    facts_list = "\n".join(
        [f"{i + 1}. {f.claim}" for i, f in enumerate(facts)]
    )

    grounding_results: list[GroundingResult] = _instructor_client.messages.create(
        model=settings.EXTRACTION_MODEL,
        max_tokens=settings.MAX_EXTRACTION_TOKENS,
        temperature=0,
        system=get_grounding_prompt(),
        messages=[
            {
                "role": "user",
                "content": (
                    f"CLAIMS:\n{facts_list}\n\n"
                    f"SOURCE DOCUMENT:\n{source_text}"
                ),
            }
        ],
        response_model=list[GroundingResult],
        max_retries=2,
    )

    # Attach grounding to facts
    for gr in grounding_results:
        if (
            gr
            and gr.fact_index >= 1
            and gr.fact_index <= len(facts)
            and gr.start_offset is not None
            and gr.end_offset is not None
            and gr.source_text is not None
        ):
            facts[gr.fact_index - 1].grounding = SourceGrounding(
                start_offset=gr.start_offset,
                end_offset=gr.end_offset,
                source_text=gr.source_text,
            )

    grounded = sum(1 for f in facts if f.grounding is not None)
    logger.info(f"Claude fallback grounded {grounded}/{len(facts)} facts")
    return facts


def _build_grounding_examples(content_type: str) -> list[dict]:
    """Build few-shot examples for LangExtract based on content type."""
    if content_type == "transcript":
        return [
            {
                "input": "Customer expressed concern about Q2 delivery timeline",
                "output": "I'm really worried we won't make the Q2 deadline",
            }
        ]
    elif content_type == "email":
        return [
            {
                "input": "Budget was confirmed at $50,000 for Phase 1",
                "output": "We've approved a budget of $50,000 for the first phase",
            }
        ]
    return []


def _find_best_langextract_match(
    claim: str,
    result: object,
    source_text: str,
) -> Optional[dict]:
    """Find the best matching extraction from LangExtract results for a claim."""
    try:
        extractions = getattr(result, "extractions", [])
        if not extractions:
            return None

        # Find the extraction most relevant to this claim
        best_match = None
        best_score = 0.0

        for extraction in extractions:
            char_start = getattr(extraction, "char_start", None)
            char_end = getattr(extraction, "char_end", None)
            text = getattr(extraction, "text", "")

            if char_start is None or char_end is None:
                continue

            # Simple relevance: check word overlap between claim and extracted text
            claim_words = set(claim.lower().split())
            text_words = set(text.lower().split())
            if not claim_words:
                continue

            overlap = len(claim_words & text_words) / len(claim_words)
            if overlap > best_score:
                best_score = overlap
                best_match = {"start": char_start, "end": char_end}

        return best_match if best_score > 0.3 else None
    except Exception:
        return None

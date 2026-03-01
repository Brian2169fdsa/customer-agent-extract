"""Second-pass extraction for high-value interactions.

For meetings flagged as high priority, runs a second extraction pass with a
different prompt specifically looking for facts the first pass might have missed.
Uses the verification model (can be Haiku for cost savings).
"""

import logging

import instructor
from anthropic import Anthropic

from ..config import settings
from ..prompts.verification import get_verification_prompt
from ..schemas.fact import ExtractionResult, Fact
from .deduplicator import _normalize_claim, _text_similarity

logger = logging.getLogger(__name__)

_anthropic_client = Anthropic()
_instructor_client = instructor.from_anthropic(_anthropic_client)


async def second_pass_extraction(
    content: str,
    content_type: str,
    first_pass_facts: list[Fact],
) -> list[Fact]:
    """Run a verification pass looking for missed facts.

    Takes the first pass results and the original content, asks Claude to find
    additional facts that were missed. Uses a different prompt focused on
    finding gaps in commitments, concerns, budget/timeline details, and preferences.
    """
    existing_claims = "\n".join(
        [f"- [{f.category.value}] {f.claim}" for f in first_pass_facts]
    )

    logger.info(
        "Running second pass extraction",
        extra={"first_pass_facts": len(first_pass_facts)},
    )

    result: ExtractionResult = _instructor_client.messages.create(
        model=settings.VERIFICATION_MODEL,
        max_tokens=settings.MAX_EXTRACTION_TOKENS,
        temperature=0,
        system=get_verification_prompt(content_type),
        messages=[
            {
                "role": "user",
                "content": (
                    f"ALREADY EXTRACTED FACTS:\n{existing_claims}\n\n"
                    f"SOURCE {content_type.upper()}:\n{content}\n\n"
                    "Find any ADDITIONAL facts that were missed in the first pass. "
                    "Focus on: commitments, concerns, budget numbers, timeline details, "
                    "and preferences that were overlooked."
                ),
            }
        ],
        response_model=ExtractionResult,
        max_retries=2,
    )

    # Filter out facts that duplicate the first pass
    new_facts = [
        f for f in result.facts if not _is_duplicate_of_existing(f, first_pass_facts)
    ]

    logger.info(
        "Second pass complete",
        extra={
            "candidates": len(result.facts),
            "genuinely_new": len(new_facts),
        },
    )

    return new_facts


def _is_duplicate_of_existing(fact: Fact, existing: list[Fact]) -> bool:
    """Check if a fact from the second pass duplicates any first-pass fact."""
    normalized = _normalize_claim(fact.claim)

    for ex in existing:
        ex_normalized = _normalize_claim(ex.claim)
        if _text_similarity(normalized, ex_normalized) > 0.80:
            return True

    return False

"""Full extraction pipeline orchestrator.

Orchestrates the complete flow: chunk -> extract -> ground -> deduplicate ->
multi-pass -> response. This is the single entry point called by the API.
"""

import logging
import time

import instructor
from anthropic import Anthropic

from ..config import settings
from ..schemas.extraction import ExtractionMetadata, ExtractionRequest, ExtractionResponse
from ..schemas.fact import ExtractionResult
from ..utils.tokenizer import estimate_tokens
from .chunker import chunk_content
from .deduplicator import deduplicate_action_items, deduplicate_facts
from .extractor import extract_from_chunk
from .grounder import ground_facts
from .multi_pass import second_pass_extraction

logger = logging.getLogger(__name__)

_anthropic_client = Anthropic()
_instructor_client = instructor.from_anthropic(_anthropic_client)


async def run_extraction_pipeline(
    request: ExtractionRequest,
) -> ExtractionResponse:
    """Full extraction pipeline.

    1. Chunk the content if needed
    2. Extract facts from each chunk (Instructor + Claude)
    3. Deduplicate across chunks
    4. Ground facts to source locations (LangExtract or fallback)
    5. Multi-pass if enabled
    6. Return structured response with metadata
    """
    start_time = time.time()

    # Step 1: Chunk
    chunks = chunk_content(
        content=request.content,
        content_type=request.content_type,
        max_tokens=settings.CHUNK_MAX_TOKENS,
        overlap_tokens=settings.CHUNK_OVERLAP_TOKENS,
    )
    logger.info(f"Content split into {len(chunks)} chunk(s)")

    # Step 2: Extract from each chunk
    chunks_facts = []
    chunks_action_items = []
    summaries = []

    for chunk in chunks:
        result = await extract_from_chunk(
            content=chunk.text,
            content_type=request.content_type,
            chunk_metadata={"index": chunk.index, "total": chunk.total_chunks},
        )
        chunks_facts.append(result.facts)
        chunks_action_items.extend(result.action_items)
        summaries.append(result.summary)

    # Step 3: Deduplicate across chunks
    facts = await deduplicate_facts(chunks_facts)

    # Step 4: Source grounding (best-effort — don't fail the whole extraction)
    try:
        facts = await ground_facts(facts, request.content, request.content_type)
    except Exception as e:
        logger.warning(f"Source grounding failed: {e}")

    # Step 5: Multi-pass if enabled
    use_multi_pass = (
        request.multi_pass
        if request.multi_pass is not None
        else settings.MULTI_PASS_DEFAULT
    )
    if request.priority == "high":
        use_multi_pass = True

    passes_completed = 1
    if use_multi_pass:
        new_facts = await second_pass_extraction(
            content=request.content,
            content_type=request.content_type,
            first_pass_facts=facts,
        )
        if new_facts:
            # Ground the new facts too
            try:
                new_facts = await ground_facts(
                    new_facts, request.content, request.content_type
                )
            except Exception:
                pass
            facts.extend(new_facts)
        passes_completed = 2

    # Step 6: Merge summaries
    if len(summaries) == 1:
        final_summary = summaries[0]
    else:
        final_summary = await _merge_summaries(summaries, request.content_type)

    # Step 7: Deduplicate action items
    action_items = deduplicate_action_items(chunks_action_items)

    # Build response
    processing_time = time.time() - start_time
    grounded_count = sum(1 for f in facts if f.grounding is not None)

    return ExtractionResponse(
        success=True,
        facts=facts,
        action_items=[a.model_dump() for a in action_items],
        summary=final_summary,
        metadata=ExtractionMetadata(
            content_length=len(request.content),
            token_count=estimate_tokens(request.content),
            chunks_processed=len(chunks),
            passes_completed=passes_completed,
            facts_extracted=len(facts),
            facts_with_grounding=grounded_count,
            action_items_extracted=len(action_items),
            model_used=settings.EXTRACTION_MODEL,
            processing_time_seconds=round(processing_time, 2),
            cost_estimate_usd=_estimate_cost(len(chunks), passes_completed),
        ),
    )


async def _merge_summaries(summaries: list[str], content_type: str) -> str:
    """Combine chunk summaries into a single coherent summary."""
    combined = "\n\n".join(
        [f"Chunk {i + 1}: {s}" for i, s in enumerate(summaries)]
    )

    result: ExtractionResult = _instructor_client.messages.create(
        model=settings.EXTRACTION_MODEL,
        max_tokens=500,
        temperature=0,
        system=(
            "Combine these chunk summaries into a single coherent 2-4 sentence "
            "summary. Preserve all key facts and decisions."
        ),
        messages=[{"role": "user", "content": combined}],
        response_model=ExtractionResult,
        max_retries=1,
    )

    return result.summary


def _estimate_cost(num_chunks: int, passes: int) -> float:
    """Estimate the API cost for this extraction in USD.

    Based on approximate token counts and Claude pricing:
    - Sonnet: ~$3/1M input, ~$15/1M output
    - Haiku: ~$0.25/1M input, ~$1.25/1M output
    """
    # Extraction: ~2K input + ~2K output per chunk (Sonnet)
    extraction_cost = num_chunks * ((2000 * 3 / 1_000_000) + (2000 * 15 / 1_000_000))

    # Grounding: ~3K input + ~1K output (Sonnet)
    grounding_cost = (3000 * 3 / 1_000_000) + (1000 * 15 / 1_000_000)

    # Multi-pass: ~3K input + ~1K output (Haiku)
    multi_pass_cost = 0.0
    if passes > 1:
        multi_pass_cost = (3000 * 0.25 / 1_000_000) + (1000 * 1.25 / 1_000_000)

    return round(extraction_cost + grounding_cost + multi_pass_cost, 4)

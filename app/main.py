"""FastAPI application — routes, startup, and middleware.

This is the entry point for the extraction service. Make.com's EA-03 calls
POST /extract instead of hitting the Claude API directly.
"""

import logging

from fastapi import Depends, FastAPI, HTTPException

from .config import settings
from .extraction.pipeline import run_extraction_pipeline
from .schemas.extraction import ExtractionRequest, ExtractionResponse
from .schemas.fact import Fact
from .extraction.grounder import ground_facts
from .utils.auth import verify_api_key

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Customer Agent Extraction Service",
    description="Schema-validated fact extraction with source grounding",
    version="1.0.0",
)


@app.post("/extract", response_model=ExtractionResponse)
async def extract(
    request: ExtractionRequest,
    _: str = Depends(verify_api_key),
) -> ExtractionResponse:
    """Extract structured facts from interaction content.

    This is the endpoint Make.com's EA-03 calls instead of Claude API directly.
    Handles chunking, extraction, grounding, deduplication, and multi-pass.
    """
    if not request.content or not request.content.strip():
        return ExtractionResponse(
            success=True,
            facts=[],
            action_items=[],
            summary="No content provided.",
            metadata={
                "content_length": 0,
                "token_count": 0,
                "chunks_processed": 0,
                "passes_completed": 0,
                "facts_extracted": 0,
                "facts_with_grounding": 0,
                "action_items_extracted": 0,
                "model_used": settings.EXTRACTION_MODEL,
                "processing_time_seconds": 0.0,
                "cost_estimate_usd": 0.0,
            },
        )

    try:
        result = await run_extraction_pipeline(request)
        return result
    except Exception as e:
        logger.error(f"Extraction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ground", response_model=list[Fact])
async def ground(
    facts: list[dict],
    source_text: str,
    _: str = Depends(verify_api_key),
) -> list[Fact]:
    """Add source grounding to pre-extracted facts.

    Useful for re-grounding existing facts against updated source text.
    """
    parsed_facts = [Fact(**f) for f in facts]
    grounded = await ground_facts(parsed_facts, source_text, "transcript")
    return grounded


@app.get("/health")
async def health() -> dict:
    """Health check endpoint for monitoring and Docker healthchecks."""
    return {"status": "ok", "model": settings.EXTRACTION_MODEL}

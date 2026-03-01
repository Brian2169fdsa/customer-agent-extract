"""Core extraction engine using Instructor + Claude for schema-validated output.

Uses Instructor to wrap the Anthropic client, providing automatic Pydantic
schema validation with retry on validation failures. If Claude returns an
invalid category or missing field, Instructor sends the error back to Claude
and asks it to fix just that field — up to 3 retries.
"""

import logging
from typing import Optional

import instructor
from anthropic import Anthropic

from ..config import settings
from ..prompts.fact_extraction import get_system_prompt
from ..schemas.fact import ExtractionResult

logger = logging.getLogger(__name__)

# Instructor-wrapped Anthropic client for schema-validated responses
_anthropic_client = Anthropic()
client = instructor.from_anthropic(_anthropic_client)


async def extract_from_chunk(
    content: str,
    content_type: str,
    chunk_metadata: Optional[dict] = None,
) -> ExtractionResult:
    """Extract structured facts from a single chunk of content.

    Uses Instructor for schema validation with automatic retry.
    If Claude returns invalid data (wrong category, missing field, etc.),
    Instructor catches the Pydantic ValidationError, sends the error message
    back to Claude, and retries — up to 3 attempts.
    """
    system_prompt = get_system_prompt(content_type)

    # Add chunk context if processing a multi-chunk document
    user_content = f"Process this {content_type}:\n\n{content}"
    if chunk_metadata:
        user_content = (
            f"[Processing chunk {chunk_metadata['index'] + 1} of "
            f"{chunk_metadata['total']}]\n\n{user_content}"
        )

    logger.info(
        "Extracting from chunk",
        extra={
            "content_type": content_type,
            "content_length": len(content),
            "chunk": chunk_metadata,
        },
    )

    result = client.messages.create(
        model=settings.EXTRACTION_MODEL,
        max_tokens=settings.MAX_EXTRACTION_TOKENS,
        temperature=settings.EXTRACTION_TEMPERATURE,
        system=system_prompt,
        messages=[{"role": "user", "content": user_content}],
        response_model=ExtractionResult,
        max_retries=3,
    )

    logger.info(
        "Extraction complete",
        extra={
            "facts_count": len(result.facts),
            "action_items_count": len(result.action_items),
        },
    )

    return result

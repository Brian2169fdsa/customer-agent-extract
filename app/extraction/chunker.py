"""Smart document chunking with overlap for long content.

Splits long documents into overlapping chunks that fit within the model's optimal
extraction window. Preserves speaker boundaries in transcripts and paragraph
boundaries in emails.
"""

import re
import logging
from dataclasses import dataclass

from ..config import settings
from ..utils.tokenizer import estimate_tokens

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A segment of the original document with positional metadata."""

    text: str
    index: int
    total_chunks: int
    start_offset: int  # character offset in original doc
    end_offset: int  # character offset in original doc


def chunk_content(
    content: str,
    content_type: str,
    max_tokens: int | None = None,
    overlap_tokens: int | None = None,
) -> list[Chunk]:
    """Split content into overlapping chunks if it exceeds the token threshold.

    For transcripts: splits on speaker turn boundaries.
    For emails: splits on email boundaries (From: / --- dividers).
    For Pipedrive: always returns a single chunk (short content).

    Each chunk includes overlap from the previous chunk to catch facts
    that span chunk boundaries.
    """
    max_tokens = max_tokens or settings.CHUNK_MAX_TOKENS
    overlap_tokens = overlap_tokens or settings.CHUNK_OVERLAP_TOKENS

    total_tokens = estimate_tokens(content)

    # No chunking needed for short content
    if total_tokens <= max_tokens:
        return [Chunk(text=content, index=0, total_chunks=1, start_offset=0, end_offset=len(content))]

    # Pipedrive content should never need chunking
    if content_type in ("pipedrive_activity", "pipedrive_deal"):
        logger.warning("Pipedrive content exceeds chunk threshold — processing as single chunk")
        return [Chunk(text=content, index=0, total_chunks=1, start_offset=0, end_offset=len(content))]

    # Split into segments based on content type
    if content_type == "email":
        segments = _split_email_segments(content)
    else:
        segments = _split_transcript_segments(content)

    # Build chunks from segments with overlap
    return _build_chunks_from_segments(segments, content, max_tokens, overlap_tokens)


def _split_transcript_segments(content: str) -> list[tuple[int, int]]:
    """Split transcript on speaker turn boundaries.

    Returns list of (start_offset, end_offset) tuples for each segment.
    Speaker turns are identified by lines matching patterns like:
    - "HH:MM Speaker:" or "HH:MM:SS Speaker:"
    - "Speaker:" at the start of a line
    """
    pattern = re.compile(
        r"^(?:\d{1,2}:\d{2}(?::\d{2})?\s+)?[A-Z][a-zA-Z\s.'-]+:",
        re.MULTILINE,
    )
    matches = list(pattern.finditer(content))

    if not matches:
        # No speaker turns found — fall back to paragraph splitting
        return _split_paragraph_segments(content)

    segments = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        segments.append((start, end))

    return segments


def _split_email_segments(content: str) -> list[tuple[int, int]]:
    """Split email thread on email boundaries (From: lines or --- dividers).

    Returns list of (start_offset, end_offset) tuples for each email.
    """
    pattern = re.compile(r"^(?:From:|---+\s*$)", re.MULTILINE)
    matches = list(pattern.finditer(content))

    if not matches:
        return _split_paragraph_segments(content)

    segments = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        segments.append((start, end))

    # Include any content before the first boundary
    if matches[0].start() > 0:
        segments.insert(0, (0, matches[0].start()))

    return segments


def _split_paragraph_segments(content: str) -> list[tuple[int, int]]:
    """Fallback: split on double newlines (paragraph boundaries)."""
    pattern = re.compile(r"\n\s*\n")
    parts = pattern.split(content)

    segments = []
    offset = 0
    for part in parts:
        start = content.index(part, offset)
        end = start + len(part)
        segments.append((start, end))
        offset = end

    return segments


def _build_chunks_from_segments(
    segments: list[tuple[int, int]],
    content: str,
    max_tokens: int,
    overlap_tokens: int,
) -> list[Chunk]:
    """Assemble segments into chunks respecting token limits and overlap."""
    chunks: list[Chunk] = []
    current_segments: list[tuple[int, int]] = []
    current_tokens = 0

    for seg_start, seg_end in segments:
        seg_text = content[seg_start:seg_end]
        seg_tokens = estimate_tokens(seg_text)

        # If adding this segment would exceed the limit, finalize the current chunk
        if current_segments and current_tokens + seg_tokens > max_tokens:
            chunk_start = current_segments[0][0]
            chunk_end = current_segments[-1][1]
            chunks.append(Chunk(
                text=content[chunk_start:chunk_end],
                index=len(chunks),
                total_chunks=0,  # updated after all chunks built
                start_offset=chunk_start,
                end_offset=chunk_end,
            ))

            # Start new chunk with overlap from the end of the previous chunk
            overlap_segments = _get_overlap_segments(
                current_segments, content, overlap_tokens
            )
            current_segments = overlap_segments
            current_tokens = sum(
                estimate_tokens(content[s:e]) for s, e in current_segments
            )

        current_segments.append((seg_start, seg_end))
        current_tokens += seg_tokens

    # Finalize the last chunk
    if current_segments:
        chunk_start = current_segments[0][0]
        chunk_end = current_segments[-1][1]
        chunks.append(Chunk(
            text=content[chunk_start:chunk_end],
            index=len(chunks),
            total_chunks=0,
            start_offset=chunk_start,
            end_offset=chunk_end,
        ))

    # Update total_chunks count
    for chunk in chunks:
        chunk.total_chunks = len(chunks)

    logger.info(f"Split content into {len(chunks)} chunks")
    return chunks


def _get_overlap_segments(
    segments: list[tuple[int, int]],
    content: str,
    overlap_tokens: int,
) -> list[tuple[int, int]]:
    """Get the trailing segments that fit within the overlap token budget."""
    overlap_segs: list[tuple[int, int]] = []
    tokens_accumulated = 0

    for seg_start, seg_end in reversed(segments):
        seg_tokens = estimate_tokens(content[seg_start:seg_end])
        if tokens_accumulated + seg_tokens > overlap_tokens:
            break
        overlap_segs.insert(0, (seg_start, seg_end))
        tokens_accumulated += seg_tokens

    return overlap_segs

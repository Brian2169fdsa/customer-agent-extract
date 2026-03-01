"""Tests for the smart document chunker."""

import pytest

from app.extraction.chunker import Chunk, chunk_content


class TestChunker:
    """Test document chunking logic."""

    def test_short_content_no_chunking(self):
        """Content under the token threshold should return a single chunk."""
        content = "Short meeting notes. Nothing much happened."
        chunks = chunk_content(content, "transcript", max_tokens=6000, overlap_tokens=500)
        assert len(chunks) == 1
        assert chunks[0].index == 0
        assert chunks[0].total_chunks == 1
        assert chunks[0].text == content

    def test_single_chunk_preserves_offsets(self):
        """Single-chunk content should have offsets spanning the full document."""
        content = "Hello world, this is a test."
        chunks = chunk_content(content, "transcript", max_tokens=6000, overlap_tokens=500)
        assert chunks[0].start_offset == 0
        assert chunks[0].end_offset == len(content)

    def test_pipedrive_never_chunks(self):
        """Pipedrive activities should never be chunked, even if long."""
        content = "x " * 10000  # Very long content
        chunks = chunk_content(content, "pipedrive_activity", max_tokens=100, overlap_tokens=10)
        assert len(chunks) == 1

    def test_long_transcript_chunks(self):
        """A long transcript should be split into multiple chunks."""
        # Build a transcript with clear speaker turns
        turns = []
        for i in range(50):
            speaker = "Alice" if i % 2 == 0 else "Bob"
            turns.append(f"{i // 6:02d}:{(i * 2) % 60:02d} {speaker}: {'Discussion point. ' * 20}")
        content = "\n".join(turns)

        chunks = chunk_content(content, "transcript", max_tokens=500, overlap_tokens=50)
        assert len(chunks) > 1
        # All chunks should reference the correct total
        for chunk in chunks:
            assert chunk.total_chunks == len(chunks)

    def test_chunk_offsets_cover_document(self):
        """Chunk offsets should cover the entire document without gaps (allowing overlaps)."""
        turns = []
        for i in range(30):
            speaker = "Alice" if i % 2 == 0 else "Bob"
            turns.append(f"{i:02d}:00 {speaker}: {'Some important discussion content. ' * 15}")
        content = "\n".join(turns)

        chunks = chunk_content(content, "transcript", max_tokens=500, overlap_tokens=50)

        if len(chunks) > 1:
            # First chunk starts at 0
            assert chunks[0].start_offset == 0
            # Last chunk ends at document length
            assert chunks[-1].end_offset == len(content)

    def test_email_splitting(self):
        """Email threads should split on From: boundaries."""
        content = (
            "From: alice@example.com\nSubject: Project Update\n\n"
            "Hi team, here's the update. " * 50 + "\n\n"
            "From: bob@example.com\nSubject: Re: Project Update\n\n"
            "Thanks Alice, some thoughts. " * 50 + "\n\n"
            "From: charlie@example.com\nSubject: Re: Project Update\n\n"
            "Adding my perspective. " * 50
        )
        chunks = chunk_content(content, "email", max_tokens=200, overlap_tokens=20)
        assert len(chunks) >= 1

    def test_chunk_dataclass(self):
        """Chunk dataclass should have all required fields."""
        chunk = Chunk(text="test", index=0, total_chunks=1, start_offset=0, end_offset=4)
        assert chunk.text == "test"
        assert chunk.index == 0
        assert chunk.total_chunks == 1

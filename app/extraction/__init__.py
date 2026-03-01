"""Extraction engine: chunking, extraction, grounding, deduplication, multi-pass.

Import the pipeline entry point at call time to avoid eager loading of
instructor/anthropic at module import (allows unit tests for chunker,
deduplicator, etc. to run without API credentials).
"""


def run_extraction_pipeline(*args, **kwargs):
    """Lazy wrapper — imports the real pipeline on first call."""
    from .pipeline import run_extraction_pipeline as _impl

    return _impl(*args, **kwargs)

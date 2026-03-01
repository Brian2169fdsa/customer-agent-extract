"""Token counting utilities for chunking decisions.

Uses Anthropic's token counter when available, falls back to a ~4 chars/token
approximation which is close enough for chunking decisions.
"""

import logging

logger = logging.getLogger(__name__)

# Approximate characters per token for Claude models
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Estimate the token count for a piece of text.

    Tries the Anthropic SDK's token counter first. If unavailable,
    falls back to a simple character-based approximation (4 chars/token).
    """
    try:
        from anthropic import Anthropic

        client = Anthropic()
        count = client.count_tokens(text)
        return count
    except Exception:
        # Fallback: ~4 characters per token is a reasonable approximation
        return len(text) // CHARS_PER_TOKEN

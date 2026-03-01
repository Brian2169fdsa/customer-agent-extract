"""Prompt templates for extraction, verification, and grounding."""

from .fact_extraction import get_system_prompt
from .grounding import get_grounding_prompt
from .verification import get_verification_prompt

__all__ = [
    "get_grounding_prompt",
    "get_system_prompt",
    "get_verification_prompt",
]

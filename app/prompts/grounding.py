"""Prompt for source grounding — mapping extracted facts back to source text."""

GROUNDING_SYSTEM_PROMPT = """\
You are a source grounding engine. Given a list of claims and a source document,
find the EXACT text span in the source that supports each claim.

Return a JSON array where each element corresponds to a claim (by index).

For each claim:
- Find the most relevant passage in the source document that directly supports it
- Return the VERBATIM text from the source (do not paraphrase)
- Provide the character offsets (start_offset, end_offset) in the source document
- start_offset is the 0-based character index where the supporting text begins
- end_offset is the 0-based character index where the supporting text ends

If a claim is a synthesis of multiple passages, return the single most relevant passage.
If a claim cannot be grounded to a specific span (e.g., it's inferred from overall context),
return null values for that entry.
"""


def get_grounding_prompt() -> str:
    """Return the system prompt for the grounding pass."""
    return GROUNDING_SYSTEM_PROMPT

# CLAUDE.md — Customer Agent Extraction Service

## Project Overview

You are building the **Customer Agent Extraction Service** — a Python microservice that supercharges the Claude EA's fact extraction pipeline (EA-03) with schema-validated structured outputs, character-level source grounding, intelligent chunking for long documents, and multi-pass extraction for high-value interactions.

This replaces the raw Claude API call inside Make.com's EA-03 scenario with a single HTTP endpoint that handles all extraction complexity. Make calls `POST /extract` with raw content, gets back guaranteed-valid structured facts with exact source locations.

### Why This Exists

The raw Claude API approach in EA-03 has three failure modes at scale:
1. **Schema drift** — Claude occasionally returns malformed JSON, invalid categories, or missing required fields. At 20+ interactions/day, this means broken data in Notion.
2. **Long document blindness** — A 45-minute transcript exceeds optimal context for single-pass extraction. Facts in the middle get missed ("needle in a haystack").
3. **No source grounding** — Storing "approximately 22:00 mark" is useful but not traceable. Character-level offsets enable "show me exactly where Dave said that" with highlighting.

This service solves all three by combining:
- **Instructor** — Pydantic schema validation with automatic retry on validation failure
- **LangExtract** — Character-level source grounding with offset mapping
- **Custom chunking** — Smart splitting of long documents with overlap for continuity
- **Multi-pass extraction** — High-value meetings get a second pass to catch missed facts

### Architecture

```
Make.com (EA-03)                    This Service (Hetzner/Coolify)
┌─────────────┐    POST /extract    ┌─────────────────────────────────┐
│ EA-03 calls ├───────────────────→│ FastAPI Endpoint                │
│ HTTP module │                     │                                 │
│             │    JSON response    │ ┌─────────────────────────────┐ │
│             │←───────────────────│ │ 1. Chunker                  │ │
└─────────────┘                     │ │    Split long docs into     │ │
                                    │ │    overlapping segments      │ │
                                    │ ├─────────────────────────────┤ │
                                    │ │ 2. Instructor + Claude API  │ │
                                    │ │    Schema-validated extract  │ │
                                    │ │    Auto-retry on bad output  │ │
                                    │ ├─────────────────────────────┤ │
                                    │ │ 3. LangExtract              │ │
                                    │ │    Source grounding pass     │ │
                                    │ │    Character offset mapping  │ │
                                    │ ├─────────────────────────────┤ │
                                    │ │ 4. Deduplicator             │ │
                                    │ │    Merge facts from chunks   │ │
                                    │ │    Remove duplicates         │ │
                                    │ ├─────────────────────────────┤ │
                                    │ │ 5. Multi-pass (if enabled)  │ │
                                    │ │    Second extraction pass    │ │
                                    │ │    Diff with first pass      │ │
                                    │ │    Merge new facts           │ │
                                    │ └─────────────────────────────┘ │
                                    └─────────────────────────────────┘
```

---

## Environment Setup

### Prerequisites
- Python 3.11+
- pip

### Required Packages
```
fastapi>=0.115.0
uvicorn>=0.32.0
instructor>=1.7.0
anthropic>=0.40.0
langextract>=0.3.0
pydantic>=2.9.0
python-dotenv>=1.0.0
httpx>=0.27.0
```

### Environment Variables (.env)
```env
# ═══════ ANTHROPIC ═══════
ANTHROPIC_API_KEY=sk-ant-REPLACE_ME

# ═══════ EXTRACTION CONFIG ═══════
# Primary model for extraction (Sonnet for cost efficiency)
EXTRACTION_MODEL=claude-sonnet-4-5-20250929
# Secondary model for multi-pass verification (can be same or Haiku for cost)
VERIFICATION_MODEL=claude-haiku-4-5-20251001
# Max tokens per extraction call
MAX_EXTRACTION_TOKENS=8000
# Temperature for extraction (0 = deterministic)
EXTRACTION_TEMPERATURE=0

# ═══════ CHUNKING CONFIG ═══════
# Max tokens per chunk (leave headroom for system prompt + response)
CHUNK_MAX_TOKENS=6000
# Overlap between chunks in tokens (ensures facts spanning chunk boundaries aren't missed)
CHUNK_OVERLAP_TOKENS=500
# Min chunk size — don't bother chunking if content is smaller than this
CHUNK_MIN_TOKENS=500

# ═══════ MULTI-PASS CONFIG ═══════
# Enable multi-pass extraction by default (can be overridden per-request)
MULTI_PASS_DEFAULT=false
# Max number of extraction passes
MAX_PASSES=2

# ═══════ SERVICE CONFIG ═══════
HOST=0.0.0.0
PORT=8100
# API key for Make.com to authenticate with this service
SERVICE_API_KEY=REPLACE_WITH_RANDOM_SECRET
# Max concurrent extraction requests
MAX_CONCURRENT=5
```

---

## Project Structure

```
customer-agent-extract/
├── CLAUDE.md                          ← YOU ARE HERE
├── .env
├── .env.example
├── .gitignore
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── README.md
│
├── app/
│   ├── __init__.py
│   ├── main.py                        ← FastAPI app, routes, startup
│   ├── config.py                      ← Settings from .env via pydantic-settings
│   │
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── fact.py                    ← Pydantic models for Facts, ActionItems
│   │   ├── extraction.py             ← ExtractionRequest/Response models
│   │   └── grounding.py              ← Source grounding offset models
│   │
│   ├── extraction/
│   │   ├── __init__.py
│   │   ├── extractor.py              ← Core extraction engine (Instructor + Claude)
│   │   ├── chunker.py                ← Smart document chunking with overlap
│   │   ├── grounder.py               ← LangExtract source grounding pass
│   │   ├── deduplicator.py           ← Cross-chunk fact deduplication
│   │   ├── multi_pass.py             ← Second-pass extraction + diff/merge
│   │   └── pipeline.py               ← Orchestrates the full extraction pipeline
│   │
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── fact_extraction.py         ← System prompt for fact extraction
│   │   ├── verification.py            ← System prompt for multi-pass verification
│   │   └── grounding.py              ← Prompt for source grounding pass
│   │
│   └── utils/
│       ├── __init__.py
│       ├── tokenizer.py               ← Token counting for chunking decisions
│       └── auth.py                    ← API key auth middleware
│
├── tests/
│   ├── __init__.py
│   ├── test_extraction.py
│   ├── test_chunker.py
│   ├── test_grounding.py
│   ├── test_deduplication.py
│   ├── test_multi_pass.py
│   └── test_api.py
│
└── test_data/
    ├── short-meeting.txt              ← 10 min transcript (under chunk threshold)
    ├── long-meeting.txt               ← 45 min transcript (requires chunking)
    ├── email-thread.txt               ← 3-message email thread
    ├── pipedrive-activity.json        ← Pipedrive activity note
    └── expected/                      ← Expected extraction outputs for validation
        ├── short-meeting.json
        ├── long-meeting.json
        ├── email-thread.json
        └── pipedrive-activity.json
```

---

## Schema Definitions

### app/schemas/fact.py

These Pydantic models are the heart of the system. Instructor uses them to guarantee every extraction matches the schema. If Claude returns a fact with `category: "Commitment/Decision"`, Instructor catches the validation error and retries automatically.

```python
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal
from datetime import date
from enum import Enum

class FactCategory(str, Enum):
    COMMITMENT = "Commitment"
    DECISION = "Decision"
    CONCERN = "Concern"
    FEEDBACK = "Feedback"
    PRIORITY = "Priority"
    PREFERENCE = "Preference"
    CONTEXT = "Context"
    REQUEST = "Request"
    MILESTONE = "Milestone"

class Confidence(str, Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"

class Sentiment(str, Enum):
    POSITIVE = "Positive"
    NEUTRAL = "Neutral"
    NEGATIVE = "Negative"
    URGENT = "Urgent"

class SourceGrounding(BaseModel):
    """Character-level source grounding for a fact."""
    start_offset: int = Field(description="Start character offset in source document")
    end_offset: int = Field(description="End character offset in source document")
    source_text: str = Field(description="The exact text span from the source that supports this fact")
    
class Fact(BaseModel):
    """A single structured fact extracted from an interaction."""
    claim: str = Field(description="Concise factual statement that stands alone")
    category: FactCategory
    relationship: str = Field(description="Person or company name this fact belongs to")
    speaker: str = Field(description="Who said or wrote this")
    timestamp: str = Field(description="HH:MM for transcripts, 'paragraph N' for emails, 'activity' for Pipedrive")
    confidence: Confidence
    sentiment: Sentiment
    due_date: Optional[date] = Field(default=None, description="For commitments/requests with deadlines")
    tags: list[str] = Field(default_factory=list, description="Topic tags: pricing, product, timeline, etc.")
    
    # Source grounding — populated by the grounding pass
    grounding: Optional[SourceGrounding] = Field(default=None, description="Character-level source location")
    
    @field_validator('claim')
    @classmethod
    def claim_not_empty(cls, v):
        if len(v.strip()) < 5:
            raise ValueError('Claim must be at least 5 characters — be specific')
        return v.strip()
    
    @field_validator('relationship')
    @classmethod
    def relationship_not_empty(cls, v):
        if len(v.strip()) < 1:
            raise ValueError('Relationship must identify a person or company')
        return v.strip()

class ActionItem(BaseModel):
    """An action item extracted from the interaction."""
    title: str = Field(description="Specific actionable task")
    owner: str = Field(default="Brian", description="Person responsible")
    priority: Literal["P1", "P2", "P3"] = Field(default="P2")
    due_date: Optional[date] = None
    relationship: str = Field(description="Related person or company")

class ExtractionResult(BaseModel):
    """Complete extraction output from a single pass."""
    facts: list[Fact] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    summary: str = Field(description="One paragraph summary, 2-4 sentences")
```

### app/schemas/extraction.py

```python
from pydantic import BaseModel, Field
from typing import Optional, Literal
from .fact import ExtractionResult, Fact

class ExtractionRequest(BaseModel):
    """Request body for the /extract endpoint."""
    content: str = Field(description="Raw interaction content to process")
    content_type: Literal["transcript", "email", "chat", "pipedrive_activity", "pipedrive_deal"] = Field(default="transcript")
    interaction_id: Optional[str] = Field(default=None, description="Notion page ID for reference")
    multi_pass: Optional[bool] = Field(default=None, description="Override multi-pass setting for this request")
    priority: Literal["normal", "high"] = Field(default="normal", description="High priority = multi-pass + deeper extraction")

class ExtractionResponse(BaseModel):
    """Response body from the /extract endpoint."""
    success: bool
    facts: list[Fact]
    action_items: list[dict]  # Serialized ActionItems
    summary: str
    metadata: ExtractionMetadata

class ExtractionMetadata(BaseModel):
    """Metadata about the extraction process."""
    content_length: int = Field(description="Length of input content in characters")
    token_count: int = Field(description="Approximate token count of input")
    chunks_processed: int = Field(description="Number of chunks (1 if no chunking needed)")
    passes_completed: int = Field(description="Number of extraction passes (1 or 2)")
    facts_extracted: int
    facts_with_grounding: int = Field(description="Facts that have character-level source grounding")
    action_items_extracted: int
    model_used: str
    processing_time_seconds: float
    cost_estimate_usd: float = Field(description="Estimated API cost for this extraction")
```

---

## Core Components

### app/extraction/chunker.py — Smart Document Chunking

**Purpose:** Splits long documents into overlapping chunks that fit within the model's optimal extraction window. Preserves speaker boundaries in transcripts and paragraph boundaries in emails.

**Logic:**
1. Count tokens using `anthropic`'s token counter (or approximate at 4 chars/token)
2. If total tokens < `CHUNK_MIN_TOKENS`, return content as single chunk (no splitting needed)
3. If total tokens < `CHUNK_MAX_TOKENS`, return content as single chunk
4. Otherwise, split into chunks:
   - For **transcripts**: split on speaker turn boundaries (lines starting with timestamps or speaker labels). Never split mid-turn. Each chunk starts with the last `CHUNK_OVERLAP_TOKENS` worth of turns from the previous chunk.
   - For **emails**: split on email boundaries (lines starting with "From:" or "---" dividers). Never split mid-email in a thread.
   - For **Pipedrive activities/deals**: these are always short enough for single chunk. No splitting.
5. Each chunk carries metadata: `{chunk_index, total_chunks, start_offset, end_offset}` — the character offsets in the original document that this chunk covers.

```python
from dataclasses import dataclass

@dataclass
class Chunk:
    text: str
    index: int
    total_chunks: int
    start_offset: int  # character offset in original doc
    end_offset: int    # character offset in original doc
```

**Key rule:** Overlap is critical. Facts that span a chunk boundary (e.g., someone starts a commitment at the end of chunk 1 and finishes it at the start of chunk 2) must be catchable by the overlap. 500 tokens of overlap (~2000 chars) handles most cases.

---

### app/extraction/extractor.py — Instructor-Powered Extraction

**Purpose:** The core extraction engine. Uses Instructor with Claude to extract schema-validated facts from a single chunk of content.

**Implementation:**

```python
import instructor
from anthropic import Anthropic
from ..schemas.fact import ExtractionResult
from ..prompts.fact_extraction import get_system_prompt

client = instructor.from_anthropic(Anthropic())

async def extract_from_chunk(
    content: str,
    content_type: str,
    chunk_metadata: dict = None,
) -> ExtractionResult:
    """
    Extract structured facts from a single chunk of content.
    Uses Instructor for schema validation with automatic retry.
    """
    system_prompt = get_system_prompt(content_type)
    
    result = client.messages.create(
        model=settings.EXTRACTION_MODEL,
        max_tokens=settings.MAX_EXTRACTION_TOKENS,
        temperature=settings.EXTRACTION_TEMPERATURE,
        system=system_prompt,
        messages=[{
            "role": "user",
            "content": f"Process this {content_type}:\n\n{content}"
        }],
        response_model=ExtractionResult,
        max_retries=3,  # Instructor auto-retries on validation failure
    )
    
    return result
```

**What Instructor does here that raw API doesn't:**
- Defines the response schema from the `ExtractionResult` Pydantic model
- If Claude returns `category: "Commitment/Decision"` (invalid), Instructor catches the Pydantic ValidationError, sends the error message back to Claude, and asks it to fix just that field
- Up to 3 retry attempts before giving up
- The response is a typed Python object, not a JSON string to parse

---

### app/extraction/grounder.py — LangExtract Source Grounding

**Purpose:** After facts are extracted, this pass maps each fact's `claim` back to the exact character offsets in the original source document. This enables "show me where Dave said that" queries with exact text highlighting.

**How it works:**

LangExtract's core capability is mapping extracted entities back to source text with character offsets. We use it as a second pass:

1. Take the list of extracted facts (claims only)
2. Feed them to LangExtract along with the original document
3. LangExtract returns character offset pairs `(start, end)` for each fact
4. Attach the offsets to each fact's `grounding` field

```python
import langextract as lx

def ground_facts(
    facts: list[Fact],
    source_text: str,
    content_type: str,
) -> list[Fact]:
    """
    Map each extracted fact back to its exact location in the source text.
    Uses LangExtract for character-level offset grounding.
    """
    # Build LangExtract extraction task
    # We're asking it to find each fact claim in the source text
    prompt = (
        "For each claim provided, find the exact passage in the source text "
        "that supports this claim. Return the verbatim text span. "
        "If the claim is a synthesis of multiple passages, return the most "
        "relevant single passage."
    )
    
    # Create examples to guide LangExtract (few-shot)
    examples = build_grounding_examples(content_type)
    
    # Run LangExtract with Claude as the model
    result = lx.extract(
        text_or_documents=source_text,
        prompt_description=prompt,
        examples=examples,
        model_id="anthropic/claude-sonnet-4-5-20250929",
        # Use the Anthropic provider plugin for LangExtract
    )
    
    # Map results back to facts
    for fact in facts:
        grounding = find_best_match(fact.claim, result.extractions, source_text)
        if grounding:
            fact.grounding = SourceGrounding(
                start_offset=grounding.char_start,
                end_offset=grounding.char_end,
                source_text=source_text[grounding.char_start:grounding.char_end]
            )
    
    return facts
```

**Important implementation note:** LangExtract natively supports Anthropic models through community provider plugins. If the Anthropic provider isn't available yet, implement the grounding manually:

**Fallback grounding approach (no LangExtract dependency):**

```python
async def ground_facts_fallback(
    facts: list[Fact],
    source_text: str,
) -> list[Fact]:
    """
    Fallback: Use Claude directly to find source locations.
    For each fact, ask Claude to find the exact supporting text span.
    """
    # Batch all facts into one Claude call for efficiency
    facts_list = "\n".join([f"{i+1}. {f.claim}" for i, f in enumerate(facts)])
    
    response = client.messages.create(
        model=settings.EXTRACTION_MODEL,
        max_tokens=settings.MAX_EXTRACTION_TOKENS,
        temperature=0,
        system=(
            "You are a source grounding engine. Given a list of claims and a source document, "
            "find the EXACT text span in the source that supports each claim. "
            "Return JSON array: [{\"fact_index\": 1, \"start_offset\": N, \"end_offset\": N, \"source_text\": \"exact quote\"}]. "
            "start_offset and end_offset are character positions in the source document. "
            "If a claim cannot be grounded to a specific span, return null for that fact."
        ),
        messages=[{
            "role": "user",
            "content": f"CLAIMS:\n{facts_list}\n\nSOURCE DOCUMENT:\n{source_text}"
        }],
        response_model=list[GroundingResult],  # Instructor validates this too
        max_retries=2,
    )
    
    # Attach grounding to facts
    for gr in response:
        if gr and gr.fact_index <= len(facts):
            facts[gr.fact_index - 1].grounding = SourceGrounding(
                start_offset=gr.start_offset,
                end_offset=gr.end_offset,
                source_text=gr.source_text,
            )
    
    return facts
```

**Decision:** Try LangExtract first. If the Anthropic provider plugin doesn't work cleanly, use the fallback. The fallback is one extra Claude API call per extraction, which adds ~$0.01 per interaction and ~2 seconds. Worth it for source grounding.

---

### app/extraction/deduplicator.py — Cross-Chunk Deduplication

**Purpose:** When a document is chunked with overlap, the same fact may be extracted from two adjacent chunks. This component merges duplicates.

**Logic:**
1. Compare every fact from chunk N with every fact from chunk N+1 (only adjacent chunks, since overlap only affects neighbors)
2. Two facts are duplicates if:
   - Same `relationship` AND same `category` AND semantic similarity of `claim` > 0.85
   - OR exact same `claim` text (after normalization)
3. When merging duplicates, keep the one with:
   - Higher `confidence`
   - Better `grounding` (if one has grounding and the other doesn't)
   - Earlier `timestamp`

**Semantic similarity approach:**
- Simple: Levenshtein distance / max(len) > 0.85 after lowercasing and stripping punctuation
- Better: Use Claude Haiku for a quick "are these the same fact?" classification (batch all candidates in one call)

```python
async def deduplicate_facts(
    chunks_facts: list[list[Fact]],
) -> list[Fact]:
    """
    Merge facts from multiple chunks, removing duplicates from overlap regions.
    """
    if len(chunks_facts) == 1:
        return chunks_facts[0]
    
    all_facts = []
    seen_claims = set()
    
    for chunk_idx, chunk_facts in enumerate(chunks_facts):
        for fact in chunk_facts:
            normalized = normalize_claim(fact.claim)
            if normalized not in seen_claims:
                # Check fuzzy match against existing facts
                is_dup, merge_target = find_duplicate(fact, all_facts)
                if is_dup and merge_target:
                    # Merge: keep higher confidence, better grounding
                    merge_facts(merge_target, fact)
                else:
                    all_facts.append(fact)
                    seen_claims.add(normalized)
    
    return all_facts
```

---

### app/extraction/multi_pass.py — Second Pass Extraction

**Purpose:** For high-value meetings (flagged by the user or by priority="high" in the request), run a second extraction pass with a different prompt that specifically looks for facts the first pass might have missed.

**Logic:**
1. Take the first pass results (list of facts)
2. Send the original content PLUS the first pass facts to Claude with a verification prompt:
   "Here are facts already extracted. Review the source and identify any ADDITIONAL facts that were missed, especially: commitments, concerns, and budget/timeline context."
3. Validate new facts with Instructor
4. Deduplicate against first pass
5. Merge into final result

```python
async def second_pass_extraction(
    content: str,
    content_type: str,
    first_pass_facts: list[Fact],
) -> list[Fact]:
    """
    Run a verification pass looking for missed facts.
    Uses a different prompt focused on finding gaps.
    """
    existing_claims = "\n".join([f"- [{f.category.value}] {f.claim}" for f in first_pass_facts])
    
    result = client.messages.create(
        model=settings.VERIFICATION_MODEL,  # Can use Haiku for cost savings
        max_tokens=settings.MAX_EXTRACTION_TOKENS,
        temperature=0,
        system=get_verification_prompt(content_type),
        messages=[{
            "role": "user",
            "content": (
                f"ALREADY EXTRACTED FACTS:\n{existing_claims}\n\n"
                f"SOURCE {content_type.upper()}:\n{content}\n\n"
                "Find any ADDITIONAL facts that were missed in the first pass. "
                "Focus on: commitments, concerns, budget numbers, timeline details, "
                "and preferences that were overlooked."
            )
        }],
        response_model=ExtractionResult,
        max_retries=2,
    )
    
    # Only return genuinely new facts
    new_facts = [f for f in result.facts if not is_duplicate(f, first_pass_facts)]
    return new_facts
```

---

### app/extraction/pipeline.py — Full Pipeline Orchestrator

**Purpose:** Orchestrates the complete extraction flow: chunk → extract → ground → deduplicate → multi-pass → response.

```python
async def run_extraction_pipeline(
    request: ExtractionRequest,
) -> ExtractionResponse:
    """
    Full extraction pipeline:
    1. Chunk the content if needed
    2. Extract facts from each chunk (Instructor + Claude)
    3. Ground facts to source locations (LangExtract or fallback)
    4. Deduplicate across chunks
    5. Multi-pass if enabled
    6. Return structured response
    """
    start_time = time.time()
    
    # Step 1: Chunk
    chunks = chunk_content(
        content=request.content,
        content_type=request.content_type,
        max_tokens=settings.CHUNK_MAX_TOKENS,
        overlap_tokens=settings.CHUNK_OVERLAP_TOKENS,
    )
    
    # Step 2: Extract from each chunk (parallel if multiple)
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
    
    # Step 4: Source grounding
    try:
        facts = await ground_facts(facts, request.content, request.content_type)
    except Exception as e:
        # Grounding is best-effort — don't fail the whole extraction
        logger.warning(f"Source grounding failed: {e}")
    
    # Step 5: Multi-pass if enabled
    use_multi_pass = request.multi_pass if request.multi_pass is not None else settings.MULTI_PASS_DEFAULT
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
                new_facts = await ground_facts(new_facts, request.content, request.content_type)
            except Exception:
                pass
            facts.extend(new_facts)
        passes_completed = 2
    
    # Step 6: Merge summaries
    if len(summaries) == 1:
        final_summary = summaries[0]
    else:
        # Combine chunk summaries into one
        final_summary = await merge_summaries(summaries, request.content_type)
    
    # Step 7: Deduplicate action items
    action_items = deduplicate_action_items(chunks_action_items)
    
    # Build response
    processing_time = time.time() - start_time
    grounded_count = sum(1 for f in facts if f.grounding is not None)
    
    return ExtractionResponse(
        success=True,
        facts=[f.model_dump() for f in facts],
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
            cost_estimate_usd=estimate_cost(len(chunks), passes_completed),
        ),
    )
```

---

## API Endpoints

### app/main.py

```python
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

app = FastAPI(
    title="Customer Agent Extraction Service",
    description="Schema-validated fact extraction with source grounding",
    version="1.0.0",
)

security = HTTPBearer()

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != settings.SERVICE_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

@app.post("/extract", response_model=ExtractionResponse)
async def extract(
    request: ExtractionRequest,
    _=Depends(verify_api_key),
):
    """
    Extract structured facts from interaction content.
    
    This is the endpoint Make.com's EA-03 calls instead of Claude API directly.
    """
    try:
        result = await run_extraction_pipeline(request)
        return result
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ground", response_model=list[Fact])
async def ground(
    facts: list[dict],
    source_text: str,
    _=Depends(verify_api_key),
):
    """
    Add source grounding to pre-extracted facts.
    Useful for re-grounding existing facts against updated source text.
    """
    parsed_facts = [Fact(**f) for f in facts]
    grounded = await ground_facts(parsed_facts, source_text, "transcript")
    return grounded

@app.get("/health")
async def health():
    return {"status": "ok", "model": settings.EXTRACTION_MODEL}
```

---

## Make.com Integration

### How EA-03 Changes

Current EA-03 Module 5 (HTTP: Claude API):
```
POST https://api.anthropic.com/v1/messages
Headers: x-api-key, anthropic-version
Body: {model, system, messages, max_tokens, temperature}
```

**New EA-03 Module 5 (HTTP: Extraction Service):**
```
POST https://your-service.yourdomain.com/extract
Headers: Authorization: Bearer {{SERVICE_API_KEY}}
Content-Type: application/json
Body: {
  "content": "{{raw_content}}",
  "content_type": "{{interaction_type}}",
  "interaction_id": "{{interaction_id}}",
  "priority": "normal"
}
```

**Everything else in EA-03 stays the same.** Modules 6-14 (JSON parse, iterate facts, create in Notion, etc.) work identically — the response schema is the same, just with additional `grounding` data on each fact.

### New Notion Field

Add one property to the Facts DB:
- **Source Grounding** — `rich_text` — Stores JSON: `{"start": N, "end": N, "text": "exact quote"}`

When EA-05 (Relationship Intelligence Query) assembles a brief, it can now include the exact source quote instead of an approximate timestamp.

---

## Deployment

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8100

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8100"]
```

### docker-compose.yml

```yaml
version: "3.8"
services:
  extraction:
    build: .
    ports:
      - "8100:8100"
    env_file:
      - .env
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8100/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Coolify Deployment

Since you're already running Coolify on Hetzner:
1. Push the repo to GitHub
2. In Coolify, create a new service → Docker Compose → point to the repo
3. Add the environment variables in Coolify's UI
4. Deploy
5. Set up a domain or use the Coolify-provided URL
6. Update EA-03 in Make to point to the new URL

---

## Cost Estimates

Per interaction processed:

| Component | Tokens (approx) | Cost |
|-----------|-----------------|------|
| Extraction (Sonnet, single chunk) | ~2K in + ~2K out | ~$0.02 |
| Source grounding pass | ~3K in + ~1K out | ~$0.015 |
| Multi-pass verification (Haiku) | ~3K in + ~1K out | ~$0.003 |
| **Total per interaction (normal)** | | **~$0.035** |
| **Total per interaction (high priority)** | | **~$0.038** |

At 20 interactions/day: ~$0.70/day → ~$21/month
At 50 interactions/day: ~$1.75/day → ~$53/month

The grounding pass adds ~40% to the API cost but provides character-level source tracing — well worth it for the "where did they say that" capability.

---

## Build Sequence

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run tests with sample data
pytest tests/ -v

# 3. Start the service locally
uvicorn app.main:app --reload --port 8100

# 4. Test the /extract endpoint
curl -X POST http://localhost:8100/extract \
  -H "Authorization: Bearer YOUR_SERVICE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"content": "...", "content_type": "transcript"}'

# 5. Validate grounding accuracy
python -m tests.test_grounding

# 6. Docker build + deploy
docker compose up --build

# 7. Update Make.com EA-03 to point to deployed URL
```

---

## Quality Checklist

- [ ] All Pydantic models validate correctly — invalid categories, confidences, sentiments are caught
- [ ] Instructor retries work — test with a prompt that deliberately produces invalid output
- [ ] Chunking preserves speaker turns — no mid-sentence splits
- [ ] Chunk overlap catches boundary-spanning facts
- [ ] Deduplication doesn't remove genuinely different facts with similar wording
- [ ] Source grounding offsets are accurate — spot-check by extracting the substring from source
- [ ] Multi-pass finds at least 1-2 additional facts on a long transcript
- [ ] API key auth rejects requests without valid key
- [ ] Service handles empty content gracefully (returns empty facts list, not error)
- [ ] Service handles very long content (60+ min transcript) without timeout
- [ ] Error responses include meaningful messages, not stack traces
- [ ] Health endpoint returns 200 and model name
- [ ] Docker image builds and runs correctly
- [ ] Cost estimates match actual Anthropic API billing

---

## Coding Standards

- Use async/await throughout — FastAPI is async-native
- Type everything — Pydantic models for all inputs and outputs
- Use `structlog` or `loguru` for structured logging
- Environment config via `pydantic-settings` (not raw `os.getenv`)
- All test data files are committed to the repo
- Tests use `pytest` with `pytest-asyncio` for async tests
- Every function has a docstring explaining what it does and why
- API responses always include the `metadata` block for debugging and cost tracking

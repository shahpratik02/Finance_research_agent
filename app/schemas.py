"""
Pydantic schemas for all structured inputs and outputs in the pipeline.

Import pattern:
    from app.schemas import QueryInput, PlannerOutput, SourceRecord, ...

Every LLM-facing schema (PlannerOutput, ResearchResult, ClaimReviewSet,
FinalReport) is also the JSON schema passed to the inference server for
constrained decoding — keep field names and types stable.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── Enumerations ───────────────────────────────────────────────────────────────

class OutputStyle(str, Enum):
    memo  = "memo"    # one-page: executive summary + key points
    brief = "brief"   # 2-4 sentence summary, no headings
    full  = "full"    # complete report with all sections


class ResearchAngle(str, Enum):
    company          = "company"
    macro            = "macro"
    news             = "news"
    valuation        = "valuation"
    comparison       = "comparison"
    risk             = "risk"
    business_quality = "business_quality"


class SupportType(str, Enum):
    direct        = "direct"        # evidence directly states the claim
    indirect      = "indirect"      # evidence implies the claim
    circumstantial = "circumstantial"  # weak / background support


class ClaimVerdict(str, Enum):
    verified           = "verified"            # evidence directly supports it
    partially_verified = "partially_verified"  # direction right, wording too strong
    unsupported        = "unsupported"         # evidence does not support it
    contradicted       = "contradicted"        # evidence points the other way


class MCPProvider(str, Enum):
    yahoo_finance      = "yahoo_finance"
    fred               = "fred"
    financial_datasets = "financial_datasets"
    open_web_search    = "open_web_search"
    rag_document       = "rag_document"  # user-supplied chunks (not from MCP)


# ── Input ──────────────────────────────────────────────────────────────────────

class QueryInput(BaseModel):
    """User-facing request that starts the pipeline."""
    query: str = Field(..., description="The finance research question from the user.")
    as_of: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp to anchor the research. Defaults to now.",
    )
    output_style: OutputStyle = Field(
        default=OutputStyle.memo,
        description="Controls the length and structure of the final report.",
    )
    documents: list[str] | None = Field(
        default=None,
        description=(
            "Optional raw document texts. When provided (non-empty), the pipeline runs "
            "retrieval over these docs before external research."
        ),
    )
    documents_folder: str | None = Field(
        default=None,
        description=(
            "Optional path to a directory of text files to index for RAG (merged with "
            "`documents` when both are set). Uses a local vector store when embeddings are configured."
        ),
    )


# ── Planner output ─────────────────────────────────────────────────────────────

class PlannerOutput(BaseModel):
    """Structured plan produced by the Planner agent."""
    research_angles: list[ResearchAngle] = Field(
        ...,
        min_length=1,
        max_length=4,
        description="Which research dimensions are relevant (1-4 angles).",
    )
    subquestions: list[str] = Field(
        ...,
        min_length=2,
        max_length=6,
        description="Specific questions the Researcher must answer.",
    )
    suggested_tools: list[str] = Field(
        ...,
        min_length=1,
        max_length=6,
        description="MCP tool names the Researcher should prioritise.",
    )


# ── Source record ──────────────────────────────────────────────────────────────

class SourceRecord(BaseModel):
    """
    Normalised representation of a single MCP tool result.
    Produced by source_normalizer.py; stored in the sources table.
    """
    source_id: str = Field(
        ...,
        description="Stable unique id: {run_id_prefix}-{provider_prefix}-{uuid4_short}",
    )
    run_id: str
    provider: MCPProvider
    tool: str = Field(..., description="Name of the MCP tool that produced this source.")
    title: str
    uri: str | None = None
    retrieved_at: datetime
    published_at: datetime | None = None
    entity: str | None = Field(None, description="Ticker, series ID, or other primary entity.")
    content_summary: str = Field(..., description="1-3 sentence summary of the content.")
    raw_excerpt: str = Field(..., description="Key numbers or text extracted verbatim.")
    structured_payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Full structured data from the tool (stored as JSON in SQLite).",
    )


# ── Researcher output ──────────────────────────────────────────────────────────

class SubquestionAnswer(BaseModel):
    """The Researcher's answer to a single subquestion."""
    subquestion: str
    summary: str
    source_ids: list[str]


class Claim(BaseModel):
    """A single factual claim produced by the Researcher."""
    claim_id: str = Field(..., description="Stable id: clm_{run_id_prefix}_{n:03d}")
    text: str
    source_ids: list[str] = Field(..., min_length=1)
    support_type: SupportType


class ResearchResult(BaseModel):
    """Full output of the Researcher agent."""
    subquestion_answers: list[SubquestionAnswer]
    claims: list[Claim]
    gaps: list[str] = Field(
        default_factory=list,
        description="Things the Researcher could not find evidence for.",
    )


class RagAdequacy(str, Enum):
    """Whether retrieved documents fully answer the query."""
    complete = "complete"  # docs alone suffice
    partial = "partial"    # docs help but external research is needed
    none = "none"          # docs do not answer the question


class RagPhaseOutput(BaseModel):
    """
    Structured output of the RAG extraction pass (same evidence shape as ResearchResult,
    plus an adequacy judgment).
    """
    adequacy: RagAdequacy = Field(
        ...,
        description=(
            "complete: user documents fully answer the query; partial: only some angles "
            "or evidence; none: documents are irrelevant or insufficient."
        ),
    )
    subquestion_answers: list[SubquestionAnswer]
    claims: list[Claim]
    gaps: list[str] = Field(
        default_factory=list,
        description="Topics still unclear from the retrieved passages alone.",
    )


# ── Reviewer output ────────────────────────────────────────────────────────────

class ClaimReview(BaseModel):
    """Verdict on a single claim from the Reviewer."""
    claim_id: str
    verdict: ClaimVerdict
    notes: str = Field(..., description="Why this verdict was assigned.")
    final_source_ids: list[str] = Field(
        ...,
        description="Source ids that actually back this verdict (may differ from claim's original list).",
    )
    needs_recheck: bool = False


class GlobalDecision(BaseModel):
    """Reviewer's overall decision about whether the pipeline needs a retry."""
    needs_retry: bool
    retry_focus_subquestions: list[str] = Field(
        default_factory=list,
        description="Subquestions the Researcher should re-investigate on retry.",
    )
    unsupported_claim_ids: list[str] = Field(
        default_factory=list,
        description="Claim ids that are unsupported and drove the retry decision.",
    )


class ClaimReviewSet(BaseModel):
    """Full output of the Reviewer agent."""
    claim_reviews: list[ClaimReview]
    global_decision: GlobalDecision

    def approved(self) -> list[ClaimReview]:
        """Return only verified and partially_verified reviews."""
        return [
            r for r in self.claim_reviews
            if r.verdict in (ClaimVerdict.verified, ClaimVerdict.partially_verified)
        ]

    def rejected(self) -> list[ClaimReview]:
        """Return unsupported and contradicted reviews."""
        return [
            r for r in self.claim_reviews
            if r.verdict in (ClaimVerdict.unsupported, ClaimVerdict.contradicted)
        ]


# ── Retry instruction ──────────────────────────────────────────────────────────

class UnsupportedClaimDetail(BaseModel):
    """Details about a rejected claim passed back to the Researcher on retry."""
    claim_id: str
    claim_text: str
    rejection_reason: str


class RetryInstruction(BaseModel):
    """
    Structured instruction sent to the Researcher when the Reviewer
    requests a second pass. Built by the orchestrator from ClaimReviewSet.
    """
    retry_reason: str
    focus_subquestions: list[str]
    unsupported_claims: list[UnsupportedClaimDetail]
    gaps_to_fill: list[str]
    already_retrieved_source_ids: list[str] = Field(
        ...,
        description="Source ids already in the DB — Researcher must not re-fetch these.",
    )
    suggested_tools: list[str]
    remaining_tool_budget: int


# ── Formatter output ───────────────────────────────────────────────────────────

class ReportSection(BaseModel):
    heading: str
    paragraphs: list[str]


class FinalReport(BaseModel):
    """Full output of the Formatter agent."""
    title: str
    as_of: str  # free-form date string from the LLM; parsed downstream
    output_style: OutputStyle
    executive_summary: list[str] = Field(
        ...,
        description="2-4 bullet points summarising the key findings.",
    )
    sections: list[ReportSection]
    unverified_items: list[str] = Field(
        default_factory=list,
        description="Claims or items that could not be verified — appear in caveats only.",
    )
    reference_source_ids: list[str] = Field(
        ...,
        description="All source ids cited anywhere in the report.",
    )

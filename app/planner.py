"""
Planner agent.

Turns a user query into a structured PlannerOutput (research_angles,
subquestions, suggested_tools) via a single SGLang call.

No MCP tools. No database writes. Pure in → out.

Usage:
    from app.planner import plan
    result = plan(query_input)   # returns PlannerOutput
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.schemas import PlannerOutput, QueryInput
from app.sglang_client import PLANNER_PROFILE, chat

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "planner.txt"


def plan(query_input: QueryInput) -> PlannerOutput:
    """
    Decompose a user query into a structured research plan.

    Args:
        query_input: The normalised user request.

    Returns:
        PlannerOutput with research_angles, subquestions, and suggested_tools.

    Raises:
        ValueError: If the model cannot produce a valid PlannerOutput after retry.
    """
    system_prompt = _load_prompt(query_input)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": query_input.query},
    ]

    logger.info(f"[planner] Planning query: {query_input.query!r}")

    result = chat(messages, profile=PLANNER_PROFILE, response_schema=PlannerOutput)

    if result.parsed is None:
        raise ValueError(
            f"Planner returned no structured output.\n"
            f"Raw content: {result.content}"
        )

    output: PlannerOutput = result.parsed
    logger.info(
        f"[planner] Done — angles={[a.value for a in output.research_angles]} "
        f"subquestions={len(output.subquestions)} "
        f"tools={output.suggested_tools}"
    )
    return output


# ── Prompt loading ─────────────────────────────────────────────────────────────

def _load_prompt(query_input: QueryInput) -> str:
    """Read the planner prompt template and fill in the dynamic fields."""
    template = _PROMPT_PATH.read_text(encoding="utf-8")
    return template.format(
        as_of=query_input.as_of.strftime("%Y-%m-%d %H:%M UTC"),
    )

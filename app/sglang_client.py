"""
Inference client — OpenAI-compatible backend (Ollama / SGLang / vLLM).

All agent modules call chat() to make LLM calls. The backend is configured
via SGLANG_BASE_URL and SGLANG_MODEL_ID in .env, so switching from Ollama
to SGLang or vLLM is one line change with no code edits.

Usage:
    from app.sglang_client import chat, PLANNER_PROFILE, REVIEWER_PROFILE
    result = chat(messages, profile=PLANNER_PROFILE, response_schema=PlannerOutput)
    plan = result.parsed   # already validated PlannerOutput instance
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI
from pydantic import BaseModel

from app.config import settings
from app.trace import trace_llm_request, trace_llm_response

logger = logging.getLogger(__name__)

# ── OpenAI-compatible client (points to Ollama / SGLang / vLLM) ───────────────

_client = OpenAI(
    base_url=settings.sglang_base_url,
    api_key="ollama",   # Ollama ignores this; required field for the client
)


# ── Call profiles ──────────────────────────────────────────────────────────────

@dataclass
class CallProfile:
    """Fixed inference settings for one type of agent call."""
    name:            str   = "UNKNOWN"
    temperature:     float = 0.0
    max_tokens:      int   = 2000
    enable_thinking: bool  = False   # prepends <|think|> to system prompt


# Named profiles — match the plan exactly.
PLANNER_PROFILE    = CallProfile(name="PLANNER",    temperature=0.0, max_tokens=1000,  enable_thinking=False)
RESEARCHER_PROFILE = CallProfile(name="RESEARCHER", temperature=0.0, max_tokens=4000,  enable_thinking=False)
REVIEWER_PROFILE   = CallProfile(name="REVIEWER",   temperature=0.0, max_tokens=4000,  enable_thinking=True)
FORMATTER_PROFILE  = CallProfile(name="FORMATTER",  temperature=0.0, max_tokens=3000,  enable_thinking=False)


# ── Result types ───────────────────────────────────────────────────────────────

@dataclass
class ToolCall:
    """A single tool call requested by the model."""
    id:        str
    name:      str
    arguments: dict[str, Any]


@dataclass
class ChatResult:
    """
    Unified return value from chat().

    - content:    Raw text content (may be None if the response is tool calls only).
    - tool_calls: Tool calls requested by the model (empty list if none).
    - parsed:     Pydantic-validated structured output (None if no response_schema).
    - thinking:   Extracted thinking trace, stripped from content (debug only).
    """
    content:    str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    parsed:     BaseModel | None = None
    thinking:   str | None = None


# ── Main entry point ───────────────────────────────────────────────────────────

def chat(
    messages:        list[dict[str, Any]],
    profile:         CallProfile,
    response_schema: type[BaseModel] | None = None,
    tools:           list[dict[str, Any]] | None = None,
) -> ChatResult:
    """
    Make one LLM call and return a ChatResult.

    Args:
        messages:        Standard OpenAI message list.
        profile:         CallProfile controlling temperature, max_tokens, thinking.
        response_schema: If provided, the model is asked to return JSON matching
                         this Pydantic schema. result.parsed will be populated.
        tools:           OpenAI-format tool definitions. If provided, the model
                         may return tool_calls instead of (or in addition to) content.

    Returns:
        ChatResult with content, tool_calls, parsed, and thinking fields.

    Raises:
        ValueError: If response_schema is provided but the response cannot be
                    parsed after one retry.
        openai.APIError: On persistent API-level failures.
    """
    messages = _apply_thinking(messages, profile.enable_thinking)

    kwargs: dict[str, Any] = {
        "model":      settings.sglang_model_id,
        "messages":   messages,
        "temperature": profile.temperature,
        "max_tokens":  profile.max_tokens,
        # Ollama needs num_ctx to override its 4096 default.
        # The openai client passes unknown kwargs through as extra_body.
        "extra_body": {"options": {"num_ctx": settings.sglang_context_limit}},
    }

    if response_schema is not None:
        kwargs["response_format"] = _json_schema_format(response_schema)

    if tools:
        kwargs["tools"]       = tools
        kwargs["tool_choice"] = "auto"

    logger.debug(
        f"[llm] call model={settings.sglang_model_id!r} "
        f"schema={response_schema.__name__ if response_schema else None} "
        f"tools={[t['function']['name'] for t in tools] if tools else None} "
        f"thinking={profile.enable_thinking}"
    )

    trace_llm_request(
        phase=profile.name,
        messages=messages,
        schema_name=response_schema.__name__ if response_schema else None,
        tools=tools,
    )

    response = _client.chat.completions.create(**kwargs)
    msg = response.choices[0].message

    # ── Extract tool calls ─────────────────────────────────────────────────────
    parsed_tool_calls: list[ToolCall] = []
    if msg.tool_calls:
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
                logger.warning(f"[llm] Could not parse tool call args for {tc.function.name!r}")
            parsed_tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

    # ── Extract content and thinking trace ─────────────────────────────────────
    raw_content = msg.content or ""
    thinking, clean_content = _strip_thinking(raw_content)

    # ── Parse structured output ────────────────────────────────────────────────
    parsed: BaseModel | None = None
    if response_schema is not None and clean_content.strip():
        parsed = _parse_with_retry(
            content=clean_content,
            schema=response_schema,
            messages=messages,
            profile=profile,
            kwargs=kwargs,
        )

    logger.debug(
        f"[llm] response tool_calls={len(parsed_tool_calls)} "
        f"content_len={len(clean_content)} parsed={parsed is not None} "
        f"thinking={'yes' if thinking else 'no'}"
    )

    trace_llm_response(
        phase=profile.name,
        content=clean_content or None,
        tool_calls=parsed_tool_calls,
        parsed=parsed,
    )

    return ChatResult(
        content=clean_content or None,
        tool_calls=parsed_tool_calls,
        parsed=parsed,
        thinking=thinking,
    )


# ── Thinking helpers ───────────────────────────────────────────────────────────

# Gemma 4 thinking format: <think>[reasoning]</think>
# The regex captures everything between the opening and closing tags.
_THINKING_RE = re.compile(
    r"<think>(.*?)</think>",
    re.DOTALL,
)


def _apply_thinking(
    messages: list[dict[str, Any]],
    enable: bool,
) -> list[dict[str, Any]]:
    """
    Nudge the model to produce a <think> block when thinking mode is on.

    With Ollama's gemma4:4b-thinking variant the model produces <think>...</think>
    blocks automatically. This function adds "Think step by step." to the system
    prompt as an extra cue; the _strip_thinking regex handles extraction.
    """
    if not enable:
        return messages

    messages = list(messages)  # shallow copy — don't mutate caller's list
    system_idx = next(
        (i for i, m in enumerate(messages) if m.get("role") == "system"), None
    )
    thinking_cue = "Think step by step before answering."
    if system_idx is not None:
        existing = messages[system_idx]["content"]
        if thinking_cue not in existing:
            messages[system_idx] = {
                **messages[system_idx],
                "content": thinking_cue + "\n" + existing,
            }
    else:
        messages.insert(0, {"role": "system", "content": thinking_cue})

    return messages


def _strip_thinking(text: str) -> tuple[str | None, str]:
    """
    Extract and remove the Gemma 4 thinking trace from a response.

    Returns:
        (thinking_trace, clean_content) — thinking_trace is None if no trace found.
    """
    match = _THINKING_RE.search(text)
    if not match:
        return None, text
    thinking = match.group(1).strip()
    clean = _THINKING_RE.sub("", text).strip()
    return thinking, clean


# ── Structured output helpers ──────────────────────────────────────────────────

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\n(.*?)\n```", re.DOTALL)


def _extract_json(text: str) -> str:
    """Extract a JSON object from text that may contain markdown fences or prose."""
    text = text.strip()
    # Try markdown code block first
    m = _JSON_BLOCK_RE.search(text)
    if m:
        return m.group(1).strip()
    # Find first { and last } to extract raw JSON
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return text[start:end + 1]
    return text


def _json_schema_format(schema: type[BaseModel]) -> dict[str, Any]:
    """Build the OpenAI response_format dict for a Pydantic schema."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name":   schema.__name__,
            "schema": schema.model_json_schema(),
            "strict": False,   # strict=True can cause issues with optional fields
        },
    }


def _parse_with_retry(
    content: str,
    schema: type[BaseModel],
    messages: list[dict],
    profile: CallProfile,
    kwargs: dict,
) -> BaseModel:
    """
    Try to parse content as schema. If it fails, make one correction call
    with the validation error included, then raise if it fails again.
    """
    # Try to extract JSON from the content (model may wrap it in markdown/text)
    content = _extract_json(content)
    first_err_msg = ""
    try:
        return schema.model_validate_json(content)
    except Exception as e:
        first_err_msg = str(e)
        logger.warning(
            f"[llm] Failed to parse {schema.__name__}: {first_err_msg}. "
            f"Retrying with correction prompt."
        )

    # Build a correction follow-up
    correction_messages = list(messages) + [
        {"role": "assistant", "content": content},
        {
            "role": "user",
            "content": (
                f"The previous response could not be parsed as {schema.__name__}. "
                f"Error: {first_err_msg}\n"
                f"Please return only valid JSON that exactly matches the schema. "
                f"No extra text, no markdown, no code fences."
            ),
        },
    ]
    retry_kwargs = {**kwargs, "messages": correction_messages}
    retry_response = _client.chat.completions.create(**retry_kwargs)
    retry_content = retry_response.choices[0].message.content or ""
    _, retry_content = _strip_thinking(retry_content)

    retry_content = _extract_json(retry_content)
    try:
        return schema.model_validate_json(retry_content)
    except Exception as second_err:
        raise ValueError(
            f"Could not parse {schema.__name__} after retry. "
            f"Last error: {second_err}\n"
            f"Last content: {retry_content[:300]}"
        ) from second_err


# ── Tool definition builder ────────────────────────────────────────────────────

def mcp_tools_to_openai(tool_descriptors: list[dict]) -> list[dict[str, Any]]:
    """
    Convert MCP /tools descriptors to OpenAI function-calling format.

    MCP format:   {"name": ..., "description": ..., "parameters": {...}, "required": [...]}
    OpenAI format: {"type": "function", "function": {"name": ..., "description": ...,
                    "parameters": {"type": "object", "properties": {...}, "required": [...]}}}
    """
    result = []
    for t in tool_descriptors:
        result.append({
            "type": "function",
            "function": {
                "name":        t["name"],
                "description": t.get("description", ""),
                "parameters": {
                    "type":       "object",
                    "properties": t.get("parameters", {}),
                    "required":   t.get("required", []),
                },
            },
        })
    return result

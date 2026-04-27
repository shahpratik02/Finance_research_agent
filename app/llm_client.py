"""
Inference client — OpenAI-compatible backend (Ollama, SGLang, vLLM).

All agent modules call chat() to make LLM calls. The backend is configured
via LLM_BASE_URL and LLM_MODEL_ID in .env.

Usage:
    from app.llm_client import chat, PLANNER_PROFILE, REVIEWER_PROFILE
    result = chat(messages, profile=PLANNER_PROFILE, response_schema=PlannerOutput)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI
from pydantic import BaseModel

from app.config import settings
from app.trace import trace_llm_request, trace_llm_response

logger = logging.getLogger(__name__)

# ── OpenAI-compatible client ──────────────────────────────────────────────────

_client = OpenAI(
    base_url=settings.llm_base_url,
    api_key="local",
    timeout=None,  # Disabled timeout to prevent errors on low-memory macs
)


# ── Call profiles ──────────────────────────────────────────────────────────────

@dataclass
class CallProfile:
    name: str
    temperature: float
    max_tokens: int
    enable_thinking: bool


PLANNER_PROFILE = CallProfile(
    name="PLANNER", temperature=0.0, max_tokens=1000, enable_thinking=False,
)
RESEARCHER_PROFILE = CallProfile(
    name="RESEARCHER", temperature=0.0, max_tokens=2200, enable_thinking=False,
)
REVIEWER_PROFILE = CallProfile(
    name="REVIEWER", temperature=0.0, max_tokens=2400, enable_thinking=True,
)
FORMATTER_PROFILE = CallProfile(
    name="FORMATTER", temperature=0.0, max_tokens=2000, enable_thinking=False,
)
RAG_PROFILE = CallProfile(
    name="RAG", temperature=0.0, max_tokens=4096, enable_thinking=False,
)

# ── Return types ───────────────────────────────────────────────────────────────

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatResult:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    parsed: BaseModel | None = None
    thinking: str | None = None


# ── Main chat method ───────────────────────────────────────────────────────────

def chat(
    messages: list[dict[str, Any]],
    profile: CallProfile,
    response_schema: type[BaseModel] | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> ChatResult:
    """
    Execute a single LLM chat completion using the OpenAI-compatible client.
    """
    messages = _apply_thinking(messages, profile.enable_thinking)

    kwargs: dict[str, Any] = {
        "model":      settings.llm_model_id,
        "messages":   messages,
        "temperature": profile.temperature,
        "max_tokens":  profile.max_tokens,
    }

    # Pass context limit if needed for some local backends (like Ollama v1 endpoint logic)
    if settings.llm_context_limit:
        kwargs["extra_body"] = {"options": {"num_ctx": settings.llm_context_limit}}

    if response_schema is not None:
        kwargs["response_format"] = _json_schema_format(response_schema)

    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    logger.debug(
        f"[llm] call model={settings.llm_model_id!r} "
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

    try:
        response = _client.chat.completions.create(**kwargs)
        msg = response.choices[0].message
    except Exception as e:
        logger.error(f"[llm] API error via OpenAI library: {e}")
        raise

    # ── Extract tool calls ─────────────────────────────────────────────────────
    parsed_tool_calls: list[ToolCall] = []
    if msg.tool_calls:
        for tc in msg.tool_calls:
            # Arguments from OpenAI SDK are stringified JSON objects.
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


# ── Helpers ────────────────────────────────────────────────────────────────────

def _apply_thinking(messages: list[dict], enable: bool) -> list[dict]:
    """If true, append a directive to think before answering."""
    if not enable:
        return messages
    out = list(messages)
    last = out[-1]
    if last["role"] == "user":
        content = last["content"]
        content += "\n\nFirst think step-by-step in <think>...</think> tags, then provide your final answer."
        out[-1] = {"role": "user", "content": content}
    return out


def _strip_thinking(text: str) -> tuple[str | None, str]:
    """Extract <think> tags. Return (thinking, clean_content)."""
    if not text:
        return None, ""
    
    # Try unclosed thinking blocks first
    if text.startswith("<think>") and "</think>" not in text:
        return text[7:], ""
        
    start = text.find("<think>")
    end = text.find("</think>")
    if start != -1 and end != -1:
        thinking = text[start + 7 : end].strip()
        clean = text[:start] + text[end + 8:]
        return thinking, clean.strip()
    return None, text.strip()


def _extract_json(text: str) -> str:
    """Find the first {...} or [...] block in a text string."""
    start_obj = text.find("{")
    start_arr = text.find("[")
    
    if start_obj == -1 and start_arr == -1:
        return text
        
    start = start_obj if (start_arr == -1 or (start_obj != -1 and start_obj < start_arr)) else start_arr
    opening_char = "{" if start == start_obj else "["
    closing_char = "}" if opening_char == "{" else "]"
    
    depth = 0
    in_string = False
    escape = False
    
    for i in range(start, len(text)):
        char = text[i]
        
        if escape:
            escape = False
            continue
            
        if char == "\\":
            escape = True
            continue
            
        if char == '"':
            in_string = not in_string
            continue
            
        if not in_string:
            if char == opening_char:
                depth += 1
            elif char == closing_char:
                depth -= 1
                if depth == 0:
                    return text[start:i+1]
                    
    return text


def _json_schema_format(schema: type[BaseModel]) -> dict[str, Any]:
    """Build the OpenAI response_format dict for a Pydantic schema."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name":   schema.__name__,
            "schema": schema.model_json_schema(),
            "strict": False,
        },
    }


def _parse_with_retry(
    content: str,
    schema: type[BaseModel],
    messages: list[dict],
    profile: CallProfile,
    kwargs: dict,
) -> BaseModel | None:
    content = _extract_json(content)
    parse_error: Exception | None = None
    try:
        return schema.model_validate_json(content)
    except Exception as e:
        parse_error = e
        logger.warning(f"[llm] Failed to parse output (retrying): {e}")

    correction_messages = messages + [
        {"role": "assistant", "content": content},
        {
            "role": "user",
            "content": (
                "Your previous output failed JSON validation. "
                f"Error: {parse_error}\n"
                "Please fix the error and output ONLY valid JSON matching the schema."
            ),
        },
    ]
    retry_kwargs = {**kwargs, "messages": correction_messages}
    try:
        retry_response = _client.chat.completions.create(**retry_kwargs)
        retry_content = retry_response.choices[0].message.content or ""
        _, retry_content = _strip_thinking(retry_content)
        retry_content = _extract_json(retry_content)
        return schema.model_validate_json(retry_content)
    except Exception as e2:
        logger.error(f"[llm] Retry also failed to parse: {e2}")
        return None


def embed_texts(texts: list[str], model_id: str) -> list[list[float]]:
    """
    Embedding vectors via the configured OpenAI-compatible API.
    Order matches `texts` one-to-one.
    """
    if not texts:
        return []
    resp = _client.embeddings.create(model=model_id, input=texts)
    data = list(resp.data)
    try:
        data.sort(key=lambda d: d.index)
    except Exception:
        pass
    return [d.embedding for d in data]

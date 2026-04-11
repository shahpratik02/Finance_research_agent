"""
SGLang inference client.

Responsibilities:
- Wrap the SGLang OpenAI-compatible HTTP API.
- Provide a single call() function used by all agents:
      call(prompt, system, schema, tools, profile) -> parsed structured output
- Apply per-call profiles (temperature, max_tokens, enable_thinking, tool_calls).
- Handle SGLang structured output (constrained decoding via JSON schema).
- Retry once on malformed structured output before raising.
- Strip thinking traces from Reviewer responses before returning.

All agent modules (planner, researcher, reviewer, formatter) go through this client.
No agent module makes raw HTTP calls directly.
"""

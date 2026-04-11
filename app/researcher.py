"""
Researcher agent.

Responsibilities:
- Accept the planner output (and optionally a RetryInstruction for a second pass).
- Call the SGLang model (via sglang_client) with the researcher prompt and tool access.
- Invoke MCP tools (via mcp_client) to retrieve sources, staying within the tool budget.
- Normalize each raw tool response to a SourceRecord via source_normalizer.
- Return a structured ResearchResult (subquestion_answers, claims, gaps).

Settings used: temperature=0.0, max_tokens=2200, enable_thinking=False, tool_calls=True.
Tool budget: 20 calls on first pass, 6 calls on retry pass.
"""

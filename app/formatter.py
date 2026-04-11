"""
Formatter agent.

Responsibilities:
- Accept only the verified and partially_verified claims (filtered by orchestrator).
- Accept the resolved source references for those claims.
- Call the SGLang model (via sglang_client) with the formatter prompt.
- Return a structured FinalReport (title, executive_summary, sections, unverified_items, reference_source_ids).

This agent has no MCP tool access. It performs no verification.
It only writes the report from the approved claims it receives.

Settings used: temperature=0.0, max_tokens=2000, enable_thinking=False, tool_calls=False.
"""

"""
Reviewer agent.

Responsibilities:
- Accept the ResearchResult and all normalized sources.
- Call the SGLang model (via sglang_client) with the reviewer prompt.
- Optionally invoke MCP tools (via mcp_client) for limited targeted re-checking.
- Return a structured ClaimReviewSet (verdicts per claim, global_decision with retry flag).

This agent does NOT write the final report. It only produces claim verdicts.

Settings used: temperature=0.0, max_tokens=2400, enable_thinking=True, tool_calls=True.
Tool budget: 6 calls maximum. Thinking trace is stripped before passing output downstream.
"""

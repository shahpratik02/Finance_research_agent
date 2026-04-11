"""
Planner agent.

Responsibilities:
- Accept the normalized user query.
- Call the SGLang model (via sglang_client) with the planner prompt.
- Return a structured PlannerOutput (research_angles, subquestions, suggested_tools).

Settings used: temperature=0.0, max_tokens=1000, enable_thinking=False.
No tool calls. Structured output only.
"""

"""
MCP tool client.

Responsibilities:
- Provide a unified call interface for all four MCP servers:
      yahoo_finance  — stock prices, market cap, basic fundamentals
      fred           — macro time series (rates, inflation, unemployment)
      financial_datasets — financial statements, richer company data
      open_web_search    — recent news, general discovery
- Route tool calls to the correct MCP server based on tool name.
- Enforce the per-agent tool budget (tracked externally by the agent module).
- Return raw tool responses as dicts; normalization is done in source_normalizer.py.
- Retry once on transient network errors. Log and continue on persistent failures.
"""

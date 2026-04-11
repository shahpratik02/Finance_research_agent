"""
Source normalizer.

Responsibilities:
- Accept a raw dict response from any MCP tool.
- Map it to the standard SourceRecord schema defined in schemas.py.
- Generate a stable source_id in the format:
      {run_id_prefix}-{provider_prefix}-{uuid4_short}
  e.g. a3f9b12c-yhoo-d7e4c091
- Provider prefix map:
      yahoo_finance      → yhoo
      fred               → fred
      financial_datasets → find
      open_web_search    → webs
- Extract content_summary and raw_excerpt from the raw response.
- Attach retrieved_at timestamp.
- Return a validated SourceRecord instance.

Source ids are immutable once assigned.
"""

"""
Report renderer.

Responsibilities:
- Accept a FinalReport structured object.
- Resolve reference_source_ids to full source metadata (title, URL, provider) via the db.
- Render the report as a markdown string following the required section order:
    1. Title + as-of timestamp
    2. Executive summary
    3. Main analysis sections
    4. Risks / caveats
    5. Unverified or conflicting items
    6. References

Returns a plain markdown string ready for display or storage.
"""

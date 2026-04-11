"""
SQLite database layer.

Responsibilities:
- Initialize the database and create all tables on first run.
- Provide simple insert and query functions for each table:
    - runs        (one row per user query)
    - sources     (normalized SourceRecord rows per run)
    - claims      (Researcher claims per run)
    - reviews     (Reviewer verdicts per claim)
    - reports     (final markdown report per run)

No ORM. Uses Python's built-in sqlite3 module directly.
All functions accept and return plain dicts or dataclass instances.
"""

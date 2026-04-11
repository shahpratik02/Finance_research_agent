"""
Database table definitions.

Responsibilities:
- Define the schema for all five SQLite tables as Python dataclasses or namedtuples:
    - Run
    - Source
    - Claim
    - Review
    - Report
- These are plain data containers, not ORM models.
- Used by db.py for insert/query operations and by schemas.py for validation.
"""

"""
Pipeline orchestrator.

Responsibilities:
- Coordinate the full research pipeline for a single user query.
- Call: planner → researcher → reviewer → (optional retry) → formatter → renderer.
- Build the RetryInstruction when the Reviewer requests a retry.
- Save all artifacts (sources, claims, reviews, final report) to SQLite via db.py.
- Return the completed run artifact.

This module contains no LLM logic — it only wires the other modules together.
"""

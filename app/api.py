"""
FastAPI route definitions.

Responsibilities:
- POST /research  — accept a user query, trigger the pipeline, return the final report.
- GET  /runs/{run_id} — retrieve stored artifacts for a past run.
- GET  /runs/{run_id}/sources — retrieve the normalized sources for a run.
"""

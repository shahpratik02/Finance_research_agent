"""
Pydantic schemas for all structured LLM inputs and outputs.

Defines the following schemas:
    Input schemas:
        - QueryInput           (user request: query, as_of, output_style)
        - RetryInstruction     (orchestrator → researcher on retry pass)

    Agent output schemas:
        - PlannerOutput        (research_angles, subquestions, suggested_tools)
        - ResearchResult       (subquestion_answers, claims, gaps)
        - ClaimReviewSet       (claim_reviews[], global_decision)
        - FinalReport          (title, as_of, sections[], unverified_items, reference_source_ids)

    Intermediate schemas:
        - SourceRecord         (normalized MCP tool response)
        - Claim                (single claim with source_ids)
        - ClaimReview          (single claim verdict)

All schemas are used for:
    - Validating LLM structured outputs.
    - Serializing/deserializing to/from SQLite JSON columns.
    - Defining the JSON schema passed to SGLang for constrained decoding.
"""

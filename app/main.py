"""
FastAPI application entry point.

Responsibilities:
- Create and configure the FastAPI app instance.
- Mount the API router from api.py.
- Initialize the database on startup.
- Run with uvicorn.
"""

import logging

from fastapi import FastAPI

from app.api import router
from app.config import settings
from app.db import init_db

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)

# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Finance Research Agent",
    version="1.0.0",
    description="Local LLM-powered finance research pipeline.",
)

app.include_router(router)


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    logging.getLogger(__name__).info("Database initialised, app ready.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)

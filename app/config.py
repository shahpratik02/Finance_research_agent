"""
Configuration management.

Responsibilities:
- Load .env via python-dotenv at import time.
- Expose a single Settings object with typed fields for every config value.
- Raise clearly if a required field is missing.

All other modules import Settings from here. Nothing is hardcoded elsewhere.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (parent of the app/ directory).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_env_path = _PROJECT_ROOT / ".env"
if _env_path.exists():
    load_dotenv(_env_path, override=True)
else:
    load_dotenv(override=True)  # fall back to env vars already in the environment


class Settings:
    """
    Typed, validated access to every configuration value the agent needs.

    Instantiate once at module level and import the singleton everywhere:

        from app.config import settings
    """

    # ── SGLang / inference ─────────────────────────────────────────────────────
    sglang_base_url: str
    sglang_model_id: str
    sglang_context_limit: int

    # ── MCP server URLs ────────────────────────────────────────────────────────
    yahoo_finance_mcp_url: str
    fred_mcp_url: str
    financial_datasets_mcp_url: str
    open_web_search_mcp_url: str

    # ── Tool budgets ───────────────────────────────────────────────────────────
    researcher_tool_budget: int
    reviewer_tool_budget: int
    retry_tool_budget: int

    # ── Storage ────────────────────────────────────────────────────────────────
    sqlite_path: Path

    # ── Logging ────────────────────────────────────────────────────────────────
    log_level: str

    # ── Debug ──────────────────────────────────────────────────────────────────
    debug_trace: bool

    def __init__(self) -> None:
        # ── SGLang ────────────────────────────────────────────────────────────
        self.sglang_base_url = self._require("SGLANG_BASE_URL")
        self.sglang_model_id = self._require("SGLANG_MODEL_ID")
        self.sglang_context_limit = int(os.environ.get("SGLANG_CONTEXT_LIMIT", "32768"))

        # ── MCP server URLs ───────────────────────────────────────────────────
        self.yahoo_finance_mcp_url = os.environ.get(
            "YAHOO_FINANCE_MCP_URL", "http://127.0.0.1:8001"
        ).rstrip("/")
        self.fred_mcp_url = os.environ.get(
            "FRED_MCP_URL", "http://127.0.0.1:8002"
        ).rstrip("/")
        self.financial_datasets_mcp_url = os.environ.get(
            "FINANCIAL_DATASETS_MCP_URL", "http://127.0.0.1:8003"
        ).rstrip("/")
        self.open_web_search_mcp_url = os.environ.get(
            "OPEN_WEB_SEARCH_MCP_URL", "http://127.0.0.1:8004"
        ).rstrip("/")

        # ── Tool budgets ──────────────────────────────────────────────────────
        self.researcher_tool_budget = int(os.environ.get("RESEARCHER_TOOL_BUDGET", "20"))
        self.reviewer_tool_budget = int(os.environ.get("REVIEWER_TOOL_BUDGET", "6"))
        self.retry_tool_budget = int(os.environ.get("RETRY_TOOL_BUDGET", "6"))

        # ── Storage ───────────────────────────────────────────────────────────
        sqlite_raw = os.environ.get("SQLITE_PATH", "data/app.db")
        self.sqlite_path = (
            Path(sqlite_raw)
            if Path(sqlite_raw).is_absolute()
            else _PROJECT_ROOT / sqlite_raw
        )

        # ── Logging ───────────────────────────────────────────────────────────
        self.log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
        logging.basicConfig(
            level=getattr(logging, self.log_level, logging.INFO),
            format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        )
        # ── Debug ─────────────────────────────────────────────────────────
        self.debug_trace = os.environ.get("DEBUG_TRACE", "").lower() in ("1", "true", "yes")
    # ── MCP URL lookup helper ──────────────────────────────────────────────────

    def mcp_url_for(self, provider: str) -> str:
        """
        Return the base URL for a given MCP provider name.

        Args:
            provider: One of 'yahoo_finance', 'fred', 'financial_datasets',
                      'open_web_search'.

        Raises:
            ValueError: If the provider name is not recognized.
        """
        mapping = {
            "yahoo_finance": self.yahoo_finance_mcp_url,
            "fred": self.fred_mcp_url,
            "financial_datasets": self.financial_datasets_mcp_url,
            "open_web_search": self.open_web_search_mcp_url,
        }
        if provider not in mapping:
            raise ValueError(
                f"Unknown MCP provider {provider!r}. "
                f"Valid providers: {sorted(mapping.keys())}"
            )
        return mapping[provider]

    # ── Internal helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _require(key: str) -> str:
        value = os.environ.get(key, "")
        if not value:
            raise EnvironmentError(
                f"Required environment variable {key!r} is not set. "
                f"Add it to your .env file."
            )
        return value

    def __repr__(self) -> str:
        return (
            f"Settings("
            f"sglang_base_url={self.sglang_base_url!r}, "
            f"yahoo_finance_mcp_url={self.yahoo_finance_mcp_url!r}, "
            f"fred_mcp_url={self.fred_mcp_url!r}, "
            f"financial_datasets_mcp_url={self.financial_datasets_mcp_url!r}, "
            f"open_web_search_mcp_url={self.open_web_search_mcp_url!r}, "
            f"researcher_tool_budget={self.researcher_tool_budget}, "
            f"reviewer_tool_budget={self.reviewer_tool_budget}"
            f")"
        )


# ── Module-level singleton ─────────────────────────────────────────────────────
# Import this everywhere: `from app.config import settings`
settings = Settings()

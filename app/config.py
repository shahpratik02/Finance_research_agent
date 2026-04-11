"""
Configuration management.

Responsibilities:
- Load .env via python-dotenv at import time.
- Expose a single Settings object with typed fields for every config value.
- Raise clearly if a required field is missing.

All other modules import Settings from here. Nothing is hardcoded elsewhere.
"""

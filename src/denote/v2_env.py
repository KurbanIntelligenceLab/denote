"""Load the OpenRouter API key from the repo root's .env."""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

# src/denote/v2_env.py -> repo root is three parents up (denote/ -> src/ -> root)
ROOT = Path(__file__).resolve().parents[2]


def bootstrap() -> None:
    """Load OPENROUTER_API_KEY from the repo root's .env, if present."""
    load_dotenv(ROOT / ".env", override=False)

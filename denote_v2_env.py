"""Load version1 credentials without copying .env into version2."""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
V1 = ROOT.parent / "version1"


def bootstrap() -> None:
    """Load OpenRouter key from version1/.env if present."""
    env_path = V1 / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)
    load_dotenv(ROOT / ".env", override=False)

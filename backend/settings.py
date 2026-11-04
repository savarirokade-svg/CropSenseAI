"""Typed persistence configuration loaded from backend/.env."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

load_dotenv(BACKEND_DIR / ".env")

DEFAULT_SESSION_TTL_DAYS = 30
DEFAULT_MAX_UPLOAD_BYTES = 10 * 1024 * 1024
DEFAULT_POSTGRES_HOST = "127.0.0.1"
DEFAULT_POSTGRES_PORT = 5433


class SettingsError(ValueError):
    """Raised when a required setting is missing or invalid."""


def _require(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise SettingsError(f"{name} is not set. Add it to backend/.env.")
    return value


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise SettingsError(f"{name} must be an integer, got {raw!r}.") from None
    if value <= 0:
        raise SettingsError(f"{name} must be greater than zero, got {value}.")
    return value


def _build_database_url() -> str:
    url = os.getenv("DATABASE_URL", "").strip()
    if url:
        if url.startswith("postgresql://"):
            return "postgresql+asyncpg://" + url[len("postgresql://"):]
        if url.startswith("postgres://"):
            return "postgresql+asyncpg://" + url[len("postgres://"):]
        return url

    user = _require("POSTGRES_USER")
    password = _require("POSTGRES_PASSWORD")
    database = _require("POSTGRES_DB")
    host = os.getenv("POSTGRES_HOST", "").strip() or DEFAULT_POSTGRES_HOST
    port = _positive_int("POSTGRES_PORT", DEFAULT_POSTGRES_PORT)

    return (
        f"postgresql+asyncpg://{quote_plus(user)}:{quote_plus(password)}"
        f"@{host}:{port}/{quote_plus(database)}"
    )


def _resolve_uploads_dir() -> Path:
    raw = os.getenv("UPLOADS_DIR", "").strip()
    if not raw:
        return BACKEND_DIR / "data" / "uploads"
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


@dataclass(frozen=True)
class Settings:
    database_url: str = field(repr=False)
    uploads_dir: Path
    session_ttl_days: int
    max_upload_bytes: int


def load_settings() -> Settings:
    return Settings(
        database_url=_build_database_url(),
        uploads_dir=_resolve_uploads_dir(),
        session_ttl_days=_positive_int("SESSION_TTL_DAYS", DEFAULT_SESSION_TTL_DAYS),
        max_upload_bytes=_positive_int("MAX_UPLOAD_BYTES", DEFAULT_MAX_UPLOAD_BYTES),
    )


settings = load_settings()

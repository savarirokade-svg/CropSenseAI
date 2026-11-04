"""Anonymous session token generation, validation and FastAPI dependency."""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db import get_session
from models import Session as AnonymousSession
from settings import settings

_UNAUTHORIZED_DETAIL = "A valid X-Session-Token header is required."


def generate_session_token() -> str:
    """Return a new URL-safe 256-bit session token."""
    return secrets.token_urlsafe(32)


def hash_session_token(token: str) -> str:
    """Return the 64-character SHA-256 hex digest stored in the database."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def session_expires_at(now: datetime | None = None) -> datetime:
    """Return the expiry timestamp for a new session."""
    base = now or datetime.now(timezone.utc)
    return base + timedelta(days=settings.session_ttl_days)


async def create_session(db: AsyncSession) -> tuple[AnonymousSession, str]:
    """Create and commit an anonymous session, returning it with its raw token."""
    token = generate_session_token()
    session = AnonymousSession(
        token_hash=hash_session_token(token),
        expires_at=session_expires_at(),
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return session, token


async def validate_session_token(
    db: AsyncSession, token: str
) -> AnonymousSession | None:
    """Return the matching unexpired session, or None."""
    result = await db.execute(
        select(AnonymousSession).where(
            AnonymousSession.token_hash == hash_session_token(token)
        )
    )
    session = result.scalar_one_or_none()
    if session is None:
        return None
    if session.expires_at <= datetime.now(timezone.utc):
        return None
    return session


async def get_current_session(
    x_session_token: str | None = Header(default=None, alias="X-Session-Token"),
    db: AsyncSession = Depends(get_session),
) -> AnonymousSession:
    """FastAPI dependency returning the caller's anonymous session."""
    if not x_session_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=_UNAUTHORIZED_DETAIL,
        )

    session = await validate_session_token(db, x_session_token)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=_UNAUTHORIZED_DETAIL,
        )
    return session

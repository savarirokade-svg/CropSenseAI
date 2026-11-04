"""Database operations for persistent conversations and history."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

import storage
from models import (
    Conversation,
    Image,
    Message,
    MessageKGEdge,
    MessageSource,
    Session as AnonymousSession,
)

logger = logging.getLogger(__name__)

_TITLE_LENGTH = 60
_TITLE_MAX_LENGTH = 200


def _as_uuid(value: uuid.UUID | str) -> uuid.UUID:
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


def _title_from_message(message: str) -> str:
    text = " ".join((message or "").split())
    if not text:
        return "New conversation"
    if len(text) <= _TITLE_LENGTH:
        return text
    truncated = text[:_TITLE_LENGTH].rstrip()
    if " " in truncated:
        truncated = truncated.rsplit(" ", 1)[0].rstrip()
    return truncated or text[:_TITLE_LENGTH]


def _normalise_title(title: str) -> str:
    normalised = (title or "").strip()
    if not 1 <= len(normalised) <= _TITLE_MAX_LENGTH:
        raise ValueError("Conversation title must be between 1 and 200 characters.")
    return normalised


def _to_message_source(source: Mapping[str, Any]) -> MessageSource:
    citation = source.get("citation_number", source.get("citation"))
    if citation is None:
        raise ValueError("Source is missing a citation number.")
    return MessageSource(
        citation_number=int(citation),
        chunk_id=str(source["chunk_id"]),
        filename=source.get("filename"),
        title=source.get("title"),
        url=source.get("url"),
        institution=source.get("institution"),
        evidence_level=source.get("evidence_level"),
        disease_name=source.get("disease_name"),
        crop=source.get("crop"),
    )


def _to_message_kg_edge(edge: Mapping[str, Any]) -> MessageKGEdge:
    return MessageKGEdge(
        subject=str(edge["subject"]),
        relation=str(edge["relation"]),
        object=str(edge["object"]),
    )


async def _locked_conversation(
    db: AsyncSession,
    session_id: uuid.UUID | str,
    conversation_id: uuid.UUID | str,
) -> Conversation | None:
    result = await db.execute(
        select(Conversation)
        .where(
            Conversation.id == _as_uuid(conversation_id),
            Conversation.session_id == _as_uuid(session_id),
        )
        .with_for_update()
    )
    return result.scalar_one_or_none()


async def _next_sequence(db: AsyncSession, conversation_id: uuid.UUID) -> int:
    result = await db.execute(
        select(func.coalesce(func.max(Message.sequence), 0) + 1).where(
            Message.conversation_id == conversation_id
        )
    )
    return result.scalar_one()


async def create_conversation(
    db: AsyncSession,
    session_id: uuid.UUID | str,
    first_message: str,
    crop: str | None = None,
) -> Conversation:
    session_uuid = _as_uuid(session_id)
    if await db.get(AnonymousSession, session_uuid) is None:
        raise ValueError("Session not found.")

    conversation = Conversation(
        session_id=session_uuid,
        title=_title_from_message(first_message),
        crop=crop,
    )
    db.add(conversation)
    await db.commit()
    await db.refresh(conversation)
    return conversation


async def set_conversation_crop(
    db: AsyncSession,
    session_id: uuid.UUID | str,
    conversation_id: uuid.UUID | str,
    crop: str,
) -> Conversation | None:
    """Record the crop a conversation's question resolved to."""
    conversation = await _locked_conversation(db, session_id, conversation_id)
    if conversation is None:
        return None

    conversation.crop = crop
    await db.commit()
    return conversation


async def list_conversations(
    db: AsyncSession, session_id: uuid.UUID | str
) -> list[Conversation]:
    result = await db.execute(
        select(Conversation)
        .where(Conversation.session_id == _as_uuid(session_id))
        .order_by(Conversation.updated_at.desc(), Conversation.id.desc())
    )
    return list(result.scalars().all())


async def conversation_stats(
    db: AsyncSession, session_id: uuid.UUID | str
) -> dict[uuid.UUID, tuple[int, bool]]:
    """
    Message count and diagnosis flag for every conversation in one session.

    A single grouped query rather than a count per conversation, so listing stays
    a fixed number of round trips. ``has_diagnosis`` is true when any stored
    message carries a vision assessment, which is what the history presents as a
    diagnosis. Conversations with no messages simply have no row here.
    """
    result = await db.execute(
        select(
            Message.conversation_id,
            func.count(Message.id),
            func.count(Message.vision_assessment),
        )
        .join(Conversation, Conversation.id == Message.conversation_id)
        .where(Conversation.session_id == _as_uuid(session_id))
        .group_by(Message.conversation_id)
    )
    return {
        conversation_id: (int(message_count), bool(diagnoses))
        for conversation_id, message_count, diagnoses in result.all()
    }


async def get_conversation(
    db: AsyncSession,
    session_id: uuid.UUID | str,
    conversation_id: uuid.UUID | str,
) -> Conversation | None:
    result = await db.execute(
        select(Conversation)
        .where(
            Conversation.id == _as_uuid(conversation_id),
            Conversation.session_id == _as_uuid(session_id),
        )
        .options(
            selectinload(Conversation.messages).selectinload(Message.sources),
            selectinload(Conversation.messages).selectinload(Message.kg_edges),
            selectinload(Conversation.messages).selectinload(Message.image),
        )
    )
    return result.scalar_one_or_none()


async def rename_conversation(
    db: AsyncSession,
    session_id: uuid.UUID | str,
    conversation_id: uuid.UUID | str,
    title: str,
) -> Conversation | None:
    conversation = await _locked_conversation(db, session_id, conversation_id)
    if conversation is None:
        return None

    conversation.title = _normalise_title(title)
    await db.commit()
    return conversation


async def delete_conversation(
    db: AsyncSession,
    session_id: uuid.UUID | str,
    conversation_id: uuid.UUID | str,
) -> bool:
    session_uuid = _as_uuid(session_id)
    conversation = await _locked_conversation(db, session_uuid, conversation_id)
    if conversation is None:
        return False

    images = (
        await db.execute(
            select(Image)
            .join(Message, Message.image_id == Image.id)
            .where(
                Message.conversation_id == conversation.id,
                Image.session_id == session_uuid,
            )
            .distinct()
        )
    ).scalars().all()

    await db.delete(conversation)
    await db.flush()

    storage_keys: list[str] = []
    for image in images:
        still_referenced = (
            await db.execute(
                select(Message.id).where(Message.image_id == image.id).limit(1)
            )
        ).scalar_one_or_none()
        if still_referenced is None:
            storage_keys.append(image.storage_key)
            await db.delete(image)

    await db.commit()

    for storage_key in storage_keys:
        try:
            storage.delete_image(storage_key)
        except Exception as exc:
            logger.warning(
                "Could not delete image file %s after conversation delete (%s).",
                storage_key,
                type(exc).__name__,
            )

    return True


async def create_image(
    db: AsyncSession,
    session_id: uuid.UUID | str,
    storage_key: str,
    original_filename: str | None,
    mime_type: str,
    byte_size: int,
    sha256: str,
) -> Image:
    session_uuid = _as_uuid(session_id)
    if await db.get(AnonymousSession, session_uuid) is None:
        raise ValueError("Session not found.")
    if not storage_key.startswith(f"sessions/{session_uuid}/"):
        raise ValueError("Image does not belong to this session.")

    image = Image(
        session_id=session_uuid,
        storage_key=storage_key,
        original_filename=original_filename,
        mime_type=mime_type,
        byte_size=byte_size,
        sha256=sha256,
    )
    db.add(image)
    await db.commit()
    await db.refresh(image)
    return image


async def get_image(
    db: AsyncSession, session_id: uuid.UUID | str, image_id: uuid.UUID | str
) -> Image | None:
    """Return one image owned by the session, or None for missing or foreign."""
    result = await db.execute(
        select(Image).where(
            Image.id == _as_uuid(image_id),
            Image.session_id == _as_uuid(session_id),
        )
    )
    return result.scalar_one_or_none()


async def delete_image(
    db: AsyncSession, session_id: uuid.UUID | str, image_id: uuid.UUID | str
) -> bool:
    """Delete one image row, refusing an image owned by another session."""
    result = await db.execute(
        select(Image).where(
            Image.id == _as_uuid(image_id),
            Image.session_id == _as_uuid(session_id),
        )
    )
    image = result.scalar_one_or_none()
    if image is None:
        return False

    await db.delete(image)
    await db.commit()
    return True


async def append_user_message(
    db: AsyncSession,
    session_id: uuid.UUID | str,
    conversation_id: uuid.UUID | str,
    content: str,
    image_id: uuid.UUID | str | None = None,
) -> Message | None:
    session_uuid = _as_uuid(session_id)
    conversation = await _locked_conversation(db, session_uuid, conversation_id)
    if conversation is None:
        return None

    image_uuid = _as_uuid(image_id) if image_id is not None else None
    if image_uuid is not None:
        owned_image = (
            await db.execute(
                select(Image.id).where(
                    Image.id == image_uuid,
                    Image.session_id == session_uuid,
                )
            )
        ).scalar_one_or_none()
        if owned_image is None:
            raise ValueError("Image does not belong to this session.")

    message = Message(
        conversation_id=conversation.id,
        sequence=await _next_sequence(db, conversation.id),
        role="user",
        content=content,
        image_id=image_uuid,
    )
    db.add(message)
    conversation.updated_at = datetime.now(timezone.utc)
    await db.commit()
    return message


async def append_assistant_message(
    db: AsyncSession,
    session_id: uuid.UUID | str,
    conversation_id: uuid.UUID | str,
    content: str,
    mode: str,
    vision_assessment: str | None = None,
    grounded_answer: str | None = None,
    error_category: str | None = None,
    sources: Sequence[Mapping[str, Any]] | None = None,
    kg_edges: Sequence[Mapping[str, Any]] | None = None,
) -> Message | None:
    conversation = await _locked_conversation(db, session_id, conversation_id)
    if conversation is None:
        return None

    message = Message(
        conversation_id=conversation.id,
        sequence=await _next_sequence(db, conversation.id),
        role="assistant",
        content=content,
        mode=mode,
        vision_assessment=vision_assessment,
        grounded_answer=grounded_answer,
        error_category=error_category,
    )
    if sources:
        message.sources = [_to_message_source(source) for source in sources]
    if kg_edges:
        message.kg_edges = [_to_message_kg_edge(edge) for edge in kg_edges]

    db.add(message)
    conversation.updated_at = datetime.now(timezone.utc)
    await db.commit()
    return message

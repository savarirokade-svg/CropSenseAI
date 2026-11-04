"""
Multi-turn context resolution — integration tests (Phase 2E.2).

These run against ``cropsense_test`` only (see ``conftest.py``), through the real
ASGI app and the real repository/PostgreSQL path. Retrieval and generation are
the shared deterministic fakes from ``conftest.py``, so no live API call is made.

The fake corpus contains two tomato diseases:
``Tomato Nutrient Deficiency`` and ``Tomato Waterlogging``.
"""

from __future__ import annotations

import uuid

import pytest

from conftest import count_rows, fetch_all

pytestmark = pytest.mark.asyncio


async def new_session(client) -> dict:
    response = await client.post("/sessions")
    assert response.status_code == 201, response.text
    body = response.json()
    return {"headers": {"X-Session-Token": body["token"]}}


async def chat(client, headers, message: str, conversation_id: str | None = None) -> dict:
    data = {"message": message}
    if conversation_id:
        data["conversation_id"] = conversation_id
    response = await client.post("/chat", data=data, headers=headers)
    assert response.status_code == 200, response.text
    return response.json()


async def latest_source_disease(conversation_id: str) -> str | None:
    rows = await fetch_all(
        """
        select disease_name from message_sources
        where message_id = (
            select id from messages
            where conversation_id = :cid and role = 'assistant'
            order by sequence desc limit 1
        )
        """,
        cid=uuid.UUID(conversation_id),
    )
    return rows[0]["disease_name"] if rows else None


async def all_source_diseases(conversation_id: str) -> list[str]:
    rows = await fetch_all(
        """
        select ms.disease_name
        from message_sources ms
        join messages m on m.id = ms.message_id
        where m.conversation_id = :cid
        order by m.sequence, ms.citation_number
        """,
        cid=uuid.UUID(conversation_id),
    )
    return [row["disease_name"] for row in rows]


async def start_tomato_conversation(client, session) -> str:
    first = await chat(
        client, session["headers"], "Tomato nutrient deficiency symptoms?"
    )
    assert first["mode"] == "grounded_rag"
    assert first["conversation_id"]
    return first["conversation_id"]


# ── Inherited context reaches retrieval and generation ───────────────────────


async def test_favourable_conditions_followup_resolves_context(client, session):
    conversation_id = await start_tomato_conversation(client, session)

    follow_up = await chat(
        client,
        session["headers"],
        "What conditions favour it?",
        conversation_id=conversation_id,
    )

    assert follow_up["conversation_id"] == conversation_id
    assert follow_up["mode"] != "ambiguous_crop"
    assert follow_up["mode"] in {"grounded_rag", "grounded_rag_lexical_only"}
    assert follow_up["sources"]
    assert await latest_source_disease(conversation_id) == "Tomato Nutrient Deficiency"


async def test_management_followup_stays_in_the_same_conversation(client, session):
    conversation_id = await start_tomato_conversation(client, session)

    follow_up = await chat(
        client, session["headers"], "How do I manage it?", conversation_id=conversation_id
    )

    assert follow_up["mode"] != "ambiguous_crop"
    assert follow_up["conversation_id"] == conversation_id
    assert await count_rows("conversations") == 1
    assert await count_rows("messages") == 4


async def test_symptoms_followup_after_a_disease_reply_resolves_context(client, session):
    conversation_id = await start_tomato_conversation(client, session)

    follow_up = await chat(
        client, session["headers"], "Tell me the symptoms.", conversation_id=conversation_id
    )

    assert follow_up["mode"] != "ambiguous_crop"
    assert await latest_source_disease(conversation_id) == "Tomato Nutrient Deficiency"


# ── Explicit switches ────────────────────────────────────────────────────────


async def test_explicit_disease_followup_switches_within_the_crop(client, session):
    conversation_id = await start_tomato_conversation(client, session)

    follow_up = await chat(
        client, session["headers"], "What about waterlogging?", conversation_id=conversation_id
    )

    assert follow_up["mode"] in {"grounded_rag", "grounded_rag_lexical_only"}
    assert await latest_source_disease(conversation_id) == "Tomato Waterlogging"


async def test_explicit_crop_switch_is_not_overridden_by_context(client, session):
    conversation_id = await start_tomato_conversation(client, session)

    follow_up = await chat(
        client, session["headers"], "What are the symptoms on potato?", conversation_id=conversation_id
    )

    # The fake index only covers tomato, so an explicit potato question is
    # declined as an unsupported crop rather than answered with tomato context.
    assert follow_up["mode"] == "unsupported_crop"
    assert follow_up["sources"] == []
    # The earlier grounded turn is untouched; the declined turn adds no source.
    assert await all_source_diseases(conversation_id) == ["Tomato Nutrient Deficiency"]


# ── Isolation and stateless behaviour ────────────────────────────────────────


async def test_another_session_cannot_use_the_conversation(client, session):
    conversation_id = await start_tomato_conversation(client, session)
    other = await new_session(client)

    forbidden = await client.post(
        "/chat",
        data={"message": "What conditions favour it?", "conversation_id": conversation_id},
        headers=other["headers"],
    )
    assert forbidden.status_code == 404

    # An empty-history follow-up in the other session must not inherit anything.
    fresh = await chat(client, other["headers"], "What conditions favour it?")
    assert fresh["mode"] == "ambiguous_crop"


async def test_stateless_followup_does_not_inherit_history(client):
    response = await client.post("/chat", data={"message": "What conditions favour it?"})

    assert response.status_code == 200
    assert response.json()["mode"] == "ambiguous_crop"
    assert await count_rows("conversations") == 0
    assert await count_rows("messages") == 0

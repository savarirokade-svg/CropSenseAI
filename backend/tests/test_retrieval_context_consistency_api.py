"""
Retrieval / knowledge-graph context consistency — integration tests (Phase 2E.3).

Runs against ``cropsense_test`` through the real ASGI app and repository; the
Gemini pipeline is the shared deterministic fake from ``conftest.py``.
"""

from __future__ import annotations

import uuid

import pytest

from conftest import count_rows, fetch_all

pytestmark = pytest.mark.asyncio


async def new_session(client) -> dict:
    response = await client.post("/sessions")
    assert response.status_code == 201, response.text
    return {"headers": {"X-Session-Token": response.json()["token"]}}


async def chat(client, headers, message: str, conversation_id: str | None = None) -> dict:
    data = {"message": message}
    if conversation_id:
        data["conversation_id"] = conversation_id
    response = await client.post("/chat", data=data, headers=headers)
    assert response.status_code == 200, response.text
    return response.json()


async def source_diseases(conversation_id: str) -> list[str]:
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


async def latest_assistant_source_count(conversation_id: str) -> int:
    rows = await fetch_all(
        """
        select count(ms.id) as n
        from messages m
        left join message_sources ms on ms.message_id = m.id
        where m.id = (
            select id from messages
            where conversation_id = :cid and role = 'assistant'
            order by sequence desc limit 1
        )
        """,
        cid=uuid.UUID(conversation_id),
    )
    return int(rows[0]["n"])


# ── Knowledge graph receives resolved context ────────────────────────────────


async def test_followup_knowledge_graph_uses_resolved_context(client, session):
    first = await chat(client, session["headers"], "Why are my tomato leaves turning yellow?")
    assert first["knowledge_graph"]
    conversation_id = first["conversation_id"]

    follow_up = await chat(
        client, session["headers"], "Are the leaves still yellow?", conversation_id=conversation_id
    )

    assert follow_up["mode"] != "ambiguous_crop"
    assert follow_up["knowledge_graph"], "resolved crop must reach the knowledge graph"
    assert any(edge["object"] == "Yellow leaves" for edge in follow_up["knowledge_graph"])


async def test_general_crop_question_keeps_graph_and_sources(client, session):
    body = await chat(client, session["headers"], "Why are my tomato leaves turning yellow?")

    assert body["mode"] in {"grounded_rag", "grounded_rag_lexical_only"}
    assert body["sources"]
    assert body["knowledge_graph"]


# ── Unsupported disease: no sources, no unrelated graph edges ────────────────


async def test_unknown_disease_returns_no_sources_or_graph_edges(client, session):
    first = await chat(client, session["headers"], "Why are my tomato leaves turning yellow?")
    conversation_id = first["conversation_id"]

    declined = await chat(
        client,
        session["headers"],
        "Tomato bacterial wilt with yellow leaves",
        conversation_id=conversation_id,
    )

    assert declined["mode"] == "unsupported_disease"
    assert declined["sources"] == []
    assert declined["knowledge_graph"] == []
    assert await latest_assistant_source_count(conversation_id) == 0


async def test_unknown_disease_is_declined_without_a_conversation(client):
    body = await chat(client, {}, "Tomato bacterial wilt symptoms?")

    assert body["mode"] == "unsupported_disease"
    assert body["sources"] == []
    assert body["knowledge_graph"] == []
    assert await count_rows("conversations") == 0


# ── Isolation / stateless regressions ────────────────────────────────────────


async def test_stateless_followup_has_no_context(client):
    response = await client.post("/chat", data={"message": "Are the leaves still yellow?"})

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "ambiguous_crop"
    assert body["knowledge_graph"] == []


async def test_cross_session_conversation_is_rejected(client, session):
    first = await chat(client, session["headers"], "Why are my tomato leaves turning yellow?")
    other = await new_session(client)

    forbidden = await client.post(
        "/chat",
        data={
            "message": "Are the leaves still yellow?",
            "conversation_id": first["conversation_id"],
        },
        headers=other["headers"],
    )
    assert forbidden.status_code == 404


# ── Supported disease switching stays isolated ──────────────────────────────


async def test_supported_disease_switch_keeps_evidence_isolated(client, session):
    first = await chat(client, session["headers"], "Tomato nutrient deficiency symptoms?")
    conversation_id = first["conversation_id"]

    switched = await chat(
        client, session["headers"], "What about waterlogging?", conversation_id=conversation_id
    )

    assert switched["mode"] in {"grounded_rag", "grounded_rag_lexical_only"}
    assert await source_diseases(conversation_id) == [
        "Tomato Nutrient Deficiency",
        "Tomato Waterlogging",
    ]

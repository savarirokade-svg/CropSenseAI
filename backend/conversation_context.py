"""
conversation_context.py — multi-turn reference helpers for /chat.

These helpers are pure: they read a plain snapshot of a conversation's stored
turns and never query the database, a model, or the network. They exist so a
follow-up such as "What conditions favour it?" can reuse the crop and disease
that the same conversation already established, instead of being rejected as an
ambiguous crop question.

Safety rules encoded here:

* Only context that was already validated for the same crop is reused. A
  disease is returned only when the chunk inventory backs it for that crop.
* Explicit text always wins. Callers run their normal crop/disease detection
  first and consult these helpers only for a turn that names neither.
* History is whatever the caller passes in. The endpoint supplies only the
  session-scoped conversation, so one session can never see another's context.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from crop_registry import find_crops, normalize_text
from disease_registry import DISEASE_HEAD_WORDS, detect_disease_intent

#: Words that refer back to something already discussed. A follow-up may also
#: omit the noun entirely ("What are the symptoms?"), which the topical cues
#: below cover.
_REFERENTIAL = re.compile(
    r"\b(it|its|it's|this|that|these|those|they|them|their|same|such)\b"
    r"|\b(what|how)\s+about\b|\balso\b"
    r"|the\s+(disease|infection|problem|pathogen)",
    re.IGNORECASE,
)

#: Domain terms that signal an elliptical follow-up about the current subject.
#: Deliberately domain-specific: generic interrogatives ("what", "is", "how")
#: are not cues, so unrelated small talk is not answered with the last crop.
_TOPIC_WORDS = frozenset(
    {
        "symptom",
        "symptoms",
        "sign",
        "signs",
        "condition",
        "conditions",
        "favorable",
        "favourable",
        "favor",
        "favour",
        "weather",
        "climate",
        "spread",
        "spreads",
        "spreading",
        "transmit",
        "transmitted",
        "transmission",
        "management",
        "manage",
        "managed",
        "control",
        "controlled",
        "treatment",
        "treat",
        "treated",
        "prevention",
        "prevent",
        "prevented",
        "cure",
        "cause",
        "causes",
        "caused",
        "risk",
        "risks",
        "fertilizer",
        "fertiliser",
        "irrigation",
        "yield",
        "loss",
        "losses",
        "cycle",
        "host",
        "hosts",
        "damage",
        "severity",
        "severe",
        "yellow",
        "yellowing",
        "spots",
        "lesion",
        "lesions",
    }
)

_WORD = re.compile(r"[a-z']+")


def is_followup(message: str) -> bool:
    """Whether ``message`` reads as a follow-up to an established subject."""
    text = normalize_text(message)
    if not text:
        return False
    if _REFERENTIAL.search(text):
        return True
    words = _WORD.findall(text)
    if any(word in _TOPIC_WORDS for word in words):
        return True
    return any(word in DISEASE_HEAD_WORDS for word in words)


def mentions_disease_term(message: str) -> bool:
    """Whether the text contains a word that names (part of) a disease."""
    return any(
        word in DISEASE_HEAD_WORDS for word in _WORD.findall(normalize_text(message))
    )


def crop_from_history(history: Iterable[dict[str, Any]], supported: Iterable[str]) -> str | None:
    """
    Return the crop the conversation last discussed, or ``None``.

    Only user turns are scanned, newest first. A turn that names several crops
    is ambiguous and yields ``None`` rather than guessing.
    """
    candidates = sorted(set(supported))
    for record in reversed(list(history)):
        if record.get("role") != "user":
            continue
        found = find_crops(record.get("content") or "", candidates)
        if len(found) == 1:
            return found[0]
        if len(found) > 1:
            return None
    return None


def disease_from_history(
    history: Iterable[dict[str, Any]],
    crop: str | None,
    inventory: Iterable[dict[str, Any]],
) -> str | None:
    """
    Return the disease the conversation last discussed for ``crop``, or ``None``.

    Assistant citations are preferred because they record what the guide
    actually answered with; user turns are the fallback. A candidate is only
    returned when the current inventory contains it for the same crop, so a
    stale or cross-crop value can never become evidence.
    """
    if not crop:
        return None

    crop = crop.strip().lower()
    valid = {
        str(record["disease_name"])
        for record in inventory
        if record.get("crop") == crop and record.get("disease_name")
    }
    if not valid:
        return None

    for record in reversed(list(history)):
        if record.get("role") == "assistant":
            names = set()
            for source in record.get("sources") or []:
                name = source.get("disease_name")
                source_crop = source.get("crop")
                if not isinstance(name, str) or name not in valid:
                    continue
                if source_crop is not None and str(source_crop).strip().lower() != crop:
                    continue
                names.add(name)
            # A general answer may cite several diseases. Inheriting one of them
            # would narrow the follow-up arbitrarily, so only an unambiguous
            # single-disease turn becomes context.
            if len(names) == 1:
                return names.pop()
            continue

        scope = detect_disease_intent(record.get("content") or "", crop, list(inventory))
        if scope.get("status") != "supported":
            continue
        names = [name for name in scope.get("disease_names") or [] if name in valid]
        if len(names) == 1:
            return names[0]

    return None


def history_record(message: Any) -> dict[str, Any]:
    """Flatten one stored message (and its citation disease names) into plain data."""
    sources = []
    for source in getattr(message, "sources", None) or []:
        sources.append(
            {
                "disease_name": getattr(source, "disease_name", None),
                "crop": getattr(source, "crop", None),
            }
        )
    return {
        "role": getattr(message, "role", None),
        "content": getattr(message, "content", "") or "",
        "sources": sources,
    }

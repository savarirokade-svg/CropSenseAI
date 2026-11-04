"""
disease_registry.py — Disease inventory and named-disease detection.

The inventory is derived from the validated chunk metadata (``disease_name`` and
``crop``), so it can never advertise coverage the index does not have. Matching
works on canonical names, the disease core (the canonical name without its crop
prefix) and a small alias table for short forms whose distinguishing words would
otherwise be lost.

The detector answers one question: did the user explicitly name a disease, and is
that disease available for the crop they named? It deliberately does NOT treat
generic words such as "disease", "problem" or "symptoms" as a disease name, so a
general crop question is never rejected as unsupported.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Iterable

from crop_registry import (
    CROP_ALIASES,
    SUPPORTED_CROPS,
    UNSUPPORTED_CROPS,
    normalize_text,
)

#: Extra phrases for diseases whose short form is not derivable from the
#: canonical name, or whose distinguishing words must win over a shared suffix.
#: Keyed by canonical ``disease_name`` exactly as stored in chunk metadata.
DISEASE_ALIASES: dict[str, tuple[str, ...]] = {
    "Maize Sorghum Downy Mildew": (
        "sorghum downy mildew",
        "sorghum mildew",
    ),
    "Maize Crazy Top Downy Mildew": (
        "crazy top",
        "crazy top downy mildew",
    ),
    "Wheat Yellow Rust": (
        "yellow rust",
        "stripe rust",
        "wheat rust",
    ),
    "Groundnut Rust": (
        "groundnut rust",
    ),
    "Soybean Yellow Mosaic": (
        "yellow mosaic",
    ),
}

#: Words that make up a disease head but are not, on their own, an explicit
#: disease request. Kept for documentation of the boundary; matching is driven by
#: the inventory rather than this list.
_GENERIC_DISEASE_WORDS = frozenset({"disease", "diseases", "problem", "problems"})

#: Disease-name heads. Used only when no inventory phrase matched: a phrase the
#: guide does not contain but that is shaped like a disease name is declined
#: rather than answered from unrelated crop material. This is morphology, not a
#: list of diseases, so no unsupported name has to be enumerated.
DISEASE_HEAD_WORDS: tuple[str, ...] = (
    "blight",
    "rust",
    "mildew",
    "wilt",
    "rot",
    "mosaic",
    "smut",
    "mould",
    "mold",
    "canker",
    "virus",
    "nematode",
    "nematodes",
)

#: Heads that are as often a symptom description as a disease name. They need a
#: disease qualifier before they count as a named disease.
_AMBIGUOUS_HEADS = frozenset({"spot", "spots", "curl"})

#: Qualifiers that turn an ambiguous head into a disease name ("bacterial spot").
_DISEASE_QUALIFIERS = frozenset({"bacterial", "fungal", "viral", "target"})

#: Function words and verbs that cannot be part of a disease name.
_STOPWORDS = frozenset(
    {
        "a", "an", "the", "my", "our", "your", "their", "its", "his", "her",
        "this", "that", "these", "those", "some", "any", "no", "every", "such",
        "is", "are", "was", "were", "be", "been", "being", "am",
        "has", "have", "had", "do", "does", "did", "can", "could", "should",
        "would", "will", "may", "might", "must", "shall",
        "with", "without", "on", "in", "of", "for", "from", "to", "and", "or",
        "at", "by", "about", "what", "which", "who", "whom", "whose", "how",
        "why", "when", "where", "there", "here", "it", "they", "them",
        "showing", "shows", "show", "showed", "see", "seeing", "seen",
        "notice", "noticed", "noticing", "get", "gets", "getting", "got",
        "treat", "treating", "treated", "prevent", "preventing", "prevented",
        "manage", "managing", "managed", "control", "controlling", "controlled",
        "cure", "curing", "stop", "stopping", "kill", "killing",
        "disease", "diseases", "infection", "infections", "problem", "problems",
        "issue", "issues", "symptom", "symptoms", "sign", "signs",
    }
)

#: Plant parts: a descriptor, not by itself a disease name ("leaf spots").
_PLANT_PARTS = frozenset(
    {
        "leaf", "leaves", "stem", "stems", "root", "roots", "fruit", "fruits",
        "plant", "plants", "seedling", "seedlings", "flower", "flowers",
        "pod", "pods", "tuber", "tubers", "shoot", "shoots",
    }
)

#: Colours and quantities: symptom descriptions, never disease names by
#: themselves ("dark spots", "yellow leaves").
_GENERIC_MODIFIERS = frozenset(
    {
        "yellow", "yellowing", "brown", "brownish", "black", "white", "green",
        "orange", "purple", "red", "grey", "gray", "dark", "light", "pale",
        "small", "large", "tiny", "big", "many", "few", "several", "some",
        "more", "less", "new", "old", "early", "late", "severe",
    }
)


def _crop_words() -> frozenset[str]:
    """Every word used by a supported or recognised crop name or alias."""
    names = set(SUPPORTED_CROPS) | set(UNSUPPORTED_CROPS)
    for aliases in CROP_ALIASES.values():
        names.update(alias.lower() for alias in aliases)

    words: set[str] = set()
    for name in names:
        words.update(name.lower().split())
    return frozenset(words)


_CROP_WORDS = _crop_words()

_HEAD_PATTERN = re.compile(
    r"(?<![a-z0-9])(?P<head>"
    + "|".join(
        sorted(
            set(DISEASE_HEAD_WORDS) | set(_AMBIGUOUS_HEADS),
            key=len,
            reverse=True,
        )
    )
    + r")(?![a-z0-9])",
    re.IGNORECASE,
)

_WORD_EDGES = re.compile(r"^[^a-z]+|[^a-z]+$")


def _generic_disease_phrase(text: str) -> str | None:
    """
    Return a disease-shaped phrase the inventory does not contain, or ``None``.

    Called only after every inventory phrase has failed to match. The rule is
    deliberately conservative so general crop and symptom questions survive:

    * an unambiguous head (blight, rust, rot …) needs at least one content word
      before it ("bacterial wilt", "root rot");
    * an ambiguous head (spot, curl) needs a disease qualifier ("bacterial spot");
    * bare heads, plant-part-only phrases and colour/quantity descriptions are
      not names.
    """
    for match in _HEAD_PATTERN.finditer(text):
        head = match.group("head").lower()

        # Take up to two words immediately before the head. Scanning the head
        # only (rather than a combined "pre + head" regex) keeps a nested match
        # such as "leaf curl virus" from hiding the "virus" head.
        prefix_words = text[: match.start()].split()[-2:]
        before_words = [
            _WORD_EDGES.sub("", word) for word in prefix_words
        ]
        words = [word for word in before_words if word] + [head]

        # Trim leading function words and crop names from the display phrase.
        while words and (words[0] in _STOPWORDS or words[0] in _CROP_WORDS):
            words.pop(0)

        if len(words) < 2:
            continue

        before = [
            word
            for word in words[:-1]
            if word not in _STOPWORDS and word not in _CROP_WORDS
        ]
        if not before:
            continue

        if head in _AMBIGUOUS_HEADS:
            if not any(word in _DISEASE_QUALIFIERS for word in before):
                continue
        elif all(word in _GENERIC_MODIFIERS for word in before):
            # "root rot" and "leaf blight" are names; "yellow rot" alone is a
            # description. Plant parts are allowed, pure colours/quantities are not.
            continue

        return " ".join(words)

    return None



def _core_for(disease_name: str, crop: str) -> str:
    """Return the disease name without its leading crop word."""
    prefix = f"{crop} "
    if disease_name.lower().startswith(prefix):
        return disease_name[len(prefix):].strip()
    return disease_name.strip()


@lru_cache(maxsize=None)
def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    """Whole-word, case-insensitive pattern with simple plural support."""
    words = [re.escape(word) for word in phrase.split()]
    words[-1] = words[-1] + r"(?:s|es)?"
    body = r"\s+".join(words)
    return re.compile(rf"(?<![a-z0-9]){body}(?![a-z0-9])", re.IGNORECASE)


def build_disease_inventory(chunks: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Build one record per distinct ``(disease_name, crop)`` in the chunk corpus.

    The result is deterministic (first-seen chunk order) and contains no invented
    metadata: every field comes from the chunk's own ``disease_name`` / ``crop``.
    """
    records: dict[str, dict[str, Any]] = {}

    for chunk in chunks:
        name = chunk.get("disease_name")
        crop = chunk.get("crop")

        if not isinstance(name, str) or not name.strip():
            continue
        if not isinstance(crop, str) or not crop.strip():
            continue

        name = name.strip()
        crop = crop.strip().lower()

        if name in records:
            continue

        core = _core_for(name, crop)
        phrases = {
            name.lower(),
            core.lower(),
            *(alias.lower() for alias in DISEASE_ALIASES.get(name, ())),
        }
        phrases = {phrase for phrase in phrases if phrase and phrase not in _GENERIC_DISEASE_WORDS}

        records[name] = {
            "disease_name": name,
            "crop": crop,
            "core": core,
            "phrases": tuple(
                sorted(phrases, key=lambda phrase: (-len(phrase.split()), -len(phrase), phrase))
            ),
        }

    return list(records.values())


def _phrase_hits(text: str, inventory: list[dict[str, Any]]) -> list[tuple[int, int, dict[str, Any], str]]:
    """Return ``(word_count, char_length, record, phrase)`` for every phrase hit."""
    hits: list[tuple[int, int, dict[str, Any], str]] = []

    for record in inventory:
        for phrase in record["phrases"]:
            if _phrase_pattern(phrase).search(text):
                hits.append((len(phrase.split()), len(phrase), record, phrase))

    return hits


def _best_hits(hits: list[tuple[int, int, dict[str, Any], str]]) -> list[tuple[int, int, dict[str, Any], str]]:
    """
    Keep the most specific phrase hits.

    Specificity is measured in words, so "sorghum downy mildew" beats the shared
    "downy mildew". When two distinct diseases are named with equally specific
    phrases, both are kept: the caller then retrieves for both rather than
    silently dropping one.
    """
    if not hits:
        return []
    most_words = max(hit[0] for hit in hits)
    return [hit for hit in hits if hit[0] == most_words]


def detect_disease_intent(
    message: str,
    crop: str | None,
    inventory: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Decide whether the question explicitly names a disease for ``crop``.

    Returns
    -------
    ``{"status": "supported", "disease_names": [...], "named_phrase": str, ...}``
        The named disease is in the inventory for this crop.
    ``{"status": "unsupported", "named_phrase": str, "known_crops": [...]}``
        A recognised disease phrase was named, but not for this crop. No
        retrieval should run.
    ``{"status": "none", ...}``
        No specific disease was named (a general crop question), or the phrase is
        a shared generic term covering several diseases of this crop.
    """
    text = normalize_text(message)
    crop = normalize_text(crop)
    result: dict[str, Any] = {
        "status": "none",
        "crop": crop or None,
        "disease_names": [],
        "named_phrase": None,
        "known_crops": [],
    }

    if not text or not crop or not inventory:
        return result

    hits = _phrase_hits(text, inventory)
    if not hits:
        # No guide phrase matched. A question can still be naming a disease the
        # guide does not cover; decline it instead of answering from unrelated
        # crop material. General questions have no disease-shaped phrase and
        # keep their normal behaviour.
        generic = _generic_disease_phrase(text)
        if generic:
            return {
                **result,
                "status": "unsupported",
                "named_phrase": generic,
                "known_crops": [],
            }
        return result

    same_crop = [hit for hit in hits if hit[2]["crop"] == crop]
    best_same = _best_hits(same_crop)

    if best_same:
        names = sorted({hit[2]["disease_name"] for hit in best_same})
        return {
            **result,
            "status": "supported",
            "disease_names": names,
            "named_phrase": best_same[0][3],
        }

    best_other = _best_hits([hit for hit in hits if hit[2]["crop"] != crop])
    if not best_other:
        return result

    phrase = best_other[0][3]

    # A generic head shared by several diseases of this crop (for example
    # "downy mildew" for maize) is not an unsupported disease: it is a broad
    # question the retrieval layer should answer with the available records.
    same_crop_cores = [
        record["core"].lower() for record in inventory if record["crop"] == crop
    ]
    if any(
        core == phrase or core.endswith(" " + phrase) for core in same_crop_cores
    ):
        return result

    return {
        **result,
        "status": "unsupported",
        "named_phrase": phrase,
        "known_crops": sorted({hit[2]["crop"] for hit in best_other}),
    }

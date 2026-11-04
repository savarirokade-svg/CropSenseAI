"""
crop_registry.py — single source of truth for crop names and aliases.

Crop identifiers are the lowercase canonical names written into chunk metadata
(``chunk["crop"]``) and used at the end of every knowledge-base heading
(``### 3.1 Blast of Rice``). Aliases are accepted from user questions only;
guide headings must always use the canonical spelling.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from functools import lru_cache

#: Crops the knowledge base is being expanded to cover, in guide-number order:
#: 1 tomato, 2 potato, 3 rice, 4 wheat, 5 maize, 6 cotton, 7 soybean,
#: 8 groundnut, 9 chickpea, 10 mustard, 11 onion, 12 chilli.
SUPPORTED_CROPS: tuple[str, ...] = (
    "tomato",
    "potato",
    "rice",
    "wheat",
    "maize",
    "cotton",
    "soybean",
    "groundnut",
    "chickpea",
    "mustard",
    "onion",
    "chilli",
)

#: Alternative spellings accepted in user questions, mapped to a canonical crop.
#: Simple plurals are handled by the pattern suffix, so they are not repeated.
CROP_ALIASES: dict[str, tuple[str, ...]] = {
    "rice": ("paddy",),
    "maize": ("corn",),
    "chilli": ("chili",),
    "groundnut": ("peanut",),
    "chickpea": ("chana", "chick pea"),
    "soybean": ("soya", "soy"),
    # Not currently supported, but recognised so the request is declined as
    # "unsupported" rather than being reported as an unknown crop.
    "grapes": ("grape",),
}

#: Crops the guide deliberately does not cover. Supported crops are never here.
UNSUPPORTED_CROPS: tuple[str, ...] = (
    "apple",
    "banana",
    "grapes",
    "mango",
)


def normalize_text(text: str | None) -> str:
    """Lowercase and collapse whitespace so matching is consistent."""
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def names_for(crop: str) -> tuple[str, ...]:
    """The canonical name plus every accepted alias."""
    return (crop,) + CROP_ALIASES.get(crop, ())


@lru_cache(maxsize=None)
def pattern_for_name(name: str) -> re.Pattern[str]:
    """Case-insensitive, word-boundary pattern with simple plural support."""
    return re.compile(rf"\b{re.escape(name)}(?:e?s)?\b", re.IGNORECASE)


def matches_crop(text: str, crop: str) -> bool:
    """Whether ``text`` names ``crop`` (canonical name or any alias)."""
    return any(pattern_for_name(name).search(text) for name in names_for(crop))


def find_crops(text: str, candidates: Iterable[str]) -> list[str]:
    """Return the candidates named in ``text``, preserving candidate order."""
    return [crop for crop in candidates if matches_crop(text, crop)]

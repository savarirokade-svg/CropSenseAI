"""
index_builder.py — FAISS index builder for CropSense AI (Task 11d, Stage 1).

Parses backend/data/agricultural_guide.md and converts the *verified* disease
knowledge base (Part 1 only) into source-attributed chunks that are suitable
for semantic retrieval with FAISS.

WHAT THIS STAGE DOES
--------------------
By default the script runs in DRY-RUN mode: it parses the knowledge base,
applies the inclusion rules, builds deterministic chunk IDs and prints a
report. It performs no network access, imports no embedding/FAISS module and
writes no files.

    python index_builder.py                # dry-run (default)

WHAT THE NEXT STAGE WILL DO
---------------------------
``--build`` is the explicit opt-in for the live run. It embeds each chunk with
``embeddings.embed_document`` and writes ``index.faiss`` / ``chunks.json`` /
``manifest.json`` into backend/data/vector_store/. It is intentionally NOT run
in this stage, so that code path has not been executed yet.

EVIDENCE RULES
--------------
* Part 1 sections are the only indexed content. Part 2 (general guidance, no
  cited source) and Part 3 (explicitly excluded, unverifiable content) are
  never indexed, so excluded material can never surface as treatment advice.
* ``evidence_level`` is read from each section's ``Source-Access-Status`` line:
  full-page reads become ``full_page``, abstract-only reads become
  ``abstract_only``. That distinction is preserved per chunk and abstract-only
  chunks keep the document's own "Abstract-Level Source Only" marker in text.
* Nothing is invented: chunk text is a verbatim slice of the Markdown body,
  and metadata values are taken from the section's own source block.
"""

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from crop_registry import SUPPORTED_CROPS

# ── Load backend/.env so GEMINI_EMBEDDING_MODEL/DIM match embeddings.py ─────────
BACKEND_DIR = Path(__file__).resolve().parent
load_dotenv(BACKEND_DIR / ".env")

# ── Paths ─────────────────────────────────────────────────────────────────────

DEFAULT_KB_PATH = BACKEND_DIR / "data" / "agricultural_guide.md"
DEFAULT_OUTPUT_DIR = BACKEND_DIR / "data" / "vector_store"
DEFAULT_SOURCE_REGISTER_PATH = (
    BACKEND_DIR.parent / "research" / "agricultural_knowledge" / "sources.md"
)

INDEX_FILENAME = "index.faiss"
CHUNKS_FILENAME = "chunks.json"
MANIFEST_FILENAME = "manifest.json"

# ── Chunking configuration ────────────────────────────────────────────────────

TARGET_CHUNK_TOKENS = 400   # aim for 300-500 tokens where practical
MAX_CHUNK_TOKENS = 500      # hard ceiling before a subsection is split
OVERLAP_BULLETS = 2         # bullets repeated at the start of a continuation chunk
RENDER_SAFETY_MARGIN = 50   # reserve tokens for rendered labels/separators/header variance

# Approximation only (no tokenizer dependency): 1 token ~= 4 characters.
CHARS_PER_TOKEN = 4

INDEX_TYPE = "IndexFlatIP"
VECTOR_DTYPE = "float32"
CHUNK_ID_LENGTH = 16        # hex chars of the SHA-256 digest used as chunk_id

# Defaults mirror backend/embeddings.py. Read from the environment here so that
# a dry run never has to import the embedding module (and therefore never has
# any reason to talk to the network). Change together with embeddings.py.
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-2"
DEFAULT_EMBEDDING_DIM = 768

# ── Markdown structure patterns ───────────────────────────────────────────────

PART_HEADING_RE = re.compile(r"^##\s+Part\s+(\d+)\b(.*)$", re.IGNORECASE)
H2_RE = re.compile(r"^##\s+")
H3_RE = re.compile(r"^###\s+(.+?)\s*$")
H4_RE = re.compile(r"^####\s+(.+?)\s*$")
METADATA_RE = re.compile(r"^\*\*(?P<label>[^*:]+?):\*\*\s*(?P<value>.*?)\s*$")
RULE_RE = re.compile(r"^\s*(-{3,}|\*{3,}|_{3,})\s*$")

# Only this part is indexed. Part 2 and Part 3 are excluded by rule.
INCLUDED_PART = 1

# Maps the guide's bold labels onto the chunk metadata field names.
SOURCE_METADATA_LABELS = {
    "source-title": "source_title",
    "source-url": "source_url",
    "source-institution": "source_institution",
    "source-access-date": "source_access_date",
    "source-access-status": "source_access_status",
}

# Source-attribution labels whose values are captured as structured metadata
# but are NOT part of the five required source fields. The guide writes these
# as bold labels immediately below the five fields. They are parsed here and
# stored on the chunk so retrieval code can expose multi-source provenance.
SOURCE_ATTRIBUTION_LABELS = {
    "source ids": "source_ids",          # e.g. "S01, S02, S46"
    "source-ids": "source_ids",          # hyphenated variant
    "additional sources": "additional_sources",
    "additional source": "additional_sources",
}

# Not source metadata: a disease fact that stays in the chunk text.
CAUSAL_ORGANISM_LABELS = {"causal organism", "causal-organism"}

# Canonical crop names, shared with the API crop guard so a crop cannot be
# parsed here but rejected there (or vice versa).
CROPS = SUPPORTED_CROPS

# ── Exceptions ────────────────────────────────────────────────────────────────

class KnowledgeBaseError(Exception):
    """Raised when the knowledge base cannot be parsed at all."""


class IndexBuildError(KnowledgeBaseError):
    """Raised when a live index build cannot be completed."""


# ── Small helpers ─────────────────────────────────────────────────────────────

def approx_tokens(text: str) -> int:
    """Approximate token count (character-based, deterministic)."""
    return max(1, round(len(text) / CHARS_PER_TOKEN))


def _clean_inline(value: str) -> str:
    """Strip Markdown emphasis from a metadata value, keeping the wording."""
    cleaned = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", value)
    cleaned = re.sub(r"`(.+?)`", r"\1", cleaned)
    return cleaned.strip()


def _normalize_url(url: str | None) -> str:
    """Normalize a source URL for stable hashing."""
    if not url:
        return ""
    return url.strip().lower().rstrip("/")


def _normalize_text(text: str) -> str:
    """Normalize whitespace for stable hashing and empty-chunk checks."""
    return re.sub(r"\s+", " ", text).strip()


def load_source_register(
    path: Path = DEFAULT_SOURCE_REGISTER_PATH,
) -> tuple[dict[str, str], set[str]]:
    """
    Read the research source register into ``(url_to_id, known_ids)``.

    The register is optional at build time: if it is absent, the guide still
    builds, but primary-ID mapping and source-ID validation are skipped. The
    file is read-only and never modified.
    """
    if not path.exists():
        return {}, set()

    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return {}, set()

    url_to_id: dict[str, str] = {}
    known_ids: set[str] = set()

    for line in text.splitlines():
        if not line.lstrip().startswith("| S"):
            continue

        columns = [column.strip() for column in line.strip().strip("|").split("|")]
        if len(columns) < 4:
            continue

        source_id = columns[0]
        url = columns[3]

        if not re.fullmatch(r"S\d{2}", source_id):
            continue

        known_ids.add(source_id)
        if url:
            url_to_id[_normalize_url(url)] = source_id

    return url_to_id, known_ids


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _chunk_id(source_url: str | None, section_number: str, disease_name: str,
              section_name: str, chunk_index: int, text: str) -> str:
    """
    Deterministic chunk ID: SHA-256 over normalized source URL, section
    identity and chunk text. Never random, never time-based.
    """
    identity = "|".join([
        _normalize_url(source_url),
        f"{section_number}:{disease_name}",
        section_name,
        str(chunk_index),
    ])
    payload = "\n".join([identity, _normalize_text(text)])
    return _sha256_hex(payload)[:CHUNK_ID_LENGTH]


def _title_case_crop(crop: str | None) -> str | None:
    return crop.capitalize() if crop else None


# ── Knowledge-base structure parsing ─────────────────────────────────────────

def read_knowledge_base(path: Path) -> str:
    """Read the Markdown knowledge base or raise KnowledgeBaseError."""
    if not path.exists():
        raise KnowledgeBaseError(f"Knowledge base not found at {path}.")

    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise KnowledgeBaseError(f"Could not read {path}: {exc}") from None


def extract_included_part(markdown: str) -> tuple[str, list[str]]:
    """
    Return the raw Markdown of Part 1 plus warnings.

    Everything else -- the document preamble, Part 2, Part 3 and the trailing
    Source Summary -- is dropped here, which is what keeps general guidance and
    unverifiable content out of the index.
    """
    warnings: list[str] = []
    lines = markdown.splitlines()

    part_start = None
    part_number = None

    for index, line in enumerate(lines):
        match = PART_HEADING_RE.match(line)
        if match:
            if int(match.group(1)) == INCLUDED_PART:
                part_start = index
                part_number = int(match.group(1))
                break

    if part_start is None:
        raise KnowledgeBaseError(
            f"Part {INCLUDED_PART} ('Verified Disease Knowledge Base') was not "
            f"found in the knowledge base. Refusing to build an empty index."
        )

    end = len(lines)
    for index in range(part_start + 1, len(lines)):
        if H2_RE.match(lines[index]):
            end = index
            break

    part_lines = lines[part_start + 1:end]

    if not any(H3_RE.match(line) for line in part_lines):
        warnings.append(
            "Part 1 contains no '###' disease sections; nothing was indexed."
        )

    return "\n".join(part_lines), warnings


def split_sections(part_markdown: str) -> list[dict]:
    """Split Part 1 into '###' sections, discarding the part preamble."""
    sections: list[dict] = []
    current: dict | None = None

    for line in part_markdown.splitlines():
        heading = H3_RE.match(line)

        if heading:
            if current is not None:
                sections.append(current)
            current = {"heading": heading.group(1), "lines": []}
            continue

        # Text before the first '###' belongs to the part banner, not a disease.
        if current is not None:
            current["lines"].append(line)

    if current is not None:
        sections.append(current)

    return sections


def parse_section_heading(heading: str) -> dict:
    """
    Derive section number, crop and the disease name(s) a '###' heading covers.

    Handles single-disease headings ("1.1 Early Blight of Tomato") and combined
    headings ("2.3 Common Scab and 2.4 Black Scurf of Potato — Abstract-Level
    Source Only"), which is how the guide records two potato diseases under one
    source.
    """
    number_match = re.match(r"^\s*(\d+(?:\.\d+)*)\s+(.*)$", heading)
    section_number = number_match.group(1) if number_match else ""
    remainder = number_match.group(2) if number_match else heading

    # Keep the qualifier (e.g. "Abstract-Level Source Only") separate.
    parts = re.split(r"\s+[—-]\s+", remainder, maxsplit=1)
    title = parts[0].strip()
    qualifier = parts[1].strip() if len(parts) > 1 else ""

    crop = None
    for candidate in CROPS:
        if re.search(rf"\b{candidate}\b", title, re.IGNORECASE):
            crop = candidate
            break

    # Disease core = the title with the trailing "of <crop>" removed.
    core = re.sub(
        rf"\s+of\s+{re.escape(crop)}\s*$", "", title, flags=re.IGNORECASE
    ) if crop else title

    # Drop any inner section numbering, then split multi-disease lists.
    core = re.sub(r"\d+(?:\.\d+)*\.?\s*", "", core).strip(" .,;")
    disease_cores = [part.strip(" .,;") for part in re.split(r"\s+and\s+", core)]
    disease_cores = [part for part in disease_cores if part]

    if not disease_cores:
        # Nothing usable in the heading: fall back to the title, never a guess.
        disease_cores = [title]

    crop_label = _title_case_crop(crop)
    disease_names = [
        f"{crop_label} {disease}" if crop_label else disease
        for disease in disease_cores
    ]

    return {
        "section_number": section_number,
        "section_title": title,
        "heading_qualifier": qualifier,
        "crop": crop,
        "disease_cores": disease_cores,
        "disease_names": disease_names,
    }


def parse_source_metadata(lines: list[str], warnings: list[str],
                          section_label: str) -> tuple[dict, list[str], list[str]]:
    """
    Pull the bold source block out of a section.

    Returns (metadata, causal_organism_lines, remaining_lines). Labels in
    SOURCE_METADATA_LABELS become the five required source fields. Labels in
    SOURCE_ATTRIBUTION_LABELS become additional structured metadata (source_ids,
    additional_sources). Anything else is left in the body rather than invented.
    Missing or malformed values are reported into ``warnings``.
    """
    metadata: dict[str, str | None] = {field: None for field in SOURCE_METADATA_LABELS.values()}
    # Initialise the attribution extras as None so they are always present on
    # the returned dict and downstream code can check them without KeyError.
    for attr_field in SOURCE_ATTRIBUTION_LABELS.values():
        metadata.setdefault(attr_field, None)

    causal_lines: list[str] = []
    remaining: list[str] = []

    def warn(message: str):
        warnings.append(f"{section_label}: {message}")

    for line in lines:
        stripped = line.strip()

        if RULE_RE.match(stripped):
            continue

        match = METADATA_RE.match(stripped)

        if match:
            raw_label = match.group("label").strip()
            # Normalise: lower-case, collapse spaces, replace underscores with hyphens.
            label = raw_label.lower().replace("_", "-").strip()
            label_spaces = raw_label.lower().strip()  # for space-separated lookup
            value = _clean_inline(match.group("value"))

            if label in SOURCE_METADATA_LABELS:
                field = SOURCE_METADATA_LABELS[label]

                if metadata[field] is not None:
                    warn(f"Duplicate '{label}' line ignored for section metadata.")
                elif not value:
                    warn(f"Empty '{label}' value in source block.")
                else:
                    metadata[field] = value
                continue

            # Check the space-normalised label against attribution labels.
            attr_field = SOURCE_ATTRIBUTION_LABELS.get(label) or \
                         SOURCE_ATTRIBUTION_LABELS.get(label_spaces)
            if attr_field:
                if value:
                    if metadata[attr_field] is not None:
                        # Append rather than drop; two lines may both say "Additional source:".
                        metadata[attr_field] += "; " + value
                    else:
                        metadata[attr_field] = value
                continue

            if label in CAUSAL_ORGANISM_LABELS:
                # A disease fact, not source metadata: keep it in the text.
                if value:
                    causal_lines.append(f"Causal organism: {value}")
                continue

            # Unknown bold label: leave it in the body rather than inventing a field.
            remaining.append(line)
            continue

        remaining.append(line)

    missing = [label for label, field in SOURCE_METADATA_LABELS.items()
               if metadata[field] is None]
    if missing:
        warn("Missing source metadata: " + ", ".join(sorted(missing)) + ".")

    return metadata, causal_lines, remaining


def evidence_level_from_status(status: str | None, heading_qualifier: str,
                               warnings: list[str]) -> str:
    """
    Classify how much of the source was actually read.

    full_page      -> the section's Source-Access-Status reports a full page or
                      full PDF read with relevant content verified
    abstract_only  -> it reports an abstract-level read, or the guide's own
                      heading qualifier says the source is abstract-level only
    unknown        -> the status is ambiguous or absent

    Recognised full-source statuses (case-insensitive):
    - "Full page read ..."              (TNAU-style pages)
    - "Full page text read ..."         (alternate phrasing)
    - "Full PDF read; relevant ... verified"   (ICAR bulletin PDFs)
    - "Full PDF read; relevant chapter verified"

    Abstract-only statuses:
    - "Abstract ..."
    - "Abstract page read ..."

    A partial read (e.g. "Full page read; partial ...") is still classified
    full_page because the full page was accessed, even though some fields were
    supplemented from other sources; the partiality is about coverage of the
    disease record's fields, not about access level.
    """
    text = (status or "").lower()

    # ── Abstract-only variants (checked first so they can never be promoted) ──
    if (
        text.startswith("abstract")
        or "abstract page read" in text
        or "abstract-level source only" in heading_qualifier.lower()
    ):
        return "abstract_only"

    # ── Full page variants ────────────────────────────────────────────────────
    if text.startswith("full page") or "full page text read" in text:
        if any(
            marker in text
            for marker in ("unverified", "not verified", "partial verification")
        ):
            warnings.append(
                "Full-page status reports unverified/partial verification; "
                "recorded as 'unknown'."
            )
            return "unknown"
        return "full_page"

    # ── Verified full-PDF variants ────────────────────────────────────────────
    if text.startswith("full pdf read") or "full pdf read" in text:
        if any(
            marker in text
            for marker in (
                "unverified",
                "not verified",
                "partial",
                "not accessed",
                "abstract",
            )
        ):
            warnings.append(
                "Full-PDF status is partial/unverified; recorded as 'unknown'."
            )
            return "unknown"
        if "verified" in text:
            return "full_page"
        warnings.append(
            "Full-PDF status does not state that relevant content was verified; "
            "recorded as 'unknown'."
        )
        return "unknown"

    warnings.append(
        "Could not determine evidence level from Source-Access-Status; "
        "recorded as 'unknown'."
    )
    return "unknown"


def split_subsections(lines: list[str]) -> tuple[list[str], list[tuple[str, list[str]]]]:
    """
    Split a section body into (preamble lines, [(subsection heading, lines)]).

    Content before the first '####' heading (for example the guide's
    "Evidence limitation" note) is returned as the preamble so it is not lost.
    """
    preamble: list[str] = []
    subsections: list[tuple[str, list[str]]] = []
    current: list[str] | None = None

    for line in lines:
        heading = H4_RE.match(line)

        if heading:
            # The heading text is carried as the subsection name, so it is not
            # repeated inside the subsection body.
            current = []
            subsections.append((heading.group(1), current))
            continue

        if RULE_RE.match(line.strip()):
            continue

        if current is None:
            preamble.append(line)
        else:
            current.append(line)

    return preamble, subsections


def assign_disease(subsection_name: str, section_info: dict,
                   warnings: list[str]) -> str:
    """
    Pick the disease a subsection belongs to.

    For a heading that lists several diseases, the subsection naming the disease
    wins; otherwise the section's single disease is used.
    """
    names = section_info["disease_names"]

    if len(names) == 1:
        return names[0]

    matched = [
        name for name, core in zip(names, section_info["disease_cores"])
        if core and core.lower() in subsection_name.lower()
    ]

    if len(matched) == 1:
        return matched[0]

    warnings.append(
        f"Subsection '{subsection_name}' does not identify exactly one disease "
        f"for section '{section_info['section_title']}'; using combined label."
    )
    return " / ".join(names)


def _rendered_overhead(disease_name: str, evidence: str,
                        causal_lines: list[str]) -> int:
    """
    Estimate the token cost of the header lines that _render_chunk_text prepends
    to every chunk (disease name + section-names line, causal organism, evidence
    marker).  This overhead must be reserved when deciding whether a new entry
    fits in the current buffer.
    """
    overhead_text = disease_name
    for line in causal_lines:
        overhead_text += "\n" + line
    if evidence == "abstract_only":
        overhead_text += "\n[Abstract-level source only: claims are limited to what the accessible source text states.]"
    return approx_tokens(overhead_text) + 15  # +15 for the section-names line allowance


# ── Chunk construction ────────────────────────────────────────────────────────

def _body_line(raw: str) -> str:
    """
    Normalize Markdown noise in one body line while keeping its wording.

    Blockquote markers are removed and emphasis/backtick markup is stripped, so
    the embedded text reads as plain prose. The bullet marker itself is kept
    unless the caller says otherwise.
    """
    line = raw.strip()

    while line.startswith(">"):
        line = line.lstrip(">").strip()

    return _clean_inline(line)


def _blocks_from_body(lines: list[str]) -> list[str]:
    """
    Break one subsection body into atomic text blocks.

    A bullet list is a single block, so bullets are never split unless the list
    on its own is already too long. Paragraphs and blockquote notes are blocks
    too. Blank lines and horizontal rules are dropped.
    """
    blocks: list[str] = []
    paragraph: list[str] = []

    def flush_paragraph():
        if paragraph:
            blocks.append("\n".join(paragraph).strip())
            paragraph.clear()

    index = 0
    while index < len(lines):
        raw = lines[index]
        stripped = _body_line(raw)

        if not stripped or RULE_RE.match(stripped):
            flush_paragraph()
            index += 1
            continue

        if re.match(r"^([-*]|\d+\.)\s", stripped):
            # Collect the whole list (bullet or numbered) as one block.
            list_lines: list[str] = []

            while index < len(lines):
                raw_candidate = lines[index]
                candidate = _body_line(raw_candidate)

                is_item = bool(re.match(r"^([-*]|\d+\.)\s", candidate))
                is_continuation = raw_candidate[:1] in (" ", "\t") and list_lines

                if not (is_item or is_continuation):
                    break

                list_lines.append(candidate)
                index += 1

            flush_paragraph()
            blocks.append("\n".join(list_lines).strip())
            continue

        paragraph.append(stripped)
        index += 1

    flush_paragraph()
    return [block for block in blocks if _normalize_text(block)]


def _split_bullet_block(block: str) -> list[str]:
    """
    Split a bullet/numbered block into individual items, keeping any indented
    continuation lines attached to their parent bullet.
    """
    items: list[str] = []
    current: list[str] = []

    for line in block.splitlines():
        if re.match(r"^\s*([-*]|\d+\.)\s", line):
            if current:
                items.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)

    if current:
        items.append("\n".join(current).strip())

    return items


def _split_paragraph_block(block: str) -> list[str]:
    """
    Split a paragraph block at sentence or paragraph boundaries.

    Sentence boundaries are matched as ./?/! followed by whitespace and an
    uppercase letter, which keeps decimal points and abbreviations intact.
    Remaining paragraph breaks are also used as safe split points.
    """
    # First, split at blank-line boundaries.
    paragraphs = [p.strip() for p in block.split("\n\n") if p.strip()]
    items: list[str] = []

    for paragraph in paragraphs:
        # Split at sentence endings, but keep the terminator with the sentence.
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", paragraph)
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                items.append(sentence)

    # If the whole block is one giant sentence, return it unchanged; the
    # caller will emit it as a single (oversized) piece rather than drop it.
    return items if items else [block.strip()]


def _split_oversized_subsection(blocks: list[str]) -> list[str]:
    """
    Split one long subsection at safe boundaries, with a small overlap.

    * Bullet/numbered blocks are split between top-level items; indented
      continuation lines stay with their parent bullet.
    * Paragraph blocks are split at sentence or paragraph boundaries.
    * Each continuation piece carries the last few bullets (or the last
      sentence) of the previous piece as overlap, so it keeps context.
    """
    items: list[str] = []

    for block in blocks:
        first_non_blank = next((line for line in block.splitlines() if line.strip()), "")

        if re.match(r"^\s*([-*]|\d+\.)\s", first_non_blank):
            items.extend(_split_bullet_block(block))
        else:
            items.extend(_split_paragraph_block(block))

    groups: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0

    for item in items:
        item_tokens = approx_tokens(item)

        if current and current_tokens + item_tokens > TARGET_CHUNK_TOKENS:
            groups.append(current)
            # Carry context forward: bullets carry the last few bullets,
            # paragraphs carry the last sentence/paragraph.
            if all(re.match(r"^\s*([-*]|\d+\.)\s", line)
                   for it in current[-1:]
                   for line in it.splitlines()[:1]):
                carry = current[-OVERLAP_BULLETS:]
            else:
                carry = current[-1:]
            current = carry + [item]
            current_tokens = sum(approx_tokens(x) for x in current)
        else:
            current.append(item)
            current_tokens += item_tokens

    if current:
        groups.append(current)

    return ["\n".join(group).strip() for group in groups if group]


def _render_chunk_text(disease_name: str, entries: list[tuple[str, list[str]]],
                       evidence: str, causal_lines: list[str]) -> str:
    """
    Assemble the final chunk text: identity line, context, then body blocks.

    Each subsection keeps its own label, so a chunk states which facts are
    symptoms and which are management. Text that repeats verbatim across
    subsections -- the guide's "Directly from source page:" provenance line --
    is emitted once per chunk instead of once per subsection.
    """
    section_names = [label for label, _ in entries if label]

    header_lines: list[str] = [f"{disease_name} — {', '.join(section_names)}"]

    if causal_lines:
        header_lines.extend(causal_lines)

    if evidence == "abstract_only":
        # Keep the evidence limitation visible inside the embedded text too, so
        # an abstract-only chunk can never read like a full-page fact.
        header_lines.append(
            "[Abstract-level source only: claims are limited to what the "
            "accessible source text states.]"
        )

    seen: set[str] = set()
    parts: list[str] = []

    # The guide repeats "Directly from source page:" before each tomato
    # subsection; emit it once per chunk. Actual disease content is never
    # de-duplicated, even if the same wording appears in multiple subsections.
    BOILERPLATE = {"Directly from source page:"}

    for label, blocks in entries:
        # De-duplicate boilerplate that repeats verbatim across subsections.
        kept: list[str] = []

        for block in blocks:
            normalized = _normalize_text(block)

            if not normalized:
                continue

            if normalized in BOILERPLATE:
                if normalized not in seen:
                    seen.add(normalized)
                    kept.append(block)
                continue

            kept.append(block)

        if not kept:
            continue

        if label:
            parts.append(f"{label}:\n\n" + "\n\n".join(kept))
        else:
            parts.append("\n\n".join(kept))

    return "\n\n".join(["\n".join(header_lines), *parts]).strip()


EVIDENCE_LIMITATIONS_LABEL = "Evidence limitations and safety warnings"
CHEMICAL_MANAGEMENT_LABEL = "Chemical management"


def _split_blocks_into_units(blocks: list[str]) -> list[str]:
    """Break blocks into atomic bullet items or sentences for repacking."""
    units: list[str] = []

    for block in blocks:
        first_non_blank = next(
            (line for line in block.splitlines() if line.strip()), ""
        )

        if re.match(r"^\s*([-*]|\d+\.)\s", first_non_blank):
            units.extend(_split_bullet_block(block))
        else:
            units.extend(_split_paragraph_block(block))

    return [unit for unit in units if _normalize_text(unit)]


def _split_unit_text_until_fits(
    unit: str,
    label: str,
    evidence_blocks: list[str] | None,
    render,
    max_tokens: int,
) -> list[str]:
    """
    Split one text unit by words/characters until it renders within the budget.

    This is the last-resort splitter: it never drops content. Bullet markers are
    preserved on each resulting piece where present.
    """
    if approx_tokens(render([(label, [unit], evidence_blocks)])) <= max_tokens:
        return [unit]

    stripped = unit.lstrip()
    prefix = ""
    body = unit

    if stripped.startswith("- "):
        prefix = "- "
        body = stripped[2:]
    elif stripped.startswith("* "):
        prefix = "* "
        body = stripped[2:]

    words = body.split()

    if len(words) > 1:
        midpoint = len(words) // 2
        parts = [
            " ".join(words[:midpoint]),
            " ".join(words[midpoint:]),
        ]
    else:
        if len(body) <= 1:
            # A single character cannot be split further. Return it so the
            # caller can report the remaining limit rather than lose content.
            return [unit]
        midpoint = max(1, len(body) // 2)
        parts = [body[:midpoint], body[midpoint:]]

    pieces: list[str] = []

    for part in parts:
        if not part:
            continue
        piece = f"{prefix}{part}" if prefix else part
        pieces.extend(
            _split_unit_text_until_fits(
                piece, label, evidence_blocks, render, max_tokens
            )
        )

    return pieces or [unit]


def _pack_units(units, render, max_tokens: int) -> list[list[tuple]]:
    """
    Pack units sequentially into groups whose rendered text stays within
    ``max_tokens``.

    A unit is ``(label, blocks, evidence_blocks)``. When ``evidence_blocks`` is
    present, the unit is a chemical-management section paired with its evidence
    limitations; the warning is repeated in every group produced from it.
    """
    if not units:
        return []

    if approx_tokens(render(units)) <= max_tokens:
        return [units]

    groups: list[list[tuple]] = []
    current: list[tuple] = []

    for unit in units:
        candidate = current + [unit]

        if approx_tokens(render(candidate)) <= max_tokens:
            current = candidate
            continue

        if current:
            groups.append(current)
            current = []

        if approx_tokens(render([unit])) <= max_tokens:
            current = [unit]
            continue

        label, blocks, evidence_blocks = unit
        sub_units = _split_blocks_into_units(blocks)

        if not sub_units:
            groups.append([unit])
            continue

        sub_groups: list[list[str]] = []
        sub_current: list[str] = []

        for sub_unit in sub_units:
            sub_candidate = sub_current + [sub_unit]

            if approx_tokens(
                render([(label, sub_candidate, evidence_blocks)])
            ) <= max_tokens:
                sub_current = sub_candidate
                continue

            if sub_current:
                sub_groups.append(sub_current)
                sub_current = []

            if approx_tokens(
                render([(label, [sub_unit], evidence_blocks)])
            ) <= max_tokens:
                sub_current = [sub_unit]
                continue

            for piece in _split_unit_text_until_fits(
                sub_unit, label, evidence_blocks, render, max_tokens
            ):
                sub_groups.append([piece])

        if sub_current:
            sub_groups.append(sub_current)

        for index, sub_group in enumerate(sub_groups):
            if index == 0:
                group_label = label
            elif label:
                group_label = f"{label} (continued)"
            else:
                group_label = "Context (continued)"

            groups.append([(group_label, sub_group, evidence_blocks)])

    if current:
        groups.append(current)

    return groups


def _render_units(
    disease_name: str,
    units: list[tuple],
    evidence: str,
    causal_lines: list[str],
) -> str:
    """Render packed units, repeating a paired evidence block in each group."""
    entries: list[tuple[str, list[str]]] = []

    for label, blocks, evidence_blocks in units:
        entries.append((label, blocks))
        if evidence_blocks:
            entries.append((EVIDENCE_LIMITATIONS_LABEL, evidence_blocks))

    return _render_chunk_text(disease_name, entries, evidence, causal_lines)


def _unit_section_names(units: list[tuple]) -> list[str]:
    """Section labels represented by a packed candidate chunk."""
    names: list[str] = []

    for label, _blocks, evidence_blocks in units:
        if label and label not in names:
            names.append(label)
        if evidence_blocks and EVIDENCE_LIMITATIONS_LABEL not in names:
            names.append(EVIDENCE_LIMITATIONS_LABEL)

    return names


def build_chunks(markdown: str, kb_name: str) -> tuple[list[dict], list[str]]:
    """
    Convert the knowledge base into indexed chunks.

    Returns (chunks, warnings). Deterministic: the same input always produces
    the same chunk IDs.
    """
    part_markdown, warnings = extract_included_part(markdown)
    chunks: list[dict] = []
    source_register, register_ids = load_source_register()

    for section in split_sections(part_markdown):
        section_info = parse_section_heading(section["heading"])

        metadata, causal_lines, body_lines = parse_source_metadata(
            section["lines"], warnings, f"Section '{section['heading']}'"
        )
        evidence = evidence_level_from_status(
            metadata["source_access_status"], section_info["heading_qualifier"], warnings
        )

        if not section_info["crop"]:
            warnings.append(
                f"Could not identify a crop for section '{section['heading']}'."
            )

        preamble, subsections = split_subsections(body_lines)

        if not subsections:
            warnings.append(
                f"Section '{section['heading']}' has no '####' subsections; "
                "indexing its body as a single chunk."
            )
            subsections = [("", preamble)]
            preamble = []

        # Group subsections per disease so two diseases never share one chunk,
        # then pack small consecutive subsections up to the token target.
        per_disease: dict[str, list[tuple[str, list[str], bool]]] = {}
        order: list[str] = []
        preamble_done: set[str] = set()

        for name, lines in subsections:
            if name:
                disease = assign_disease(name, section_info, warnings)
            elif section_info["disease_names"]:
                disease = section_info["disease_names"][0]
            else:
                disease = section_info["section_title"]

            if disease not in order:
                order.append(disease)
            per_disease.setdefault(disease, [])

            blocks = _blocks_from_body(lines)

            subsection_label = name or section_info["section_title"]

            if not blocks:
                warnings.append(
                    f"Subsection '{subsection_label}' of section "
                    f"'{section['heading']}' has no indexable text; skipped."
                )
                continue

            # The section preamble (for example the "Evidence limitation" note)
            # is indexed as its own unlabelled block, ahead of the first
            # subsection, so it is neither lost nor glued to subsection text.
            # When one section covers several diseases, every disease carries it.
            if preamble and disease not in preamble_done:
                preamble_blocks = _blocks_from_body(preamble)

                if preamble_blocks:
                    preamble_done.add(disease)
                    # The preamble carries Crop/Disease/Affected parts and the
                    # additional-source note. Give it a real label so a chunk
                    # never has an empty section_name.
                    per_disease[disease].append(("Record context", preamble_blocks, False))

            per_disease[disease].append((subsection_label, blocks, False))

        for disease in order:
            entries = per_disease[disease]
            chunk_index = 0

            # Pair each Chemical management subsection with the immediately
            # following Evidence limitations and safety warnings subsection.
            # The pair becomes one logical unit, so the warning is repeated in
            # every chunk produced when the chemical content has to be split.
            units: list[tuple] = []
            entry_index = 0

            while entry_index < len(entries):
                label, blocks, _atomic = entries[entry_index]

                has_paired_evidence = (
                    CHEMICAL_MANAGEMENT_LABEL.lower() in label.lower()
                    and entry_index + 1 < len(entries)
                    and EVIDENCE_LIMITATIONS_LABEL.lower()
                    in entries[entry_index + 1][0].lower()
                )

                if has_paired_evidence:
                    units.append((label, blocks, entries[entry_index + 1][1]))
                    entry_index += 2
                else:
                    units.append((label, blocks, None))
                    entry_index += 1

            def render(unit_group):
                return _render_units(disease, unit_group, evidence, causal_lines)

            # Keep a safety margin below the hard limit.  The repository has no
            # local Gemini tokenizer, so this uses index_builder's deterministic
            # approximation and leaves room for tokenizer variance.
            budget = MAX_CHUNK_TOKENS - RENDER_SAFETY_MARGIN
            groups = _pack_units(units, render, budget)

            for group in groups:
                text = render(group)

                if not _normalize_text(text):
                    warnings.append(f"Skipped an empty chunk for '{disease}'.")
                    continue

                rendered_tokens = approx_tokens(text)

                if rendered_tokens > MAX_CHUNK_TOKENS:
                    # This should be unreachable unless a single indivisible
                    # character is longer than the whole budget. Report rather
                    # than silently truncating or dropping content.
                    warnings.append(
                        f"Chunk {chunk_index} for '{disease}' is {rendered_tokens} "
                        f"tokens (limit {MAX_CHUNK_TOKENS}) after rendering; "
                        "content was not truncated."
                    )

                section_names = _unit_section_names(group)
                sub_warnings: list[str] = []

                chunks.append(make_chunk(
                    text=text,
                    disease_name=disease,
                    crop=section_info["crop"],
                    section_names=section_names,
                    section_number=section_info["section_number"],
                    metadata=metadata,
                    evidence=evidence,
                    kb_name=kb_name,
                    chunk_index=chunk_index,
                    source_register=source_register,
                    register_ids=register_ids,
                    warnings=sub_warnings,
                ))
                warnings.extend(sub_warnings)
                chunk_index += 1

    _check_duplicate_ids(chunks, warnings)
    return chunks, warnings


def make_chunk(*, text: str, disease_name: str, crop: str | None,
               section_names: list[str], section_number: str,
               metadata: dict, evidence: str, kb_name: str, chunk_index: int,
               warnings: list[str],
               source_register: dict[str, str] | None = None,
               register_ids: set[str] | None = None) -> dict:
    """Assemble one chunk record with its full source attribution."""
    source_url = metadata.get("source_url")

    if not source_url:
        warnings.append(
            f"Chunk for '{disease_name}' has no Source-URL; recorded as null."
        )
    if not metadata.get("source_title"):
        warnings.append(
            f"Chunk for '{disease_name}' has no Source-Title; recorded as null."
        )
    if evidence == "unknown":
        warnings.append(
            f"Chunk for '{disease_name}' has an unknown evidence level."
        )

    section_name = ", ".join(section_names)

    # ── Parse structured source IDs from the metadata field ──────────────────
    # The guide writes: **Source IDs:** S01, S02, S46
    # Split on commas and strip whitespace to produce a clean list. IDs are
    # never invented: we only split what the guide itself wrote.
    raw_ids = metadata.get("source_ids") or ""
    source_id_list: list[str] = [
        sid.strip() for sid in raw_ids.split(",") if sid.strip()
    ] if raw_ids else []

    if not source_id_list:
        warnings.append(f"Chunk for '{disease_name}' has no Source IDs.")

    if register_ids:
        unknown_ids = [sid for sid in source_id_list if sid not in register_ids]
        if unknown_ids:
            warnings.append(
                f"Chunk for '{disease_name}' lists source ID(s) not present in "
                f"sources.md: {', '.join(unknown_ids)}."
            )

    primary_source_id: str | None = None

    if len(source_id_list) == 1:
        primary_source_id = source_id_list[0]
    elif source_register and source_url:
        primary_source_id = source_register.get(_normalize_url(source_url))

    if primary_source_id and primary_source_id not in source_id_list:
        warnings.append(
            f"Chunk for '{disease_name}' primary source ID {primary_source_id} "
            "is not listed in its Source IDs line."
        )
        primary_source_id = None

    return {
        "chunk_id": _chunk_id(source_url, section_number, disease_name,
                              section_name, chunk_index, text),
        "disease_name": disease_name,
        "crop": crop,
        "source_title": metadata.get("source_title"),
        "source_url": source_url,
        "source_institution": metadata.get("source_institution"),
        "source_access_date": metadata.get("source_access_date"),
        "source_access_status": metadata.get("source_access_status"),
        "evidence_level": evidence,
        "section_name": section_name,
        "text": text,
        # ── Structured source attribution (multi-source support) ──────────────
        "source_ids": source_id_list,
        # Derived from the Source-URL -> source register mapping, not from the
        # order of the guide's Source IDs line (Groundnut Rust lists S19 before
        # its S20 primary). None when the register is unavailable or the URL
        # cannot be mapped.
        "primary_source_id": primary_source_id,
        "additional_sources": metadata.get("additional_sources"),
        # ── Additional bookkeeping (not required, but useful for retrieval) ───
        "section_number": section_number,
        "subsections": list(section_names),
        "chunk_index": chunk_index,
        "approx_tokens": approx_tokens(text),
        "source_file": kb_name,
    }


REQUIRED_CHUNK_FIELDS = (
    "chunk_id",
    "disease_name",
    "crop",
    "source_title",
    "source_url",
    "source_institution",
    "source_access_date",
    "source_access_status",
    "evidence_level",
    "section_name",
    "text",
    "source_ids",
)


def _mandatory_metadata_problems(
    chunks: list[dict],
    known_source_ids: set[str] | None = None,
) -> list[str]:
    """
    Check that every chunk has the source metadata required for a safe build.

    These are the fields that must not be missing or null before the live
    ``--build`` path is allowed to run.
    """
    problems: list[str] = []
    mandatory = (
        "source_title", "source_url", "source_institution",
        "source_access_date", "source_access_status",
    )

    for chunk in chunks:
        for field in mandatory:
            if not chunk.get(field):
                problems.append(
                    f"Chunk {chunk['chunk_id']} ({chunk['disease_name']}) is missing "
                    f"mandatory '{field}'."
                )

        if not chunk.get("source_ids"):
            problems.append(
                f"Chunk {chunk['chunk_id']} ({chunk['disease_name']}) has no source IDs."
            )

        if chunk.get("evidence_level") == "unknown":
            problems.append(
                f"Chunk {chunk['chunk_id']} ({chunk['disease_name']}) has an "
                "unknown evidence level."
            )

    if known_source_ids:
        problems.extend(validate_source_ids(chunks, known_source_ids))

    return problems


def validate_source_ids(
    chunks: list[dict],
    known_source_ids: set[str],
) -> list[str]:
    """Check every chunk's source IDs against the research source register."""
    problems: list[str] = []

    for chunk in chunks:
        source_ids = chunk.get("source_ids") or []
        unknown = [source_id for source_id in source_ids if source_id not in known_source_ids]

        if unknown:
            problems.append(
                f"Chunk {chunk['chunk_id']} ({chunk['disease_name']}) lists "
                f"unknown source ID(s): {', '.join(unknown)}."
            )

        primary = chunk.get("primary_source_id")
        if primary and primary not in source_ids:
            problems.append(
                f"Chunk {chunk['chunk_id']} ({chunk['disease_name']}) primary "
                f"source ID {primary} is not in source_ids."
            )

    return problems


def _check_duplicate_ids(chunks: list[dict], warnings: list[str]) -> None:
    seen: set[str] = set()

    for chunk in chunks:
        if chunk["chunk_id"] in seen:
            warnings.append(f"Duplicate chunk_id detected: {chunk['chunk_id']}.")
        seen.add(chunk["chunk_id"])


# ── Reporting (dry run) ───────────────────────────────────────────────────────

def validate_chunks(chunks: list[dict]) -> list[str]:
    """Structural checks: required fields present, no empty text, IDs unique."""
    problems: list[str] = []

    if not chunks:
        problems.append("No chunks were produced.")
        return problems

    seen_ids: set[str] = set()

    for position, chunk in enumerate(chunks):
        missing = [field for field in REQUIRED_CHUNK_FIELDS if field not in chunk]
        if missing:
            problems.append(f"Chunk #{position} is missing fields: {', '.join(missing)}.")
            continue

        if not isinstance(chunk["source_ids"], list) or not chunk["source_ids"]:
            problems.append(
                f"Chunk {chunk['chunk_id']} has no usable source_ids list."
            )

        if not _normalize_text(chunk["text"]):
            problems.append(f"Chunk {chunk['chunk_id']} has empty text.")

        if not _normalize_text(chunk["chunk_id"]):
            problems.append(f"Chunk #{position} has an empty chunk_id.")

        if chunk["chunk_id"] in seen_ids:
            problems.append(f"Duplicate chunk_id: {chunk['chunk_id']}.")
        seen_ids.add(chunk["chunk_id"])

        if chunk["evidence_level"] not in ("full_page", "abstract_only", "unknown"):
            problems.append(
                f"Chunk {chunk['chunk_id']} has an invalid evidence_level "
                f"{chunk['evidence_level']!r}."
            )

    return problems


def preview_report(chunks: list[dict], warnings: list[str], problems: list[str],
                   kb_path: Path, out_dir: Path, model: str, dim: int,
                   preview_chars: int) -> str:
    """Build the human-readable dry-run report."""
    total_chars = sum(len(chunk["text"]) for chunk in chunks)
    total_tokens = sum(chunk["approx_tokens"] for chunk in chunks)
    diseases = sorted({chunk["disease_name"] for chunk in chunks})
    evidence_counts: dict[str, int] = {}

    for chunk in chunks:
        evidence_counts[chunk["evidence_level"]] = (
            evidence_counts.get(chunk["evidence_level"], 0) + 1
        )

    lines: list[str] = []
    add = lines.append

    add("=" * 78)
    add("CropSense AI — FAISS index builder (DRY RUN, no embeddings generated)")
    add("=" * 78)
    add(f"Knowledge base     : {kb_path}")
    add(f"Indexed part       : Part {INCLUDED_PART} only (Parts 2 and 3 excluded)")
    add(f"Embedding model    : {model} (not called)")
    add(f"Vector dimension   : {dim} (float32, L2-normalized by embeddings.py)")
    add(f"Index type         : {INDEX_TYPE} on inner product == cosine similarity")
    add(f"Output directory   : {out_dir} (not written in dry-run)")
    add("")
    add(f"Diseases indexed : {len(diseases)}")
    for disease in diseases:
        evidence = {c["evidence_level"] for c in chunks if c["disease_name"] == disease}
        count = sum(1 for c in chunks if c["disease_name"] == disease)
        add(f"  - {disease:<32} chunks: {count}  evidence: {', '.join(sorted(evidence))}")
    add("")
    add(f"Chunks generated           : {len(chunks)}")
    add(f"Total chunk text           : {total_chars:,} characters "
        f"(~{total_tokens:,} approximate tokens)")
    add(f"Average chunk size         : ~{round(total_tokens / max(len(chunks), 1))} tokens")
    add(f"Estimated embedding calls  : {len(chunks)} (one text per request)")
    add(f"Evidence levels            : {evidence_counts}")
    add("")
    add("-" * 78)
    add("CHUNK PREVIEWS")
    add("-" * 78)

    for position, chunk in enumerate(chunks, start=1):
        body = chunk["text"]
        shown = body if len(body) <= preview_chars else body[:preview_chars] + " …"

        add(f"[{position}/{len(chunks)}] chunk_id={chunk['chunk_id']}")
        add(f"    disease        : {chunk['disease_name']}  (crop: {chunk['crop']})")
        add(f"    section_name   : {chunk['section_name']}")
        add(f"    evidence_level : {chunk['evidence_level']}")
        add(f"    source_title   : {chunk['source_title']}")
        add(f"    source_url     : {chunk['source_url']}")
        add(f"    institution    : {chunk['source_institution']}")
        add(f"    access_date    : {chunk['source_access_date']}")
        add(f"    access_status  : {chunk['source_access_status']}")
        add(f"    approx_tokens  : {chunk['approx_tokens']}")
        add("    text:")
        for text_line in shown.splitlines():
            add(f"      | {text_line}")
        add("")

    if warnings:
        add("-" * 78)
        add(f"WARNINGS ({len(warnings)})")
        for warning in warnings:
            add(f"  ! {warning}")
        add("")

    if problems:
        add("-" * 78)
        add(f"VALIDATION PROBLEMS ({len(problems)})")
        for problem in problems:
            add(f"  x {problem}")
    else:
        add("-" * 78)
        add("VALIDATION: all chunks have complete metadata, unique IDs and non-empty text.")

    add("")
    add("Dry run complete. No Gemini request was made and no files were written.")
    add("Re-run with --build to generate embeddings and write the FAISS index.")

    return "\n".join(lines)


def manifest_record(chunk_count: int, model: str, dim: int, kb_path: Path) -> dict:
    """Build manifest.json content. Never includes secrets or API keys."""
    try:
        kb_hash = _sha256_hex(kb_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError):
        kb_hash = None

    return {
        "embedding_model": model,
        "embedding_dimension": dim,
        "num_chunks": chunk_count,
        "index_type": INDEX_TYPE,
        "vector_dtype": VECTOR_DTYPE,
        "similarity": "inner product on L2-normalized vectors (cosine)",
        "source_knowledge_base": kb_path.name,
        "source_knowledge_base_sha256": kb_hash,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "builder": Path(__file__).name,
        "indexed_parts": [INCLUDED_PART],
        "excluded_parts": [2, 3],
    }


def embedding_config() -> tuple[str, int]:
    """Read the embedding configuration the same way embeddings.py does."""
    model = os.getenv("GEMINI_EMBEDDING_MODEL", "").strip() or DEFAULT_EMBEDDING_MODEL
    raw_dim = os.getenv("GEMINI_EMBEDDING_DIM", "").strip()

    try:
        dim = int(raw_dim) if raw_dim else DEFAULT_EMBEDDING_DIM
    except ValueError as exc:
        raise IndexBuildError(
            f"GEMINI_EMBEDDING_DIM must be an integer, got {raw_dim!r}."
        ) from exc

    if not 128 <= dim <= 3072:
        raise IndexBuildError(
            f"GEMINI_EMBEDDING_DIM must be between 128 and 3072, got {dim}."
        )

    return model, dim


def prepare_output_dir(out_dir: Path) -> Path:
    """
    Create the artifact directory and make it self-ignoring.

    Generated indexes are derived from the knowledge base and can be rebuilt on
    demand, so they must not be committed. Neither root nor backend .gitignore
    covers backend/data/, and this stage must not edit those files, so the
    directory carries its own ignore rule instead.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    gitignore = out_dir / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text(
            "# Generated by index_builder.py — derived artifacts, never committed.\n"
            "*\n"
            "!.gitignore\n",
            encoding="utf-8",
        )

    return out_dir


def build_index(chunks: list[dict], out_dir: Path, model: str, dim: int,
                kb_path: Path) -> dict:
    """
    LIVE BUILD PATH — not exercised in Stage 1.

    Embeds every chunk through embeddings.embed_document (one text per request),
    stores the vectors in an IndexFlatIP and writes index.faiss, chunks.json and
    manifest.json. Vector position i always corresponds to chunks[i].
    """
    if not chunks:
        raise IndexBuildError("Refusing to build an index from zero chunks.")

    try:
        import faiss
        import numpy as np
    except ImportError as exc:
        raise IndexBuildError(
            f"FAISS build dependencies are unavailable ({exc}). "
            "Install the backend requirements first."
        ) from None

    # Imported lazily so a dry run never touches the Gemini SDK.
    from embeddings import embed_document

    vectors = []

    for chunk in chunks:
        vectors.append(embed_document(chunk["text"], title=chunk["disease_name"]))

    matrix = np.vstack(vectors).astype(np.float32)

    if matrix.shape != (len(chunks), dim):
        raise IndexBuildError(
            f"Vector matrix has shape {matrix.shape}, expected "
            f"({len(chunks)}, {dim})."
        )

    if not np.all(np.isfinite(matrix)):
        raise IndexBuildError("Some embedding vectors contain non-finite values.")

    index = faiss.IndexFlatIP(dim)
    index.add(matrix)

    if index.ntotal != len(chunks):
        raise IndexBuildError(
            f"Index holds {index.ntotal} vectors but {len(chunks)} chunks were built."
        )

    prepare_output_dir(out_dir)
    faiss.write_index(index, str(out_dir / INDEX_FILENAME))

    (out_dir / CHUNKS_FILENAME).write_text(
        json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / MANIFEST_FILENAME).write_text(
        json.dumps(manifest_record(len(chunks), model, dim, kb_path),
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "index_path": str(out_dir / INDEX_FILENAME),
        "chunks_path": str(out_dir / CHUNKS_FILENAME),
        "manifest_path": str(out_dir / MANIFEST_FILENAME),
        "num_vectors": index.ntotal,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    # The report contains em-dashes; keep them readable on Windows consoles.
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass

    parser = argparse.ArgumentParser(
        description=(
            "Build source-attributed chunks from the verified Part 1 knowledge base. "
            "Defaults to a dry run that generates no embeddings."
        )
    )
    parser.add_argument(
        "--kb", type=Path, default=DEFAULT_KB_PATH,
        help="Path to the agricultural guide Markdown file.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Where live-build artifacts are written (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--preview-chars", type=int, default=700,
        help="How many characters of each chunk to show in the dry-run preview.",
    )
    parser.add_argument(
        "--build", action="store_true",
        help="Live build: call Gemini embeddings and write the FAISS index. "
             "Not used in Stage 1.",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Print chunk metadata as JSON instead of the readable report.",
    )

    args = parser.parse_args(argv)
    model, dim = embedding_config()

    try:
        markdown = read_knowledge_base(args.kb)
    except KnowledgeBaseError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    chunks, warnings = build_chunks(markdown, args.kb.name)
    _source_register, known_source_ids = load_source_register()
    problems = validate_chunks(chunks)
    problems.extend(_mandatory_metadata_problems(chunks, known_source_ids))

    if args.json:
        print(json.dumps(chunks, indent=2, ensure_ascii=False))
    else:
        print(preview_report(chunks, warnings, problems, args.kb, args.out_dir,
                             model, dim, args.preview_chars))

    if args.build:
        # This path performs network requests; it is deliberately unrun in Stage 1.
        if problems:
            print(
                "ERROR: --build was requested but validation found problems. "
                "Fix the knowledge base or run a dry run to inspect them.",
                file=sys.stderr,
            )
            return 5
        print("Starting live index build ...")
        try:
            summary = build_index(chunks, args.out_dir, model, dim, args.kb)
        except KnowledgeBaseError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 3
        except Exception as exc:
            # embeddings.py already sanitizes its own messages; for anything
            # else only the class name is shown, never the raw SDK text.
            print(
                f"ERROR: Index build failed ({type(exc).__name__}). Check "
                "GEMINI_API_KEY in backend/.env and try again.",
                file=sys.stderr,
            )
            return 4
        print(f"Wrote {summary['num_vectors']} vectors to {summary['index_path']}")
        return 0

    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())

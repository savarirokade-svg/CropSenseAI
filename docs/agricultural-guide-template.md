# Agricultural Guide — Disease Entry Template

This document defines the one accepted format for new disease entries in
`backend/data/agricultural_guide.md`. The indexer (`backend/index_builder.py`)
parses this format directly; anything that does not match is either skipped or
rejected during the dry run.

Only **`## Part 1 — Verified Disease Knowledge Base`** is indexed. Part 2
(general guidance) and Part 3 (excluded content) are intentionally kept out of
retrieval, so new sourced disease material must go in Part 1.

## Canonical crops and numbering

Use the lowercase canonical crop name at the end of the heading. One numbering
prefix is reserved per crop:

| Prefix | Crop | Prefix | Crop |
|---|---|---|---|
| `1.x` | tomato | `7.x` | soybean |
| `2.x` | potato | `8.x` | groundnut |
| `3.x` | rice | `9.x` | chickpea |
| `4.x` | wheat | `10.x` | mustard |
| `5.x` | maize | `11.x` | onion |
| `6.x` | cotton | `12.x` | chilli |

Aliases (`corn`, `chili`, `paddy`, `peanut`, `chana`, `soya`) are accepted in
user questions and resolve to their canonical crop, but **headings must use the
canonical name**. `grape` resolves to `grapes`, which is currently unsupported
and is therefore declined, not indexed.

## Entry template

```markdown
### 3.1 Blast of Rice

**Causal organism:** *Magnaporthe oryzae*

**Source-Title:** Blast disease of rice — <institution>, Crop Protection
**Source-URL:** <the real URL of the page that was read>
**Source-Institution:** <institution name; sources cited by the page>
**Source-Access-Date:** <YYYY-MM-DD>
**Source-Access-Status:** Full page text read (HTTP 200, <bytes> bytes).

#### Symptoms

Directly from source page:
- <claim supported by the accessed text>
- <claim>

#### Survival and Spread

Directly from source page:
- **Survival:** <claim>
- **Primary spread:** <claim>
- **Secondary spread:** <claim>

#### Favourable Conditions

Directly from source page:
- <temperature, humidity, season or field-condition claim>

#### Integrated Disease Management

Directly from source page:
- <cultural, biological or chemical-class claim>
```

If only an abstract was accessible, use
`**Source-Access-Status:** Abstract page read (...)` and add an evidence
limitation block:

```markdown
> **Evidence limitation:** Only the abstract was accessible, so this section
> records only claims confirmed by the abstract.
```

## Rules enforced by the indexer

1. **Heading shape:** `### <number> <Disease> of <Crop>`.
2. **The crop name appears once**, at the end of the heading.
   `Blast of Rice` is correct; `Rice Blast of Rice` becomes `Rice Rice Blast`.
3. **Never write ` and ` in a disease title.** The parser splits the title on
   ` and ` into two separate diseases.
4. **Do not name a different crop in the title.** The parser assigns the first
   canonical crop it finds.
5. **All five source labels are mandatory and non-empty:**
   `Source-Title`, `Source-URL`, `Source-Institution`, `Source-Access-Date`,
   `Source-Access-Status`. The live build refuses to run otherwise.
6. **Evidence level comes from the status line.** It must start with `Full page`
   or `Abstract` (or contain `full page text read` / `abstract page read`).
   Anything else becomes `unknown` and blocks the build.
7. **At least one `####` subsection is required.** Use the four shown above so
   lexical ranking treats every crop consistently.
8. **Keep an entry near 250–450 approximated tokens** (1 token ≈ 4 characters).
   The hard ceiling is 500 before the section is split. Prefer a new disease
   entry over a very long one.
9. **No placeholders.** No `TODO`, `TBD`, `XXX`, invented URLs, invented
   product names, doses, or pre-harvest intervals.
10. **Do not reproduce chemical products or application rates.** Name only the
    chemical class, matching the existing guide policy.

## Validate before building

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
python index_builder.py          # dry run; must exit 0 with no problems
```

Only after the dry run is clean should `python index_builder.py --build` be run.
Building changes `data/vector_store/` and requires regenerating the integrity
fingerprints in `backend/vector_store.py`.

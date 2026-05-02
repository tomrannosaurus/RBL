# RBL Conventions

Three file classes. Content type determines which file it goes in.

## 1. Current-Truth Files

Represent our most recent understanding. **Overwrite, never append.** No history, no dates, no "previously we thought X."

- `README.md` — project description, algorithm, results, roadmap
- `TODO.md` — what is left to do
- `docs/terminology.md` — canonical definitions
- `docs/references.md` — paper corpus with IDs and verdicts
- `docs/novelty_review.md` — novelty assessment
- `rbl_optimizer_poc.py` — implementation

Rules:
- No `Status (date):` or `Updated on:` headers
- No process narrative ("we searched X", "we found Y on 2026-05-02")
- No struck-through text or "previously we thought X"
- When a verdict changes, replace it. Do not preserve the old one.

## 2. Lab Notebook — `NOTES.md`

Append-only chronological record of *how* we arrived at current understanding. `## YYYY-MM-DD` headings. Never edit or delete past entries.

Content: search logs, reasoning trails, dead ends, meeting notes, decision rationale, anything explanatory that is not itself current truth.

## 3. Changelog — `CHANGELOG.md`

One-line entries, newest on top. Format:

```
## YYYY-MM-DD
- <file>: <what changed>
```

## 4. Date-Stamped Files

Files with date prefixes/suffixes (e.g., `2026-05-02-initial-search.md`) are frozen snapshots. Never update them. If content is still relevant, reflect it in the corresponding current-truth file.

## Decision Trail

Substantive decision → current-truth file. Reasoning → `NOTES.md`. Change → `CHANGELOG.md`.

## Persistent IDs

Reference IDs in `docs/references.md` (LOOKAROUND, HGM, etc.) are stable. Never reassign.

## Agent Rules

1. About to add history/rationale/dated commentary to a current-truth file? → `NOTES.md` instead.
2. Changed a current-truth file? → Add entry in `CHANGELOG.md`.
3. No `Status (date):` or `Updated on:` in current-truth files.
4. Found history in a current-truth file? → Extract to `NOTES.md`, clean the current-truth file.
5. Date-stamped files are read-only.
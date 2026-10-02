"""What the run can PROVE it saw — shared by the layers that judge a report.

STUDENT-OWNED, like everything else under `harness/`.

`critic` (§2) and `citation_checker` (§11) ask the same two questions of
every claim:

    is this text verbatim in ONE LINE of something the agent observed?
    which observed document is that line in?

They have to answer them the same way. If `citation_checker` thinks a
claim is a quotation and `critic` does not, one layer deletes what the
other just repaired — so the answers live here, once.

Three rules everything in this file follows, because the frozen scorer
does (`arena/scorer.py`):

1. COMPARISON IS NORMALISED, THE CLAIM IS NOT. `norm` is the scorer's own
   normalisation (NFC, casefold, whitespace collapsed) and is used ONLY to
   compare. Nothing here ever writes a normalised string into a claim:
   a claim's text must stay the model's own characters.
2. A QUOTATION LIVES ON ONE LINE. Support is scoped to a single line of a
   document, so every check below is per line, never against a whole body.
3. A SOURCE IS A DOCUMENT THE RUN RETRIEVED. Citing a document the run
   never looked at scores `UNRETRIEVED` even when the document really does
   say it, so a source is only ever chosen among documents the
   observations show.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

#: The scorer does not treat anything shorter as a quotation of any
#: document (`arena.scorer.MIN_SUPPORT_CHARS`).
MIN_CLAIM_CHARS = 12

#: Where layers that watch `fetch_doc` go past record the documents the run
#: asked for. The scorer counts a fetch as retrieval whether or not the
#: content came back whole, so this is wider than "the body is in the
#: observations".
FETCHED_KEY = "fetched_doc_ids"

#: How a document id is written, in search results and everywhere else.
DOC_ID_RE = re.compile(r"doc-\d{4}")

_OPTION_MARK_RE = re.compile(r"\(([a-zA-Z])\)\s*")
#: Where one option's own wording stops: the first sentence-ending mark.
#: Whatever follows the last option ("…(c) tiếp tục hợp tác. Giải thích
#: ngắn gọn.") is the question going on, not part of the option.
_OPTION_END_RE = re.compile(r"[.;?!](?:\s|$)")
_OPTION_TAIL_RE = re.compile(r"[\s,:]*(?:\b(?:hoặc|hay)\b)?[\s,:]*$")

#: A verdict option shorter than this is not one the scorer would accept
#: either (`arena.scorer.MIN_VERDICT_PHRASE_CHARS`).
MIN_OPTION_CHARS = 8


def norm(text) -> str:
    """The scorer's comparison form. For COMPARING only — see rule 1."""
    if not isinstance(text, str):
        return ""
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


# ---------------------------------------------------------------------------
# What was observed
# ---------------------------------------------------------------------------


def raw_lines(ctx) -> list[str]:
    """Every line the agent was shown, exactly as it was shown.

    `search` returns JSON, so a line break inside a snippet arrives as the
    two characters backslash-n. Splitting there too is what stops a claim
    from being "found" straddling two lines of a document.
    """
    return ctx.observed_text.replace("\\n", "\n").splitlines()


def lines(ctx) -> list[str]:
    """`raw_lines`, normalised, blanks dropped."""
    return [line for line in (norm(raw) for raw in raw_lines(ctx)) if line]


def grounded(observed: list[str], text) -> bool:
    """Is `text` a quotation of ONE observed line?"""
    key = norm(text)
    if len(key) < MIN_CLAIM_CHARS:
        return False
    return any(key in line for line in observed)


def longest_quote(ctx, text: str) -> str:
    """The longest stretch of `text` that is verbatim in one observed line.

    Compared on the RAW characters, so the result is a substring of what
    the model wrote AND of what the document says — a legal trim, never a
    rewrite.
    """
    best = ""
    for line in raw_lines(ctx):
        if len(line) <= len(best):
            continue
        match = SequenceMatcher(None, text, line, autojunk=False).find_longest_match(
            0, len(text), 0, len(line)
        )
        candidate = text[match.a : match.a + match.size].strip()
        if len(candidate) > len(best):
            best = candidate
    return best


# ---------------------------------------------------------------------------
# Which document says it
# ---------------------------------------------------------------------------


def _doc_lines(ctx) -> list[tuple]:
    """(doc, normalised lines) for the whole corpus, computed once per run."""
    cached = ctx.state.get("_evidence_doc_lines")
    if cached is None:
        cached = [
            (doc, tuple(line for line in (norm(raw) for raw in doc.body.splitlines()) if line))
            for doc in ctx.corpus.docs
        ]
        ctx.state["_evidence_doc_lines"] = cached
    return cached


def line_of(ctx, doc_id, text) -> str | None:
    """The (normalised) line of document `doc_id` that `text` quotes."""
    if ctx.corpus is None or not isinstance(doc_id, str):
        return None
    key = norm(text)
    if not key:
        return None
    for doc, doc_lines in _doc_lines(ctx):
        if doc.doc_id == doc_id:
            return next((line for line in doc_lines if key in line), None)
    return None


def was_retrieved(ctx, doc) -> bool:
    """Do the observations show this run looked at `doc`?"""
    observed = ctx.observed_text
    if doc.doc_id in observed or doc.body in observed:
        return True
    return doc.doc_id in (ctx.state.get(FETCHED_KEY) or ())


def source(ctx, text, *, prefer=None, avoid=None) -> str | None:
    """doc_id of a RETRIEVED document with a line that `text` quotes.

    Tried in this order: `prefer` (the document the model itself cited, if
    it qualifies), a document that came back whole from a clean fetch, a
    document a search listed. `avoid` is skipped while anything else
    qualifies — that is how the two halves of a fused claim end up on two
    different documents.
    """
    if ctx.corpus is None:
        return None
    key = norm(text)
    if len(key) < MIN_CLAIM_CHARS:
        return None
    observed = ctx.observed_text
    holders = [doc for doc, doc_lines in _doc_lines(ctx) if any(key in line for line in doc_lines)]
    whole = [doc.doc_id for doc in holders if doc.body in observed]
    listed = [
        doc.doc_id for doc in holders if doc.doc_id not in whole and was_retrieved(ctx, doc)
    ]
    known = whole + listed
    if not known and len(holders) == 1 and key in norm(observed.replace("\\n", "\n")):
        # The text only ever came back from a fetch that was cut short, and
        # exactly one document in the corpus says it.
        known = [holders[0].doc_id]
    if prefer in known and prefer != avoid:
        return prefer
    for doc_id in known:
        if doc_id != avoid:
            return doc_id
    return known[0] if known else None


# ---------------------------------------------------------------------------
# Questions that ask for one of several conclusions
# ---------------------------------------------------------------------------


def verdict_options(question) -> list[str]:
    """The lettered options a question lists — "(a) …; (b) …; (c) …".

    Empty unless the question really is of that shape: at least two
    markers, starting at (a) and running in order.
    """
    if not isinstance(question, str):
        return []
    marks = list(_OPTION_MARK_RE.finditer(question))
    letters = [mark.group(1).lower() for mark in marks]
    if len(marks) < 2 or letters != [chr(ord("a") + i) for i in range(len(marks))]:
        return []
    options = []
    for index, mark in enumerate(marks):
        end = marks[index + 1].start() if index + 1 < len(marks) else len(question)
        option = _OPTION_END_RE.split(question[mark.end() : end], maxsplit=1)[0]
        option = _OPTION_TAIL_RE.sub("", option).strip()
        if len(norm(option)) < MIN_OPTION_CHARS:
            return []
        options.append(option)
    return options


def asserted_options(text, options: list[str]) -> list[str]:
    """Which of `options` this text states, in the question's order."""
    key = norm(text if isinstance(text, str) else "" if text is None else str(text))
    if not key:
        return []
    return [option for option in options if norm(option) in key]


def stated_verdicts(report: dict, options: list[str]) -> list[str]:
    """The options a report asserts, read the way the scorer reads them:
    the `verdict` field whenever it carries anything, else the answer."""
    verdict = report.get("verdict")
    if verdict is not None and str(verdict).strip():
        return asserted_options(verdict, options)
    return asserted_options(report.get("answer"), options)

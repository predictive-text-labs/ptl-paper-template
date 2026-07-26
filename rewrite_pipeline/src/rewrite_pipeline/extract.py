"""Turn a LaTeX source string into a manifest of prose-sentence ``Record``s.

Pipeline: ``scan`` (mask math/commands/envs, emit structural markers) → build
prose *containers* (body/abstract paragraphs, list items, captions, footnotes)
from the markers → segment each container into sentences with LaTeX-aware
boundary rules → assemble ``Record``s and apply the scope filter.

A "container" is a contiguous source span in which prose flows. Sentence
boundaries are detected only at *visible* (literal-prose) terminators, so
periods inside ``$...$``, citations, decimals, or ``et al.`` never split.
"""

from __future__ import annotations

import bisect
import re

from .model import (
    EXTRACTOR_VERSION,
    Manifest,
    Record,
    count_unescaped,
    make_id,
    sha256_hex,
)
from .scanner import scan as scan_text

_SPACE = frozenset(" \t\r\n\f\v~")
_TERM = frozenset(".?!")
_CLOSERS = frozenset(")]\"'” ’»".replace(" ", "")) | {"”", "’", "»"}
_OPENERS = frozenset("([") | {"`", "“", "‘", "«"}

# Classic lorem-ipsum filler tokens. The manuscript's Results section currently
# holds placeholder Latin; those "sentences" are not real prose to rewrite.
LOREM = frozenset(
    {
        "lorem",
        "ipsum",
        "dolor",
        "amet",
        "consectetur",
        "adipiscing",
        "elit",
        "eiusmod",
        "tempor",
        "incididunt",
        "labore",
        "dolore",
        "magna",
        "aliqua",
        "veniam",
        "nostrud",
        "exercitation",
        "ullamco",
        "laboris",
        "aliquip",
        "commodo",
        "consequat",
        "aute",
        "irure",
        "reprehenderit",
        "voluptate",
        "velit",
        "cillum",
        "fugiat",
        "pariatur",
        "excepteur",
        "occaecat",
        "cupidatat",
        "proident",
        "culpa",
        "officia",
        "deserunt",
        "mollit",
        "anim",
        "laborum",
        "perspiciatis",
        "voluptatem",
        "accusantium",
        "doloremque",
        "laudantium",
        "aperiam",
        "eaque",
        "inventore",
        "veritatis",
        "architecto",
        "beatae",
        "explicabo",
        "nemo",
        "ipsam",
        "voluptas",
        "aspernatur",
        "consequuntur",
        "dolores",
        "ratione",
        "nesciunt",
        "neque",
        "porro",
        "quisquam",
        "dolorem",
        "numquam",
        "quaerat",
    }
)


def _is_lorem(t: str) -> bool:
    words = re.findall(r"[A-Za-z]+", t.lower())
    if not words:
        return False
    hits = sum(1 for w in words if w in LOREM)
    return hits >= 3 or (hits >= 2 and hits / len(words) > 0.4)


def _unterminated_in_scope(kind: str, t: str) -> bool:
    """Is a span carrying no ``.``/``?``/``!`` still a complete, rewritable unit?

    Most unterminated spans are artefacts and must stay out of scope: tabular
    column specs (``{L{0.17\\textwidth} ...}``), ``\\centering``/``\\input``
    lines, and colon lead-ins ("The correct unit is:") that are welded to the
    display math following them. Two shapes are genuine prose:

    * a **list item ending in ``;``** — each ``\\item`` is its own unit, and the
      semicolon terminates it inside an enumerated series;
    * a **caption** — table and figure titles conventionally carry no full stop.

    Anything else without a terminal stays excluded as ``no_terminal``.
    """
    if kind == "list-item":
        return t.rstrip().endswith(";")
    return kind == "caption"


# Trailing tokens that abbreviate rather than end a sentence.
ABBREV = frozenset(
    {
        "al",
        "e.g",
        "i.e",
        "cf",
        "etc",
        "vs",
        "viz",
        "fig",
        "figs",
        "eq",
        "eqs",
        "no",
        "nos",
        "sec",
        "secs",
        "dr",
        "mr",
        "mrs",
        "ms",
        "prof",
        "st",
        "inc",
        "ltd",
        "co",
        "corp",
        "u.s",
        "u.k",
        "ph.d",
        "resp",
        "approx",
        "vol",
        "pp",
        "ch",
        "chs",
        "eg",
        "ie",
    }
)


def _is_space(c: str) -> bool:
    return c in _SPACE


def _starts_sentence(text: str, k: int) -> bool:
    c = text[k]
    return c.isupper() or c == "\\" or c == "$" or c.isdigit() or c in _OPENERS


def _is_abbrev(text: str, sstart: int, i: int) -> bool:
    j = i
    while j > sstart and (text[j - 1].isalpha() or text[j - 1] == "."):
        j -= 1
    raw = text[j:i].lower()
    return raw in ABBREV or raw.strip(".") in ABBREV


def _ends_terminal(text: str, s: int, e: int) -> bool:
    k = e
    while k > s and text[k - 1] in _CLOSERS:
        k -= 1
    return k > s and text[k - 1] in _TERM


def _segment(
    text: str, bvisible: bytearray, start: int, end: int
) -> list[tuple[int, int, bool]]:
    """Yield ``(s, e, has_terminal)`` sentence spans within ``[start, end)``.

    ``bvisible`` drives boundary detection (a terminator only counts where its
    char is visible prose); the emitted span is the raw substring, which may
    contain masked inline spans (math, citations)."""
    out: list[tuple[int, int, bool]] = []
    i = start
    sstart = -1
    while i < end:
        c = text[i]
        if sstart == -1:
            if _is_space(c):
                i += 1
                continue
            sstart = i
        if bvisible[i] and c in _TERM:
            j = i + 1
            while j < end and text[j] in _CLOSERS:
                j += 1
            # A boundary requires whitespace (or container end) after the closer.
            if j < end and not _is_space(text[j]):
                i += 1
                continue
            if _is_abbrev(text, sstart, i):
                i += 1
                continue
            k = j
            while k < end and _is_space(text[k]):
                k += 1
            if k >= end or _starts_sentence(text, k):
                out.append((sstart, j, True))
                sstart = -1
                i = k
                continue
            i = j
            continue
        i += 1
    if sstart != -1:
        e = end
        while e > sstart and _is_space(text[e - 1]):
            e -= 1
        if e > sstart:
            out.append((sstart, e, _ends_terminal(text, sstart, e)))
    return out


# --- container construction from markers -------------------------------------


def _doc_range(markers: list[tuple], n: int) -> tuple[int, int]:
    ds, de = 0, n
    for mk in markers:
        if mk[0] == "begin" and mk[1] == "document":
            ds = mk[3]
        elif mk[0] == "end" and mk[1] == "document":
            de = mk[2]
    return ds, de


def _appendix_off(markers: list[tuple], doc_end: int) -> int:
    offs = [mk[1] for mk in markers if mk[0] == "appendix"]
    return min(offs) if offs else doc_end


def _env_ranges(markers: list[tuple], names: frozenset[str]) -> list[tuple[int, int]]:
    stack: list[tuple[str, int]] = []
    ranges: list[tuple[int, int]] = []
    for mk in markers:
        if mk[0] == "begin":
            stack.append((mk[1], mk[2]))
        elif mk[0] == "end":
            for idx in range(len(stack) - 1, -1, -1):
                if stack[idx][0] == mk[1]:
                    name, start = stack.pop(idx)
                    if name in names:
                        ranges.append((start, mk[3]))
                    break
    return ranges


def _in_ranges(pos: int, ranges: list[tuple[int, int]]) -> bool:
    return any(s <= pos < e for s, e in ranges)


def _paragraph_breaks(text: str, markers: list[tuple]) -> list[tuple[int, int]]:
    """Positions that terminate a prose paragraph: blank lines, environment
    begins/ends, opaque environments, display math, ``\\item`` tokens, and the
    whole span of captions/sections (which become their own containers or are
    dropped). Footnotes are intentionally NOT breaks — they stay inline."""
    breaks: list[tuple[int, int]] = []
    for m in re.finditer(r"\n[ \t]*\n", text):
        breaks.append((m.start(), m.end()))
    for mk in markers:
        tag = mk[0]
        if tag in ("begin", "end") or tag == "opaque_env":
            breaks.append((mk[2], mk[3]))
        elif tag == "display" or tag == "item":
            breaks.append((mk[1], mk[2]))
        elif tag == "caption":
            breaks.append((mk[1], mk[3] + 1))
        elif tag == "section":
            breaks.append((mk[2], mk[4] + 1))
    breaks.sort()
    return breaks


def _gaps(
    breaks: list[tuple[int, int]], doc_start: int, doc_end: int
) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for bs, be in breaks:
        bs = max(bs, doc_start)
        be = min(be, doc_end)
        if bs >= be:
            continue
        if merged and bs <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], be))
        else:
            merged.append((bs, be))
    gaps: list[tuple[int, int]] = []
    cur = doc_start
    for bs, be in merged:
        if cur < bs:
            gaps.append((cur, bs))
        cur = max(cur, be)
    if cur < doc_end:
        gaps.append((cur, doc_end))
    return gaps


def _line_starts(text: str) -> list[int]:
    starts = [0]
    for m in re.finditer("\n", text):
        starts.append(m.end())
    return starts


def extract(
    text: str, *, file_path: str = "", git_commit: str | None = None
) -> Manifest:
    sr = scan_text(text)
    visible = sr.visible
    markers = sr.markers

    doc_start, doc_end = _doc_range(markers, len(text))
    appendix_off = _appendix_off(markers, doc_end)
    abstract_ranges = _env_ranges(markers, frozenset({"abstract"}))
    list_ranges = _env_ranges(
        markers, frozenset({"itemize", "enumerate", "description"})
    )
    footnote_spans = [(mk[2], mk[3]) for mk in markers if mk[0] == "footnote"]

    # Boundary-detection view for parent prose: footnote *content* masked so the
    # footnote's internal periods never split the parent sentence.
    bvisible = bytearray(visible)
    for cs, ce in footnote_spans:
        for k in range(cs, min(ce, len(bvisible))):
            bvisible[k] = 0

    breaks = _paragraph_breaks(text, markers)
    gaps = _gaps(breaks, doc_start, doc_end)

    def kind_at(pos: int) -> str:
        if _in_ranges(pos, abstract_ranges):
            return "abstract"
        if _in_ranges(pos, list_ranges):
            return "list-item"
        return "body"

    def has_visible(s: int, e: int) -> bool:
        return any(visible[k] for k in range(s, e))

    spans: list[tuple[int, int, bool, str]] = []
    for gs, ge in gaps:
        kind = kind_at(gs)
        for s, e, ht in _segment(text, bvisible, gs, ge):
            if has_visible(s, e):
                spans.append((s, e, ht, kind))
    for mk in markers:
        if mk[0] == "caption":
            cs, ce = mk[2], mk[3]
            for s, e, ht in _segment(text, visible, cs, ce):
                if has_visible(s, e):
                    spans.append((s, e, ht, "caption"))
        elif mk[0] == "footnote":
            cs, ce = mk[2], mk[3]
            for s, e, ht in _segment(text, visible, cs, ce):
                if has_visible(s, e):
                    spans.append((s, e, ht, "footnote"))
    spans.sort(key=lambda x: x[0])

    line_starts = _line_starts(text)
    records: list[Record] = []
    for ordinal, (s, e, ht, kind) in enumerate(spans):
        t = text[s:e]
        n_dollars = count_unescaped(t, "$")
        n_brace = count_unescaped(t, "{") - count_unescaped(t, "}")
        contains_footnote = kind != "footnote" and any(
            s < fe and fs < e for fs, fe in footnote_spans
        )
        line_no = bisect.bisect_right(line_starts, s)
        reason: str | None = None
        if s >= appendix_off:
            reason = "appendix"
        elif kind not in ("abstract", "body", "caption", "footnote", "list-item"):
            reason = "out_of_kind"
        elif not ht and not _unterminated_in_scope(kind, t):
            reason = "no_terminal"
        elif contains_footnote:
            reason = "contains_footnote"
        elif n_dollars % 2 != 0:
            reason = "odd_dollars"
        elif n_brace != 0:
            reason = "brace_imbalance"
        elif _is_lorem(t):
            reason = "placeholder_lorem"
        records.append(
            Record(
                id=make_id(ordinal, t),
                text=t,
                abs_start=s,
                abs_end=e,
                line_no=line_no,
                kind=kind,
                in_scope=reason is None,
                n_dollars=n_dollars,
                n_brace_delta=n_brace,
                has_terminal=ht,
                contains_footnote=contains_footnote,
                excluded_reason=reason,
            )
        )

    return Manifest(
        file_path=file_path,
        file_sha256=sha256_hex(text),
        git_commit=git_commit,
        extractor_version=EXTRACTOR_VERSION,
        records=records,
    )

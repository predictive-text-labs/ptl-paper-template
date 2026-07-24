"""Splice accepted rewrites back into the LaTeX source, verification-first.

Safety rules:
  * The manifest's ``file_sha256`` must match the current file (checked by the
    caller) — offsets are only valid against the exact bytes they were cut from.
  * Each edit's original span must still equal ``record.text`` exactly, or it is
    skipped and logged (never fuzzy-matched).
  * Each rewrite is validated (single line, balanced ``$``/braces, no block
    structures, terminal punctuation, and every immutable LaTeX token — math
    spans, ``\\ref``/``\\citep``/…, escaped literals — preserved) before it is
    allowed in.
  * Edits are applied in DESCENDING offset order so earlier offsets stay valid.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .model import Manifest, Record

_TERM = frozenset(".?!")
_CLOSERS = frozenset(")]\"'") | {"”", "’", "»"}
_FORBIDDEN = ("\n", "\\[", "\\]", "\\begin", "\\end{", "\\item", "%")

# CLASS-A immutable tokens: their multiset must be identical between the original
# and the rewrite, else a citation / reference / math span was silently dropped
# or mangled. Whitespace inside a token is normalised, so reflowing $D=M+L$ into
# $D = M + L$ still passes (render-identical), but changing the math fails.
_CLASS_A = re.compile(
    r"\$[^$]*\$"
    r"|\\\[[\s\S]*?\\\]"
    r"|\\\([\s\S]*?\\\)"
    r"|\\(?:ref|eqref|pageref|autoref|cref|Cref|citep|citet|cite|citealp|citealt"
    r"|citeauthor|citeyear|label|url|href|S)\b(?:\{[^}]*\}|\[[^\]]*\])*"
    r"|\\[%$&_#]"
)


def _class_a_tokens(s: str) -> list[str]:
    return sorted(re.sub(r"\s+", "", t) for t in _CLASS_A.findall(s))


@dataclass
class Applied:
    id: str
    original: str
    rewrite: str
    abs_start: int
    abs_end: int


@dataclass
class Skipped:
    id: str
    reason: str


def _count_unescaped(t: str, ch: str) -> int:
    cnt = 0
    k = 0
    n = len(t)
    while k < n:
        if t[k] == "\\":
            k += 2
            continue
        if t[k] == ch:
            cnt += 1
        k += 1
    return cnt


def _ends_terminal(s: str) -> bool:
    k = len(s)
    while k > 0 and s[k - 1] in _CLOSERS:
        k -= 1
    return k > 0 and s[k - 1] in _TERM


def _trailing_terminal(s: str) -> str:
    """Return the trailing terminal+closer run of ``s`` (e.g. ``.`` or ``?''``)."""
    k = len(s)
    while k > 0 and s[k - 1] in _CLOSERS:
        k -= 1
    if k > 0 and s[k - 1] in _TERM:
        start = k - 1
        return s[start:]
    return ""


def _norm(s: str) -> str:
    return " ".join(s.split())


def validate_rewrite(
    original: str, rewrite: str, rec: Record
) -> tuple[str | None, str | None]:
    """Return (clean_rewrite, None) if acceptable, else (None, reason)."""
    rw = rewrite.strip()
    if not rw:
        return None, "empty"
    for bad in _FORBIDDEN:
        if bad in rw:
            return None, f"forbidden:{bad!r}"
    if _count_unescaped(rw, "$") != rec.n_dollars:
        return None, "dollar_count_mismatch"
    if _count_unescaped(rw, "{") - _count_unescaped(rw, "}") != 0:
        return None, "brace_imbalance"
    if _class_a_tokens(rw) != _class_a_tokens(original):
        return None, "latex_token_drift"
    if not _ends_terminal(rw):
        term = _trailing_terminal(original)
        if term:
            rw = rw + term
    if not _ends_terminal(rw):
        return None, "no_terminal"
    if _norm(rw) == _norm(original):
        return None, "noop"
    return rw, None


def apply_rewrites(
    text: str, manifest: Manifest, accepted: list[dict]
) -> tuple[str, list[Applied], list[Skipped]]:
    """Apply accepted ``[{id, rewrite}]`` edits. Returns (new_text, applied, skipped)."""
    by_id = manifest.by_id()
    edits: list[
        tuple[int, int, str, str, str]
    ] = []  # (start, end, rewrite, id, original)
    skipped: list[Skipped] = []

    for item in accepted:
        rid = item.get("id")
        rw_in = item.get("rewrite") or item.get("final_rewrite")
        if not rid or rw_in is None:
            skipped.append(Skipped(str(rid), "missing_id_or_rewrite"))
            continue
        rec = by_id.get(rid)
        if rec is None:
            skipped.append(Skipped(rid, "unknown_id"))
            continue
        if text[rec.abs_start : rec.abs_end] != rec.text:
            skipped.append(Skipped(rid, "offset_mismatch"))
            continue
        rw, err = validate_rewrite(rec.text, rw_in, rec)
        if err or rw is None:
            skipped.append(Skipped(rid, err or "invalid"))
            continue
        edits.append((rec.abs_start, rec.abs_end, rw, rid, rec.text))

    # Descending offset so earlier splices don't shift later (higher) offsets.
    edits.sort(key=lambda e: e[0], reverse=True)
    new = text
    applied: list[Applied] = []
    for s, e, rw, rid, orig in edits:
        new = new[:s] + rw + new[e:]
        applied.append(
            Applied(id=rid, original=orig, rewrite=rw, abs_start=s, abs_end=e)
        )
    applied.sort(key=lambda a: a.abs_start)
    return new, applied, skipped

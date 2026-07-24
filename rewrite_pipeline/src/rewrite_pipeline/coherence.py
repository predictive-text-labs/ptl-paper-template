"""Cross-sentence coherence gate: paragraph pairs + verbatim fix application.

Per-sentence judging is structurally blind to seam damage: a rewrite that is
faithful on its own can orphan a pro-verb in the NEIGHBOURING sentence
("sampling frequency does not alter the result … X, Y, Z still do" -> the
rewrite says "irrelevant" and "still do" loses its verb), strand a
demonstrative, drop an enumeration's labels while a later sentence still cites
"(i)--(iv)", or break a "However" contrast. This module supports the pre-apply
sweep that catches those breaks:

  * ``build_pairs`` diffs the original .tex against the would-be rewritten
    text. Rewrites contain no newlines and never touch blank lines
    (``validate_rewrite``), so paragraph blocks pair 1:1 by index — though a
    hard-wrapped original sentence may collapse onto one line, so LINE counts
    can differ.
  * ``apply_fixes`` applies the sweep's verbatim substring fixes with the same
    safety posture as ``reinsert``: the quote must match the text exactly once,
    no newlines, and the Class-A LaTeX token multiset must be preserved.
  * ``accepted_fingerprint`` ties pair files and fixes to the exact accepted
    set, so a stale sign-off can never gate a different apply.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from .reinsert import _class_a_tokens

_PARA_SPLIT = re.compile(r"\n[ \t]*\n")


def accepted_fingerprint(accepted: list[dict]) -> str:
    """Stable hash of the accepted-rewrites list (order-sensitive by design)."""
    canon = json.dumps(accepted, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def build_pairs(old_text: str, new_text: str) -> list[dict]:
    """Return ``[{old, new}]`` for every changed paragraph block.

    Relies on the reinsert invariants that rewrites contain no newlines and
    spans never cross a blank line, so both texts hold the same sequence of
    paragraph blocks and changed blocks pair 1:1 by index. Line counts may
    differ (a hard-wrapped original sentence collapses onto one line), which
    is why pairing is by paragraph, not by line.
    """
    old_paras = _PARA_SPLIT.split(old_text)
    new_paras = _PARA_SPLIT.split(new_text)
    if len(old_paras) != len(new_paras):
        raise ValueError(
            f"paragraph count changed ({len(old_paras)} -> {len(new_paras)}); "
            "a rewrite broke the no-newline invariant"
        )
    return [{"old": o, "new": n} for o, n in zip(old_paras, new_paras) if o != n]


@dataclass
class FixApplied:
    quote: str
    replacement: str


@dataclass
class FixSkipped:
    quote: str
    reason: str


def load_fixes(path: Path) -> tuple[list[dict], str | None]:
    """Read a coherence-fixes file; returns (fixes, accepted_sha256 or None).

    Accepts either the coherence-sweep workflow's full return value
    (``{fixes: [...], accepted_sha256: ...}``) saved verbatim, or a bare list.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data, None
    return data.get("fixes", []), data.get("accepted_sha256")


def apply_fixes(
    text: str, fixes: list[dict]
) -> tuple[str, list[FixApplied], list[FixSkipped]]:
    """Apply verbatim substring fixes. Returns (new_text, applied, skipped).

    Each fix ``{quote, replacement}`` is applied only if the quote occurs
    exactly once in the current text (never fuzzy-matched, never ambiguous),
    introduces no newline, and preserves the Class-A LaTeX token multiset.
    """
    applied: list[FixApplied] = []
    skipped: list[FixSkipped] = []
    for fix in fixes:
        quote = fix.get("quote") or ""
        repl = fix.get("replacement") or ""
        if not quote or not repl:
            skipped.append(FixSkipped(quote, "missing_quote_or_replacement"))
            continue
        if "\n" in quote or "\n" in repl:
            skipped.append(FixSkipped(quote, "multiline"))
            continue
        if repl == quote:
            skipped.append(FixSkipped(quote, "noop"))
            continue
        n = text.count(quote)
        if n == 0:
            skipped.append(FixSkipped(quote, "not_found"))
            continue
        if n > 1:
            skipped.append(FixSkipped(quote, f"ambiguous:{n}_matches"))
            continue
        if _class_a_tokens(repl) != _class_a_tokens(quote):
            skipped.append(FixSkipped(quote, "latex_token_drift"))
            continue
        text = text.replace(quote, repl, 1)
        applied.append(FixApplied(quote, repl))
    return text, applied, skipped

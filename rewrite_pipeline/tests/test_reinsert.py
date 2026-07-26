"""Reinsertion + integrity tests."""

from __future__ import annotations

from dataclasses import replace

from rewrite_pipeline.extract import extract
from rewrite_pipeline.integrity import structural_diff
from rewrite_pipeline.model import Manifest
from rewrite_pipeline.reinsert import apply_rewrites, validate_rewrite


def wrap(body: str) -> str:
    return (
        "\\documentclass{article}\n\\begin{document}\n\n"
        + body
        + "\n\n\\end{document}\n"
    )


def _record(text: str, substr: str):
    man = extract(text)
    rec = next(r for r in man.records if substr in r.text)
    return man, rec


def test_validate_rejects_dropped_math():
    _, rec = _record(wrap("Thus $D=M+L$ holds exactly here. Next one here."), "Thus")
    rw, err = validate_rewrite(rec.text, "Thus it holds exactly.", rec)
    assert rw is None and err == "dollar_count_mismatch"


def test_validate_rejects_dropped_citation():
    _, rec = _record(
        wrap("We build on prior work \\citep{smith2020} here now. Next one here."),
        "prior work",
    )
    rw, err = validate_rewrite(rec.text, "We build on prior work here now.", rec)
    assert rw is None and err == "latex_token_drift"


def test_validate_rejects_dropped_escaped_braces():
    # \{ \} are literal set delimiters; dropping them changes the maths but
    # passes the (escape-blind) brace-balance check, so they must be Class-A.
    _, rec = _record(
        wrap("The set \\{a,b\\} is finite here now. Next one here."), "finite"
    )
    rw, err = validate_rewrite(rec.text, "The set a,b is finite now.", rec)
    assert rw is None and err == "latex_token_drift"
    rw, err = validate_rewrite(rec.text, "The set \\{a,b\\} is finite now.", rec)
    assert err is None and rw is not None


def test_validate_allows_escaped_percent():
    # \% is a literal percent sign, not a comment: a rewrite that keeps it must
    # pass, an unescaped % (comments out the rest of the line) must not, and
    # dropping the \% is token drift.
    _, rec = _record(wrap("Roughly 50\\% of runs fail here now. Next one here."), "50")
    rw, err = validate_rewrite(rec.text, "About 50\\% of runs fail.", rec)
    assert err is None and rw is not None
    rw, err = validate_rewrite(rec.text, "About 50 % of runs fail.", rec)
    assert rw is None and err == "forbidden:'%'"
    rw, err = validate_rewrite(rec.text, "About half of runs fail.", rec)
    assert rw is None and err == "latex_token_drift"


def test_validate_rejects_changed_unknown_macro():
    # Any macro invocation is immutable — \gls{latency-error} resolves a
    # defined term, so retargeting or dropping it changes content while still
    # compiling — but STYLE_CMDS prose wrappers stay rewritable.
    _, rec = _record(
        wrap("We call this \\gls{latency-error} here now. Next one here."), "call"
    )
    rw, err = validate_rewrite(rec.text, "We call this \\gls{delay} now.", rec)
    assert rw is None and err == "latex_token_drift"
    rw, err = validate_rewrite(rec.text, "We call this now.", rec)
    assert rw is None and err == "latex_token_drift"
    rw, err = validate_rewrite(rec.text, "We call this \\gls{latency-error} now.", rec)
    assert err is None and rw is not None


def test_validate_allows_rephrasing_inside_style_wrappers():
    _, rec = _record(
        wrap("We call this \\emph{latency error} here now. Next one here."), "call"
    )
    rw, err = validate_rewrite(rec.text, "We call this \\emph{delay error} now.", rec)
    assert err is None and rw is not None


def test_validate_allows_math_reflow():
    # $D=M+L$ -> $D = M + L$ is render-identical, so whitespace is normalised away.
    _, rec = _record(wrap("Thus $D=M+L$ holds here now. Next one here."), "Thus")
    rw, err = validate_rewrite(rec.text, "Thus $D = M + L$ holds now.", rec)
    assert err is None and rw is not None


def test_validate_rejects_newline_and_noop():
    _, rec = _record(wrap("A plain sentence here now. Next one here."), "plain")
    assert validate_rewrite(rec.text, "A\nB.", rec)[1].startswith("forbidden")
    assert validate_rewrite(rec.text, rec.text, rec)[1] == "noop"


def test_validate_reappends_terminal():
    _, rec = _record(wrap("A plain sentence here now. Next one here."), "plain")
    rw, err = validate_rewrite(rec.text, "A short sentence", rec)
    assert err is None
    assert rw.endswith(".")


LIST_AND_CAPTION = (
    "\\begin{itemize}\n"
    "\\item the fixed cost of training, safety work, and model development;\n"
    "\\end{itemize}\n\n"
    "\\begin{table}\n\\caption{Observed evidence and present status}\n\\end{table}\n"
)


def test_validate_keeps_semicolon_on_list_item():
    """A ';' list item keeps its ';' whether the rewrite supplies one or not."""
    _, rec = _record(wrap(LIST_AND_CAPTION), "fixed cost of training")
    assert not rec.has_terminal
    for candidate in ("training and safety work", "training and safety work;"):
        rw, err = validate_rewrite(rec.text, candidate, rec)
        assert err is None, (candidate, err)
        assert rw.endswith(";") and not rw.endswith(".;")


def test_validate_strips_added_full_stop_from_list_item():
    """A rewrite that acquires a full stop is normalised back to the ';'."""
    _, rec = _record(wrap(LIST_AND_CAPTION), "fixed cost of training")
    rw, err = validate_rewrite(rec.text, "training and safety work.", rec)
    assert err is None
    assert rw == "training and safety work;"


def test_validate_keeps_caption_title_unpunctuated():
    """A bare title must not acquire a full stop — that is a style edit."""
    _, rec = _record(wrap(LIST_AND_CAPTION), "Observed evidence")
    assert not rec.has_terminal
    rw, err = validate_rewrite(rec.text, "Evidence and current status.", rec)
    assert err is None
    assert rw == "Evidence and current status"


def test_validate_still_noops_an_unchanged_list_item():
    """The noop guard survives the clause-ending normalisation."""
    _, rec = _record(wrap(LIST_AND_CAPTION), "fixed cost of training")
    assert validate_rewrite(rec.text, rec.text, rec)[1] == "noop"


def test_apply_descending_offsets_are_correct():
    text = wrap("Alpha beta gamma here. Delta epsilon zeta here. Eta theta iota here.")
    man = extract(text)
    ins = [r for r in man.records if r.in_scope]
    acc = [
        {"id": ins[0].id, "rewrite": "Alpha beta here."},
        {"id": ins[2].id, "rewrite": "Eta theta here."},
    ]
    new, applied, skipped = apply_rewrites(text, man, acc)
    assert len(applied) == 2 and not skipped
    assert "Alpha beta here." in new
    assert "Eta theta here." in new
    assert "Delta epsilon zeta here." in new  # untouched middle sentence intact


def test_apply_skips_duplicate_id():
    # The same id twice would splice one span twice, the second pass cutting
    # into the first rewrite's text — only the first occurrence may apply.
    text = wrap("Alpha beta gamma here. Delta epsilon zeta here.")
    man = extract(text)
    rec = next(r for r in man.records if r.in_scope)
    acc = [
        {"id": rec.id, "rewrite": "Short version one here."},
        {"id": rec.id, "rewrite": "Totally different two here."},
    ]
    new, applied, skipped = apply_rewrites(text, man, acc)
    assert len(applied) == 1
    assert skipped and skipped[0].reason == "duplicate_id"
    assert "Short version one here." in new
    assert "Totally different" not in new
    # An id is only "seen" once a valid edit exists: invalid-then-valid applies
    # the valid one instead of dropping both.
    acc = [
        {"id": rec.id, "rewrite": "Bad % rewrite here."},
        {"id": rec.id, "rewrite": "Good version goes here."},
    ]
    new, applied, skipped = apply_rewrites(text, man, acc)
    assert len(applied) == 1 and "Good version goes here." in new
    assert [s.reason for s in skipped] == ["forbidden:'%'"]


def test_apply_skips_offset_mismatch():
    text = wrap("Alpha beta gamma here. Delta epsilon zeta here.")
    man = extract(text)
    rec = next(r for r in man.records if r.in_scope)
    # Tamper the record so its stored text no longer matches the source span.
    bad = Manifest(
        file_path=man.file_path,
        file_sha256=man.file_sha256,
        git_commit=None,
        extractor_version=man.extractor_version,
        records=[replace(rec, text=rec.text + "X")],
    )
    _, applied, skipped = apply_rewrites(
        text, bad, [{"id": rec.id, "rewrite": "Short one."}]
    )
    assert not applied
    assert skipped and skipped[0].reason == "offset_mismatch"


def test_structural_diff_flags_blank_line_change():
    old = "a b c here now.\n\nd e f here now.\n"
    good = "a b here now.\n\nd e f here now.\n"  # same paragraph structure
    bad = "a b.\n\nc here now.\n\nd e f here now.\n"  # a rewrite that split a paragraph
    assert structural_diff(old, good) == []
    assert any("blank-line" in p for p in structural_diff(old, bad))


def test_structural_diff_flags_brace_imbalance():
    old = "the \\emph{x} value.\n"
    bad = "the \\emph{x value.\n"
    assert any("brace" in p for p in structural_diff(old, bad))

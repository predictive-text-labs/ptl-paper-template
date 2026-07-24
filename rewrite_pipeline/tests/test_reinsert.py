"""Reinsertion + integrity tests."""

from __future__ import annotations

from dataclasses import replace

from rewrite_pipeline.extract import extract
from rewrite_pipeline.integrity import structural_diff
from rewrite_pipeline.model import Manifest
from rewrite_pipeline.reinsert import apply_rewrites, validate_rewrite


def wrap(body: str) -> str:
    return "\\documentclass{article}\n\\begin{document}\n\n" + body + "\n\n\\end{document}\n"


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
    _, applied, skipped = apply_rewrites(text, bad, [{"id": rec.id, "rewrite": "Short one."}])
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

"""Extractor tests: crafted LaTeX for each hazard + a round-trip on the paper."""

from __future__ import annotations

from pathlib import Path

from rewrite_pipeline.extract import extract

REPO_ROOT = Path(__file__).resolve().parents[2]


def wrap(body: str) -> str:
    return (
        "\\documentclass{article}\n\\begin{document}\n\n"
        + body
        + "\n\n\\end{document}\n"
    )


def sentences(body: str, in_scope_only: bool = True) -> list[str]:
    man = extract(wrap(body))
    return [r.text for r in man.records if (r.in_scope or not in_scope_only)]


def test_two_plain_sentences():
    assert sentences("First sentence here. Second one here.") == [
        "First sentence here.",
        "Second one here.",
    ]


def test_inline_math_period():
    # Period after $...$ is a real boundary; math is kept inside the sentence.
    out = sentences("Thus $D=M+L$. Next one here.")
    assert out == ["Thus $D=M+L$.", "Next one here."]


def test_decimal_not_split():
    out = sentences("The value is 0.25 today. All done here.")
    assert out == ["The value is 0.25 today.", "All done here."]


def test_et_al_not_split():
    out = sentences("Dud\\'ik et al.\\ decompose the score here. Next one here.")
    assert len(out) == 2
    assert "et al" in out[0]


def test_citation_before_period():
    out = sentences("We rely on prior work \\citep{a,b}. Next one here.")
    assert out[0] == "We rely on prior work \\citep{a,b}."
    assert out[1] == "Next one here."


def test_escaped_dollar_is_not_math():
    man = extract(wrap("It cost \\$1B last year here. All done here."))
    first = next(r for r in man.records if r.in_scope)
    assert "\\$1B" in first.text
    assert first.n_dollars == 0  # escaped dollar is not a math delimiter


def test_ref_before_period():
    out = sentences("See Section~\\ref{sec:x} for details. Next one here.")
    assert out[0] == "See Section~\\ref{sec:x} for details."


def test_emph_content_is_prose():
    out = sentences("We call this \\emph{latency error} here. Next one here.")
    assert out[0] == "We call this \\emph{latency error} here."


def test_display_math_is_a_break():
    # A fragment running into display math has no terminator -> excluded.
    man = extract(wrap("We define \\[ D = M + L \\] as the split. Next one here."))
    texts = [r.text for r in man.records if r.in_scope]
    assert "Next one here." in texts
    # the pre-display fragment "We define" has no terminal punctuation
    assert not any(t.strip() == "We define" for t in texts)


def test_lorem_excluded():
    man = extract(
        wrap(
            "Lorem ipsum dolor sit amet consectetur adipiscing elit here. Real sentence here."
        )
    )
    reasons = {r.excluded_reason for r in man.records if not r.in_scope}
    assert "placeholder_lorem" in reasons
    assert any(r.text == "Real sentence here." and r.in_scope for r in man.records)


def test_footnote_becomes_own_record_and_parent_excluded():
    body = "Everything else\\footnote{A short note here. And another.} follows here. Next one here."
    man = extract(wrap(body))
    kinds = {r.kind for r in man.records}
    assert "footnote" in kinds
    parent = next(
        r for r in man.records if r.kind != "footnote" and "Everything else" in r.text
    )
    assert parent.contains_footnote is True
    assert parent.in_scope is False


def test_section_title_not_a_sentence():
    man = extract(
        wrap("\\section{Introduction}\n\nA real sentence here. Another one here.")
    )
    texts = [r.text for r in man.records if r.in_scope]
    assert "Introduction" not in texts
    assert "A real sentence here." in texts


def test_commented_end_inside_opaque_env_is_ignored():
    # A commented-out % \end{tabular} must not terminate the masked span early,
    # or the rest of the table leaks into prose. The leaked line is terminal-
    # punctuated so it would land IN SCOPE if the mask ended early.
    body = (
        "Before text goes here now.\n\n"
        "\\begin{tabular}{ll}\n"
        "a & b \\\\\n"
        "% \\end{tabular}\n"
        "This row would leak badly here now.\n"
        "\\end{tabular}\n\n"
        "After text goes here now. Next one here."
    )
    man = extract(wrap(body))
    texts = [r.text for r in man.records if r.in_scope]
    assert "Before text goes here now." in texts
    assert "After text goes here now." in texts
    assert not any("leak badly" in t for t in texts)


def test_literal_percent_in_verb_does_not_hide_env_end():
    # \verb|%| is a literal percent, not a comment: the comment-aware scan finds
    # no end token at all and must degrade to the blind match instead of
    # masking the rest of the document.
    body = (
        "\\begin{tabular}{ll}\n"
        "a & \\verb|%| b \\end{tabular}\n\n"
        "After text goes here now. Next one here."
    )
    man = extract(wrap(body))
    texts = [r.text for r in man.records if r.in_scope]
    assert "After text goes here now." in texts


def test_verbatim_commented_end_still_closes():
    # Inside verbatim, % is a literal character, so TeX ends the environment at
    # the \end token even on a % line — the scanner must match that.
    body = "\\begin{verbatim}\n% \\end{verbatim}\nAfter text goes here now. Next one here."
    man = extract(wrap(body))
    texts = [r.text for r in man.records if r.in_scope]
    assert "After text goes here now." in texts


# ---- round-trip on whatever paper the repo holds ----


def test_repo_paper_roundtrip_and_scope():
    tex_files = sorted(REPO_ROOT.glob("*.tex"))
    assert tex_files, "no top-level .tex manuscript found — round-trip covered nothing"
    for tex in tex_files:
        text = tex.read_text(encoding="utf-8")
        man = extract(text, file_path=str(tex))
        # every record's text is exactly its source span
        for r in man.records:
            assert text[r.abs_start : r.abs_end] == r.text
        ap = text.find("\\appendix")
        for r in man.records:
            if r.in_scope:
                if ap != -1:
                    assert r.abs_start < ap  # no in-scope sentence past the appendix
                assert r.n_dollars % 2 == 0
                assert r.n_brace_delta == 0
                assert r.has_terminal

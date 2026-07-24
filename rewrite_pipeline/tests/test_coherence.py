"""Unit tests for the cross-sentence coherence gate."""

from rewrite_pipeline.coherence import (
    accepted_fingerprint,
    apply_fixes,
    build_pairs,
)


def test_fingerprint_tracks_content_and_order() -> None:
    a = [{"id": "0001", "rewrite": "Short."}]
    b = [{"id": "0001", "rewrite": "Shorter."}]
    assert accepted_fingerprint(a) == accepted_fingerprint(a)
    assert accepted_fingerprint(a) != accepted_fingerprint(b)
    two = a + b
    assert accepted_fingerprint(two) != accepted_fingerprint(list(reversed(two)))


def test_build_pairs_returns_changed_paragraphs_only() -> None:
    old = "alpha one.\n\nbeta two.\n\ngamma three.\n"
    new = "alpha 1.\n\nbeta two.\n\ngamma 3.\n"
    pairs = build_pairs(old, new)
    assert pairs == [
        {"old": "alpha one.", "new": "alpha 1."},
        {"old": "gamma three.\n", "new": "gamma 3.\n"},
    ]


def test_build_pairs_handles_hard_wrap_collapse() -> None:
    # A hard-wrapped original sentence may collapse onto one line: line counts
    # differ, paragraph counts do not.
    old = "a sentence wrapped\nacross two lines.\n\nunchanged.\n"
    new = "a shorter sentence.\n\nunchanged.\n"
    pairs = build_pairs(old, new)
    assert pairs == [
        {"old": "a sentence wrapped\nacross two lines.", "new": "a shorter sentence."}
    ]


def test_build_pairs_rejects_paragraph_count_change() -> None:
    try:
        build_pairs("a\n\nb\n", "a\n")
    except ValueError as e:
        assert "paragraph count" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_apply_fixes_exact_once() -> None:
    text = "X, Y, and Z still do.\nOther text."
    fixed, applied, skipped = apply_fixes(
        text, [{"quote": "still do.", "replacement": "still matter."}]
    )
    assert fixed == "X, Y, and Z still matter.\nOther text."
    assert len(applied) == 1 and not skipped


def test_apply_fixes_skips_missing_and_ambiguous() -> None:
    text = "one two one"
    _, applied, skipped = apply_fixes(
        text,
        [
            {"quote": "three", "replacement": "3"},
            {"quote": "one", "replacement": "1"},
        ],
    )
    assert not applied
    assert [s.reason for s in skipped] == ["not_found", "ambiguous:2_matches"]


def test_apply_fixes_skips_latex_token_drift() -> None:
    text = r"as shown \citep{smith2020} here."
    _, applied, skipped = apply_fixes(
        text,
        [{"quote": r"shown \citep{smith2020} here", "replacement": "shown here"}],
    )
    assert not applied and skipped[0].reason == "latex_token_drift"


def test_apply_fixes_allows_matching_tokens() -> None:
    text = r"This missing signal from $M$ powers our curriculum."
    fixed, applied, skipped = apply_fixes(
        text,
        [
            {
                "quote": r"This missing signal from $M$ powers our curriculum.",
                "replacement": r"$M$ provides this missing signal, powering our curriculum.",
            }
        ],
    )
    assert fixed == r"$M$ provides this missing signal, powering our curriculum."
    assert len(applied) == 1 and not skipped


def test_apply_fixes_rejects_forbidden_tokens() -> None:
    # An introduced unescaped % comments out the rest of the line and no later
    # gate counts % — the fix gate must be as strict as validate_rewrite.
    text = "The result holds for all cases here now."
    _, applied, skipped = apply_fixes(
        text, [{"quote": "holds for all cases", "replacement": "holds % for all"}]
    )
    assert not applied and skipped[0].reason == "forbidden:'%'"
    # The escaped literal \% is fine when preserved.
    text2 = r"Roughly 50\% of runs fail here."
    fixed, applied, skipped = apply_fixes(
        text2, [{"quote": r"Roughly 50\% of runs", "replacement": r"Half (50\%) of runs"}]
    )
    assert len(applied) == 1 and not skipped
    assert fixed == r"Half (50\%) of runs fail here."


def test_apply_fixes_rejects_dollar_and_brace_changes() -> None:
    # An UNPAIRED $ is invisible to the Class-A regex ($...$ spans), so the
    # count gate is the only thing standing between it and the .tex.
    text = "the value x grows here."
    _, applied, skipped = apply_fixes(
        text, [{"quote": "the value x grows", "replacement": "the value $x grows"}]
    )
    assert not applied and skipped[0].reason == "dollar_count_mismatch"
    _, applied, skipped = apply_fixes(
        text, [{"quote": "the value x grows", "replacement": "the {value x grows"}]
    )
    assert not applied and skipped[0].reason == "brace_imbalance"


def test_apply_fixes_skips_multiline_and_noop() -> None:
    text = "a sentence here."
    _, applied, skipped = apply_fixes(
        text,
        [
            {"quote": "a sentence", "replacement": "a\nsentence"},
            {"quote": "here.", "replacement": "here."},
        ],
    )
    assert not applied
    assert [s.reason for s in skipped] == ["multiline", "noop"]

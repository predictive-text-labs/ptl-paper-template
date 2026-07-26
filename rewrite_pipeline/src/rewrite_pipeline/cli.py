"""Command-line entry point for the sentence-rewrite pipeline.

Stages:
  extract  paper.tex           -> <workdir>/sentence_index.json   (manifest)
  fanout   manifest            -> <workdir>/gemini_out.json        (Gemini raw)
  split    gemini_out.json     -> <workdir>/judge_batches/*.json   (one pair/file)
  pairs    manifest + accepted -> <workdir>/coherence_pairs/*.json (changed paras)
  apply    manifest + accepted -> sidecar + diff + review.html  (--apply to commit)

Two Claude Fable steps run separately via the Workflow tool:
  * judge (judge-rewrites.workflow.mjs) over the ``split`` batch files ->
    ``<workdir>/accepted.json``;
  * coherence sweep (coherence-sweep.workflow.mjs) over the ``pairs`` files ->
    ``<workdir>/coherence_fixes.json``. Per-sentence judging cannot see seam
    damage in neighbouring sentences, so ``apply --apply`` requires this
    sign-off (fresh for the current accepted set) unless --skip-coherence.
"""

from __future__ import annotations

import argparse
import difflib
import json
import subprocess
import sys
from pathlib import Path

from .coherence import accepted_fingerprint, apply_fixes, build_pairs, load_fixes
from .extract import extract
from .integrity import compile_check, pdf_page_count, structural_diff
from .model import Manifest, sha256_hex
from .reinsert import apply_rewrites
from .review import build_review_html

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parents[1]  # rewrite_pipeline/
REPO_ROOT = PROJECT_DIR.parent
DEFAULT_WORKDIR = PROJECT_DIR / "run"


def _default_tex() -> str | None:
    """The repo's paper: the sole top-level .tex file, if unambiguous."""
    texes = sorted(REPO_ROOT.glob("*.tex"))
    return str(texes[0]) if len(texes) == 1 else None


def _add_tex_arg(sub: argparse.ArgumentParser) -> None:
    default = _default_tex()
    sub.add_argument(
        "--tex",
        default=default,
        required=default is None,
        help="paper .tex (default: the repo's sole top-level .tex)",
    )


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return out.stdout.strip() or None
    except subprocess.SubprocessError, OSError:
        return None


def _manifest_path(workdir: Path) -> Path:
    return workdir / "sentence_index.json"


def cmd_extract(args: argparse.Namespace) -> int:
    tex = Path(args.tex)
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    text = tex.read_text(encoding="utf-8")
    man = extract(text, file_path=str(tex), git_commit=_git_commit())
    _manifest_path(workdir).write_text(man.to_json(), encoding="utf-8")

    from collections import Counter

    ins = [r for r in man.records if r.in_scope]
    kinds = Counter(r.kind for r in ins)
    reasons = Counter(r.excluded_reason for r in man.records if not r.in_scope)
    print(f"extracted {len(man.records)} sentences; {len(ins)} in scope")
    print(f"  in-scope by kind: {dict(kinds)}")
    print(f"  excluded reasons: {dict(reasons)}")
    if reasons.get("placeholder_lorem"):
        print(
            f"  NOTE: {reasons['placeholder_lorem']} lorem-ipsum placeholder sentences excluded."
        )
    print(f"  manifest -> {_manifest_path(workdir)}")
    return 0


def cmd_fanout(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir)
    man = Manifest.from_json(_manifest_path(workdir).read_text(encoding="utf-8"))
    from .gemini_fanout import run as fanout_run

    out = workdir / "gemini_out.json"
    fanout_run(
        man,
        out,
        model=args.model,
        env_path=Path(args.env) if args.env else None,
    )
    return 0


def cmd_split(args: argparse.Namespace) -> int:
    """Split gemini_out.json into per-agent batch files for the judge workflow.

    Default batch size is 1 — one Fable judge per sentence pair, so each judge
    spends its whole attention on a single original<->rewrite comparison.
    """
    workdir = Path(args.workdir)
    gemini_out = json.loads((workdir / "gemini_out.json").read_text(encoding="utf-8"))
    candidates = [
        r
        for r in gemini_out
        if r.get("status") == "ok" and r.get("gemini_raw_response")
    ]
    outdir = workdir / "judge_batches"
    if outdir.exists():
        for f in outdir.glob("batch_*.json"):
            f.unlink()
    outdir.mkdir(parents=True, exist_ok=True)

    n = max(1, args.batch_size)
    paths: list[str] = []
    for i in range(0, len(candidates), n):
        chunk = [
            {
                "id": r["id"],
                "kind": r["kind"],
                "original_sentence": r["original_sentence"],
                "gemini_raw_response": r["gemini_raw_response"],
            }
            for r in candidates[i : i + n]
        ]
        p = outdir / f"batch_{i // n:04d}.json"
        p.write_text(json.dumps(chunk, ensure_ascii=False, indent=1), encoding="utf-8")
        paths.append(str(p.resolve()))

    (workdir / "judge_batch_paths.json").write_text(
        json.dumps(paths, indent=1), encoding="utf-8"
    )
    print(
        f"split {len(candidates)} candidates into {len(paths)} batch files "
        f"of <= {n} -> {outdir}"
    )
    print(f"  paths -> {workdir / 'judge_batch_paths.json'}")
    return 0


def _load_accepted(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("accepted", [])
    return data


def cmd_pairs(args: argparse.Namespace) -> int:
    """Build coherence pair files: every paragraph the accepted set would change.

    Per-sentence judging cannot see that a rewrite orphaned a pro-verb,
    demonstrative, or enumeration label in a NEIGHBOURING sentence, so the
    coherence-sweep workflow re-reads each changed paragraph whole. This stage
    prepares its inputs from a virtual apply (nothing is written to the .tex).
    """
    tex = Path(args.tex)
    workdir = Path(args.workdir)
    man = Manifest.from_json(_manifest_path(workdir).read_text(encoding="utf-8"))
    text = tex.read_text(encoding="utf-8")
    if sha256_hex(text) != man.file_sha256:
        print(
            "ABORT: file has changed since extraction (sha256 mismatch). Re-run extract."
        )
        return 2

    accepted_path = Path(args.accepted) if args.accepted else workdir / "accepted.json"
    accepted = _load_accepted(accepted_path)
    new_text, applied, _skipped = apply_rewrites(text, man, accepted)
    pairs = build_pairs(text, new_text)

    outdir = workdir / "coherence_pairs"
    if outdir.exists():
        for f in outdir.glob("pair_*.json"):
            f.unlink()
    outdir.mkdir(parents=True, exist_ok=True)
    for i, pair in enumerate(pairs):
        (outdir / f"pair_{i:03d}.json").write_text(
            json.dumps(pair, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    fingerprint = accepted_fingerprint(accepted)
    manifest = {
        "pair_dir": str(outdir.resolve()),
        "pair_count": len(pairs),
        "accepted_sha256": fingerprint,
        "tex_sha256": man.file_sha256,
    }
    (workdir / "coherence_pairs.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8"
    )
    print(
        f"pairs: {len(applied)} accepted rewrites touch {len(pairs)} paragraphs "
        f"-> {outdir}"
    )
    print("  run coherence-sweep.workflow.mjs with args:")
    print(f"  {json.dumps(manifest)}")
    print(f"  then save its full return value to {workdir / 'coherence_fixes.json'}")
    return 0


def cmd_apply(args: argparse.Namespace) -> int:
    tex = Path(args.tex)
    workdir = Path(args.workdir)
    man = Manifest.from_json(_manifest_path(workdir).read_text(encoding="utf-8"))
    text = tex.read_text(encoding="utf-8")

    # Hash gate: offsets are only valid against the exact bytes they were cut from.
    if sha256_hex(text) != man.file_sha256:
        print(
            "ABORT: file has changed since extraction (sha256 mismatch). Re-run extract."
        )
        return 2

    accepted_path = Path(args.accepted) if args.accepted else workdir / "accepted.json"
    accepted = _load_accepted(accepted_path)
    rejected = []
    rej_path = workdir / "rejected.json"
    if rej_path.exists():
        rd = json.loads(rej_path.read_text(encoding="utf-8"))
        rejected = rd.get("rejected", rd) if isinstance(rd, dict) else rd

    new_text, applied, skipped = apply_rewrites(text, man, accepted)

    # Coherence gate: per-sentence judging cannot see seam damage in the
    # neighbouring sentences, so --apply requires a sweep sign-off (a
    # coherence_fixes.json fingerprinted to THIS accepted set) unless
    # --skip-coherence. An empty fixes list is a valid all-clear.
    fixes_path = workdir / "coherence_fixes.json"
    fingerprint = accepted_fingerprint(accepted)
    fixes: list[dict] = []
    fixes_fp: str | None = None
    if fixes_path.exists():
        fixes, fixes_fp = load_fixes(fixes_path)
    coherence_ok = fixes_fp == fingerprint
    if not coherence_ok:
        why = (
            "is stale (accepted.json changed since the sweep)"
            if fixes_path.exists()
            else "is missing"
        )
        howto = (
            f"coherence sign-off {why}: run `rewrite pairs`, then the "
            "coherence-sweep workflow, and save its return value to "
            f"{fixes_path}"
        )
        if args.apply and not args.skip_coherence:
            print(f"ABORT: {howto} (or pass --skip-coherence).")
            return 6
        print(f"  WARNING: {howto}")
        fixes = []

    fx_applied, fx_skipped = [], []
    if fixes:
        new_text, fx_applied, fx_skipped = apply_fixes(new_text, fixes)
        for fx in fx_applied:
            print(f"  coherence fix: {fx.quote!r} -> {fx.replacement!r}")
        for fs in fx_skipped:
            print(f"  coherence fix SKIPPED ({fs.reason}): {fs.quote!r}")

    problems = structural_diff(text, new_text)

    print(
        f"apply: {len(applied)} applied, {len(skipped)} skipped, {len(accepted)} proposed"
    )
    if skipped:
        for s in skipped[:20]:
            print(f"  skip {s.id}: {s.reason}")
    if problems:
        print("STRUCTURAL PROBLEMS (rewrite discarded):")
        for p in problems:
            print(f"  - {p}")
        return 3

    stats = {
        "proposed": len(accepted),
        "applied": len(applied),
        "skipped": len(skipped),
        "rejected": len(rejected),
        "coherence_fixes": len(fx_applied),
    }

    # Always write the dry-run artifacts.
    sidecar = tex.with_suffix(".rewritten.tex")
    sidecar.write_text(new_text, encoding="utf-8")
    diff = "".join(
        difflib.unified_diff(
            text.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile=str(tex),
            tofile=str(sidecar),
        )
    )
    (workdir / "rewrite.diff").write_text(diff, encoding="utf-8")
    html = build_review_html(applied, skipped, rejected, stats)
    (workdir / "review.html").write_text(html, encoding="utf-8")
    print(f"  sidecar -> {sidecar}")
    print(f"  diff    -> {workdir / 'rewrite.diff'}")
    print(f"  review  -> {workdir / 'review.html'}")

    if not args.apply:
        print(
            "dry run complete (no files changed). Re-run with --apply to write + compile."
        )
        return 0

    # --apply: refuse on a dirty tex unless --force, so a revert is one step.
    if not args.force and _tex_dirty(tex):
        print(
            "ABORT: the .tex has uncommitted changes. Commit/stash first, or pass --force."
        )
        return 4

    tex.write_text(new_text, encoding="utf-8")
    print(f"  WROTE {tex}")
    build = workdir / "build"
    ok, log = compile_check(tex, build)
    if not ok:
        tex.write_text(text, encoding="utf-8")  # revert
        print("COMPILE FAILED — reverted the .tex. latexmk tail:")
        print(log)
        return 5
    pages = pdf_page_count(build / tex.with_suffix(".pdf").name)
    print(
        f"  compiled OK ({'pages=' + str(pages) if pages else 'page count unknown'})."
    )
    print(f"  review `git diff {tex.name}`; revert with `git checkout -- {tex.name}`.")
    return 0


def _tex_dirty(tex: Path) -> bool:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain", "--", tex.name],
            cwd=str(tex.parent),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return bool(out.stdout.strip())
    except subprocess.SubprocessError, OSError:
        return False


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rewrite", description=__doc__)
    p.add_argument("--workdir", default=str(DEFAULT_WORKDIR))
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("extract", help="split the paper into a sentence manifest")
    _add_tex_arg(pe)
    pe.set_defaults(func=cmd_extract)

    pf = sub.add_parser(
        "fanout", help="send in-scope sentences to Gemini (all at once)"
    )
    # Default from the one source of truth, so bumping the model is a one-line
    # change in gemini_fanout instead of a silent mismatch with this flag.
    from .gemini_fanout import MODEL as FANOUT_MODEL

    pf.add_argument("--model", default=FANOUT_MODEL)
    pf.add_argument("--env", default=None, help="path to .env (default: search upward)")
    pf.set_defaults(func=cmd_fanout)

    ps = sub.add_parser(
        "split", help="split gemini_out.json into per-judge batch files"
    )
    ps.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="sentences per judge agent (default 1 = one judge per sentence pair)",
    )
    ps.set_defaults(func=cmd_split)

    pp = sub.add_parser(
        "pairs", help="build changed-paragraph pair files for the coherence sweep"
    )
    _add_tex_arg(pp)
    pp.add_argument(
        "--accepted",
        default=None,
        help="accepted.json (default: <workdir>/accepted.json)",
    )
    pp.set_defaults(func=cmd_pairs)

    pa = sub.add_parser("apply", help="splice accepted rewrites (dry-run by default)")
    _add_tex_arg(pa)
    pa.add_argument(
        "--accepted",
        default=None,
        help="accepted.json (default: <workdir>/accepted.json)",
    )
    pa.add_argument(
        "--apply", action="store_true", help="write the .tex and run the compile gate"
    )
    pa.add_argument(
        "--force", action="store_true", help="allow --apply on a dirty .tex"
    )
    pa.add_argument(
        "--skip-coherence",
        action="store_true",
        help="allow --apply without a fresh coherence-sweep sign-off",
    )
    pa.set_defaults(func=cmd_apply)
    return p


def main(argv: list[str] | None = None) -> int:
    # Every stage reports progress on stdout, which Python block-buffers as soon
    # as it is not a TTY — so piping to a log or running as a background task
    # would otherwise swallow the whole run's output until exit and make a long
    # stage look like a hang. Line buffering makes each print land immediately.
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        try:
            reconfigure(line_buffering=True)
        except OSError, ValueError:  # pragma: no cover - detached/odd stdout
            pass
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

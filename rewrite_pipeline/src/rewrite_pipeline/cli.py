"""Command-line entry point for the sentence-rewrite pipeline.

Stages:
  extract  paper.tex           -> <workdir>/sentence_index.json   (manifest)
  fanout   manifest            -> <workdir>/gemini_out.json        (Gemini raw)
  split    gemini_out.json     -> <workdir>/judge_batches/*.json   (one pair/file)
  apply    manifest + accepted -> sidecar + diff + review.html  (--apply to commit)

The Claude Fable judging step (Stage B) runs separately via the Workflow tool over
the batch files from ``split`` and produces ``<workdir>/accepted.json``, which
``apply`` consumes.
"""

from __future__ import annotations

import argparse
import difflib
import json
import subprocess
import sys
from pathlib import Path

from .extract import extract
from .integrity import compile_check, pdf_page_count, structural_diff
from .model import Manifest, sha256_hex
from .reinsert import apply_rewrites
from .review import build_review_html

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parents[1]  # rewrite_pipeline/
REPO_ROOT = PROJECT_DIR.parent
DEFAULT_TEX = REPO_ROOT / "Is_It_Priced_In.tex"
DEFAULT_WORKDIR = PROJECT_DIR / "run"


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
    except (subprocess.SubprocessError, OSError):
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
        r for r in gemini_out if r.get("status") == "ok" and r.get("gemini_raw_response")
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
    print(
        "  review `git diff Is_It_Priced_In.tex`; revert with `git checkout -- Is_It_Priced_In.tex`."
    )
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
    except (subprocess.SubprocessError, OSError):
        return False


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rewrite", description=__doc__)
    p.add_argument("--workdir", default=str(DEFAULT_WORKDIR))
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("extract", help="split the paper into a sentence manifest")
    pe.add_argument("--tex", default=str(DEFAULT_TEX))
    pe.set_defaults(func=cmd_extract)

    pf = sub.add_parser("fanout", help="send in-scope sentences to Gemini (all at once)")
    pf.add_argument("--model", default="gemini-3.1-pro-preview")
    pf.add_argument("--env", default=None, help="path to .env (default: search upward)")
    pf.set_defaults(func=cmd_fanout)

    ps = sub.add_parser("split", help="split gemini_out.json into per-judge batch files")
    ps.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="sentences per judge agent (default 1 = one judge per sentence pair)",
    )
    ps.set_defaults(func=cmd_split)

    pa = sub.add_parser("apply", help="splice accepted rewrites (dry-run by default)")
    pa.add_argument("--tex", default=str(DEFAULT_TEX))
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
    pa.set_defaults(func=cmd_apply)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

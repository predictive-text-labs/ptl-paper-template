"""Document-integrity checks and a latexmk compile gate for Stage C."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path


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


def _blank_line_count(t: str) -> int:
    return len(re.findall(r"\n[ \t]*\n", t))


def structural_diff(old: str, new: str) -> list[str]:
    """Return a list of structural regressions introduced by the rewrite pass.

    Rewrites are single-line and confined to intra-paragraph sentence spans, so
    these invariants must hold (line *count* is allowed to shrink when a
    multi-line sentence collapses — a newline is just whitespace in LaTeX)."""
    problems: list[str] = []
    if _count_unescaped(new, "$") % 2 != 0:
        problems.append(f"unbalanced $ (count={_count_unescaped(new, '$')})")
    delta = _count_unescaped(new, "{") - _count_unescaped(new, "}")
    if delta != 0:
        problems.append(f"brace imbalance (delta={delta})")
    for tok in ("\\begin{", "\\end{", "\\[", "\\]"):
        o = old.count(tok)
        m = new.count(tok)
        if o != m:
            problems.append(f"{tok!r} count changed {o}->{m}")
    ob, nb = _blank_line_count(old), _blank_line_count(new)
    if ob != nb:
        problems.append(f"blank-line (paragraph) count changed {ob}->{nb}")
    return problems


def compile_check(tex_path: Path, outdir: Path) -> tuple[bool, str]:
    """Run latexmk on ``tex_path`` into ``outdir``. Returns (ok, tail_of_log)."""
    latexmk = shutil.which("latexmk")
    if latexmk is None:
        return False, "latexmk not found on PATH"
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        latexmk,
        "-pdf",
        "-interaction=nonstopmode",
        "-halt-on-error",
        f"-outdir={outdir}",
        str(tex_path),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(tex_path.parent),
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    ok = proc.returncode == 0
    log = proc.stdout + "\n" + proc.stderr
    tail = "\n".join(log.splitlines()[-40:])
    return ok, tail


def pdf_page_count(pdf_path: Path) -> int | None:
    """Best-effort page count via pdfinfo, else None."""
    pdfinfo = shutil.which("pdfinfo")
    if pdfinfo is None or not pdf_path.exists():
        return None
    try:
        proc = subprocess.run(
            [pdfinfo, str(pdf_path)],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except subprocess.SubprocessError, OSError:
        return None
    m = re.search(r"^Pages:\s+(\d+)", proc.stdout, re.MULTILINE)
    return int(m.group(1)) if m else None

"""Build a self-contained HTML review artifact for the dry-run apply step."""

from __future__ import annotations

import html
import re

from .reinsert import Applied, Skipped

_MATH = re.compile(r"\$[^$]*\$")
_CMD = re.compile(r"\\[A-Za-z]+\*?")
_BRACES = re.compile(r"[{}]")


def word_count(s: str) -> int:
    s = _MATH.sub(" M ", s)
    s = _CMD.sub(" ", s)
    s = _BRACES.sub(" ", s)
    return len([w for w in re.split(r"\s+", s.strip()) if w])


def _esc(s: str) -> str:
    return html.escape(s)


def build_review_html(
    applied: list[Applied],
    skipped: list[Skipped],
    rejected: list[dict],
    stats: dict,
) -> str:
    rows = []
    total_before = total_after = 0
    for a in applied:
        wb, wa = word_count(a.original), word_count(a.rewrite)
        total_before += wb
        total_after += wa
        delta = wa - wb
        cls = "cut" if delta < 0 else ("same" if delta == 0 else "grew")
        rows.append(
            f"""<tr class="{cls}">
  <td class="id">{_esc(a.id)}</td>
  <td class="orig">{_esc(a.original)}</td>
  <td class="rw">{_esc(a.rewrite)}</td>
  <td class="wc">{wb}&rarr;{wa}<br><span class="delta">{delta:+d}</span></td>
</tr>"""
        )
    saved = total_before - total_after
    skip_rows = "".join(
        f"<tr><td>{_esc(s.id)}</td><td>{_esc(s.reason)}</td></tr>" for s in skipped
    )
    rej_rows = "".join(
        f"<tr><td>{_esc(str(r.get('id')))}</td><td>{_esc(str(r.get('reason', '')))}</td>"
        f"<td>{_esc(str(r.get('flaw', '')))}</td></tr>"
        for r in rejected
    )
    stat_line = " · ".join(f"{k}: {v}" for k, v in stats.items())
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Rewrite review</title>
<style>
:root {{ color-scheme: light dark; }}
body {{ font: 15px/1.5 -apple-system, system-ui, sans-serif; margin: 2rem; max-width: 1200px; }}
h1 {{ font-size: 1.4rem; }}
.summary {{ background: rgba(127,127,127,.12); padding: .75rem 1rem; border-radius: 8px; margin: 1rem 0; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
th, td {{ text-align: left; vertical-align: top; padding: .5rem .6rem; border-bottom: 1px solid rgba(127,127,127,.25); }}
th {{ position: sticky; top: 0; background: Canvas; }}
td.id {{ font-family: ui-monospace, monospace; font-size: .8rem; white-space: nowrap; color: #888; }}
td.orig {{ width: 40%; }}
td.rw {{ width: 40%; }}
td.wc {{ white-space: nowrap; text-align: right; font-variant-numeric: tabular-nums; }}
.delta {{ font-weight: 600; }}
tr.cut .delta {{ color: #2e9e5b; }}
tr.grew .delta {{ color: #d9534f; }}
tr.same .delta {{ color: #999; }}
tr.cut td.rw {{ background: rgba(46,158,91,.08); }}
details {{ margin-top: 1.5rem; }}
code {{ font-family: ui-monospace, monospace; }}
</style></head><body>
<h1>Sentence-rewrite review</h1>
<div class="summary">
  <strong>{len(applied)} rewrites to apply</strong> · words {total_before}&rarr;{total_after}
  (<strong>{saved:+d}</strong>, {(100 * saved / total_before) if total_before else 0:.1f}% shorter)<br>
  <span style="color:#888">{_esc(stat_line)}</span>
</div>
<table>
<thead><tr><th>id</th><th>original</th><th>rewrite</th><th>words</th></tr></thead>
<tbody>
{"".join(rows)}
</tbody></table>
<details open><summary>{len(rejected)} rejected by judge/verify</summary>
<table><thead><tr><th>id</th><th>reason</th><th>flaw</th></tr></thead>
<tbody>{rej_rows}</tbody></table></details>
<details><summary>{len(skipped)} skipped at apply (offset/validation)</summary>
<table><thead><tr><th>id</th><th>reason</th></tr></thead>
<tbody>{skip_rows}</tbody></table></details>
</body></html>"""

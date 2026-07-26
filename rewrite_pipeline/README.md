# rewrite_pipeline

Shortens the sentences of the repo's paper — auto-detected as the sole
top-level `.tex` (here `paper.tex`); pass `--tex` if there are several — to
lower reading complexity.
Gemini proposes shorter rewrites; **Claude Fable** answers one question per
rewrite — *does it keep all the important details of the original?* — and keeps
it on yes; a second Fable sweep re-reads every changed paragraph whole for
cross-sentence coherence; accepted rewrites are spliced back byte-precisely for
review.

```
extract    paper.tex            → run/sentence_index.json    (deterministic Python)
fanout     in-scope sentences   → run/gemini_out.json        (async Gemini 3.6 Flash)
split      gemini_out.json      → run/judge_batches/*.json   (one sentence pair / file)
judge      batch files          → run/accepted.json          (Claude Fable, Workflow tool)
pairs      accepted + manifest  → run/coherence_pairs/*.json (deterministic Python)
coherence  pair files           → run/coherence_fixes.json   (Claude Fable, Workflow tool)
apply      accepted + fixes     → sidecar + diff + review    (deterministic Python)
```

## Setup

```bash
cd rewrite_pipeline
uv sync
# Gemini AI Studio key — goes in the repo-root .env (gitignored, never committed):
echo 'GEMINI_API_KEY=<your-ai-studio-key>' >> ../.env
```

The SDK reads `GEMINI_API_KEY` (or `GOOGLE_API_KEY`). Do **not** export
`GOOGLE_GENAI_USE_VERTEXAI` — it would flip the client off the AI Studio API.

## Run

### 1. Extract (deterministic; no key needed)
```bash
uv run rewrite extract
```
Splits the paper into a sentence manifest with byte-exact offsets. Scope is the
main body + abstract (the math-dense appendix is skipped). Prints per-kind counts
and exclusion reasons (incl. lorem-ipsum placeholder sentences).

### 2. Fan out to Gemini (needs the key)
```bash
uv run rewrite fanout           # gemini-3.6-flash at thinking_level=high
```
Sends each sentence the *verbatim* prompt, **all at once** — no stagger, no
concurrency cap. The account limits (22K RPM / 28M TPM) dwarf the job, and a
950-call burst was measured drawing zero 429s and zero 503s, so staggering buys
nothing and only delays the last call. Three parts are load-bearing:

* an httpx pool sized **above** the job (`MAX_CONNECTIONS`). The default of 100
  deadlocks a larger fan-out outright — calls queue, a queued call cancelled
  mid-acquire leaks its slot, and the pool bleeds into `CLOSE_WAIT`.
* **hedging** (`HEDGE_AFTER_S`). A stalled HTTP response never errors and never
  returns, so elapsed time is the only detector — but a slow call looks identical
  until one answers. Once a copy is 120s old a *duplicate* is raced against it
  and the first answer wins, instead of killing a copy that may only be queued.
* **infinite retry** on any error, bounded by one hard `CALL_DEADLINE_S` per
  sentence, so a cursed sentence lands `status="timeout"` instead of stalling
  the run. 400/401/403/404 are terminal and never retried.
Measured: 252/252 in ~2.5 min at high thinking. Writes `run/gemini_out.json`.

### 3. Split into per-judge batches (deterministic)
```bash
uv run rewrite split            # default --batch-size 1: one judge per sentence pair
```
Writes `run/judge_batches/batch_NNNN.json` (one Gemini result each) plus
`run/judge_batch_paths.json`. One pair per file means each Fable judge spends its
whole attention on a single original↔rewrite comparison.

### 4. Judge with Claude Fable (run via the Workflow tool — Claude triggers this)
The judging is a dynamic Workflow (`judge-rewrites.workflow.mjs`). One Fable
agent per batch file answers **one question** — *does the shortened rewrite
capture all the important details of the original?* — and keeps it on yes.
LaTeX-corruption safety (a silently-dropped `\citep`/`\ref` or mangled `$...$`)
is enforced deterministically by `apply` (`reinsert.validate_rewrite`), not by
the model. Accepted rewrites go to `run/accepted.json`, rejections to
`run/rejected.json`.

### 5. Coherence sweep (pre-apply gate; Claude triggers this)
```bash
uv run rewrite pairs             # builds run/coherence_pairs/pair_NNN.json
```
Per-sentence judging is structurally blind to seam damage: a rewrite that is
faithful on its own can orphan a pro-verb in the *neighbouring* sentence
("…does not alter the result. However, X, Y, and Z still **do**" → the rewrite
says "irrelevant" and "still do" loses its verb), strand a demonstrative, drop
an enumeration's labels a later sentence still cites, or break a "However"
contrast. `pairs` virtually applies the accepted set (nothing written) and
emits every changed paragraph as an old/new pair; the
`coherence-sweep.workflow.mjs` Fable workflow reads each pair whole and
returns minimal verbatim fixes, saved to `run/coherence_fixes.json`. The file
is fingerprinted to the exact accepted set, so a stale sign-off can't gate a
different apply.

### 6. Apply (deterministic; review first)
```bash
uv run rewrite apply             # dry run: sidecar + unified diff + review.html
uv run rewrite apply --apply     # writes the .tex, runs the latexmk compile gate
```
Dry run changes nothing — it writes `<paper>.rewritten.tex`,
`run/rewrite.diff`, and `run/review.html` (side-by-side, word-count deltas).
`--apply` requires a fresh coherence sign-off (`--skip-coherence` to override),
refuses on a dirty `.tex` (commit/stash first, or `--force`), then splices in
descending-offset order — sentence rewrites first, then coherence fixes, each
of which must match the text exactly once and preserve Class-A LaTeX tokens —
and compiles; on any compile failure it reverts. Revert manually with
`git checkout -- <paper>.tex`.

## Safety model

- **Hash gate**: apply aborts if the `.tex` changed since extraction (offsets are
  only valid against the exact bytes they were cut from).
- **Offset authority**: each edit's original span must still equal the recorded
  text, or it is skipped and logged — never fuzzy-matched.
- **Rewrite validation**: single line, `$`/brace balance preserved, immutable
  LaTeX tokens (math, `\ref`/`\citep`, escaped literals) preserved, no block
  structures, terminal punctuation — before anything is spliced.
- **Coherence gate**: `--apply` refuses without a `coherence_fixes.json`
  fingerprinted to the current accepted set; each fix must match exactly once
  (never fuzzy, never ambiguous) and preserve the Class-A token multiset.
- **Integrity**: paragraph (blank-line) count unchanged, `\begin`/`\end`/`\[`/`\]`
  conserved, then a real `latexmk` compile with revert-on-failure.

## Layout

```
src/rewrite_pipeline/
  scanner.py     masking scanner (math, comments, envs, cmd args, style wrappers)
  extract.py     containers + sentence segmentation + scope filter → manifest
  gemini_fanout.py  async Gemini fan-out (all at once; hedged + infinite retry)
  reinsert.py    verification-first splice + LaTeX-token preservation gate
  coherence.py   cross-sentence gate: paragraph pairs + verbatim fix application
  integrity.py   balance/blank-line invariants + latexmk compile gate
  review.py      side-by-side HTML review artifact
  cli.py         extract / fanout / split / pairs / apply
judge-rewrites.workflow.mjs   Stage-B Fable judge (one pair per agent, Workflow tool)
coherence-sweep.workflow.mjs  pre-apply seam check (one paragraph per agent)
tests/           scanner/extractor round-trip + reinsert/integrity unit tests
```

## Writing a new workflow script

Start every `*.workflow.mjs` in this directory by normalising `args`:

```js
const A = typeof args === 'string' ? JSON.parse(args) : args || {}
```

Depending on the host, `args` arrives either already parsed or as a JSON string —
even when the caller passed a real object. Without this line, `A.some_array` is
`undefined` and the script dies on the first `pipeline()`/`map()` call before
spawning a single agent. All four scripts here do it; a fifth one must too.

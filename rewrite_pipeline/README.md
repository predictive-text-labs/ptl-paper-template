# rewrite_pipeline

Shortens the sentences of `Is_It_Priced_In.tex` to lower reading complexity, in
four stages. Gemini proposes shorter rewrites; **Claude Fable** answers one
question per rewrite — *does it keep all the important details of the original?*
— and keeps it on yes; accepted rewrites are spliced back byte-precisely for
review.

```
extract   paper.tex            → run/sentence_index.json   (deterministic Python)
fanout    in-scope sentences   → run/gemini_out.json       (async Gemini 3.1 Pro)
split     gemini_out.json      → run/judge_batches/*.json  (one sentence pair / file)
judge     batch files          → run/accepted.json         (Claude Fable, Workflow tool)
apply     accepted + manifest  → sidecar + diff + review   (deterministic Python)
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
uv run rewrite fanout           # gemini-3.1-pro-preview, ~250 in-scope sentences
```
Sends each sentence the *verbatim* prompt, **all at once** — no stagger, no
concurrency cap. Gemini 3.1 Pro's limits (22K RPM / 28M TPM) dwarf this ~250-call
job, so the only load-bearing parts are (a) a per-call **timeout** that turns a
stalled HTTP response into a retry, and (b) **infinite retry** on any error (with
a bounded give-up after 20 timeouts so one cursed sentence can't stall the run).
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

### 5. Apply (deterministic; review first)
```bash
uv run rewrite apply             # dry run: sidecar + unified diff + review.html
uv run rewrite apply --apply     # writes the .tex, runs the latexmk compile gate
```
Dry run changes nothing — it writes `Is_It_Priced_In.rewritten.tex`,
`run/rewrite.diff`, and `run/review.html` (side-by-side, word-count deltas).
`--apply` refuses on a dirty `.tex` (commit/stash first, or `--force`), then
splices in descending-offset order, and compiles; on any compile failure it
reverts. Revert manually with `git checkout -- Is_It_Priced_In.tex`.

## Safety model

- **Hash gate**: apply aborts if the `.tex` changed since extraction (offsets are
  only valid against the exact bytes they were cut from).
- **Offset authority**: each edit's original span must still equal the recorded
  text, or it is skipped and logged — never fuzzy-matched.
- **Rewrite validation**: single line, `$`/brace balance preserved, immutable
  LaTeX tokens (math, `\ref`/`\citep`, escaped literals) preserved, no block
  structures, terminal punctuation — before anything is spliced.
- **Integrity**: paragraph (blank-line) count unchanged, `\begin`/`\end`/`\[`/`\]`
  conserved, then a real `latexmk` compile with revert-on-failure.

## Layout

```
src/rewrite_pipeline/
  scanner.py     masking scanner (math, comments, envs, cmd args, style wrappers)
  extract.py     containers + sentence segmentation + scope filter → manifest
  gemini_fanout.py  async Gemini 3.1 Pro fan-out (all at once; timeout + infinite retry)
  reinsert.py    verification-first splice + LaTeX-token preservation gate
  integrity.py   balance/blank-line invariants + latexmk compile gate
  review.py      side-by-side HTML review artifact
  cli.py         extract / fanout / split / apply
judge-rewrites.workflow.mjs   Stage-B Fable judge (one pair per agent, Workflow tool)
tests/           scanner/extractor round-trip + reinsert/integrity unit tests
```

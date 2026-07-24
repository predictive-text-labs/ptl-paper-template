# PTL Paper Template

A minimal NeurIPS 2025 LaTeX paper with automatic build-on-save and an in-editor PDF preview.

## Prerequisites

- [Visual Studio Code](https://code.visualstudio.com/)
- [LaTeX Workshop](https://marketplace.visualstudio.com/items?itemName=James-Yu.latex-workshop)
- [Code Spell Checker](https://marketplace.visualstudio.com/items?itemName=streetsidesoftware.code-spell-checker)
- A TeX distribution that provides `pdflatex` and `latexindent`

The included formatter path is `/opt/homebrew/bin/latexindent`. Update `latex-workshop.formatting.latexindent.path` in `.vscode/settings.json` if `latexindent` is installed elsewhere.

## Live PDF preview

1. Open `Paper-Template.code-workspace` in VS Code.
2. Open `paper.tex`.
3. Run **LaTeX Workshop: View LaTeX PDF** once to open the PDF tab.
4. Save `paper.tex`. LaTeX Workshop rebuilds `paper.pdf`, and the open PDF tab refreshes automatically.

## Command-line build

```bash
pdflatex -interaction=nonstopmode -halt-on-error paper.tex
```

Generated LaTeX files and `paper.pdf` are ignored by Git.

## Sentence-rewrite pipeline

[`rewrite_pipeline/`](rewrite_pipeline/) shortens the paper's sentences to
lower reading complexity while provably keeping every claim, hedge, citation,
and piece of math. Gemini 3.1 Pro proposes shorter rewrites; Claude Fable
judges each one in isolation (*does it keep all the important details?*), then
a second Fable sweep re-reads every changed paragraph whole to catch
cross-sentence damage (an orphaned "still do", a dangling "this", a broken
"However") that per-sentence judging is structurally blind to. Accepted
rewrites are spliced back byte-precisely — offsets verified against a source
hash, LaTeX tokens conserved, `latexmk` compile gate with revert-on-failure.

```bash
cd rewrite_pipeline
uv sync
echo 'GEMINI_API_KEY=<your-ai-studio-key>' >> ../.env   # gitignored
uv run rewrite extract      # sentence manifest with byte-exact offsets
uv run rewrite fanout       # Gemini proposals, all at once
uv run rewrite split        # one judge batch file per sentence pair
# judge + coherence sweep run via Claude Code workflows — see the pipeline README
uv run rewrite apply        # dry run; --apply to write + compile-gate
```

The paper is auto-detected as the repo's sole top-level `.tex` (`paper.tex`
here); pass `--tex` if you add more. See
[`rewrite_pipeline/README.md`](rewrite_pipeline/README.md) for the full
stage-by-stage guide and safety model.

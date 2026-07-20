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

"""A masking scanner for LaTeX prose.

The scanner never mutates the source. It produces a per-character ``visible``
overlay (1 = literal prose text, 0 = masked: math, comments, command tokens,
braces, opaque environments/arguments) plus structural ``markers`` (environment
begin/end, opaque-environment spans, display-math spans, captions, footnotes,
section titles, list items, ``\\appendix``).

Because visibility is an overlay of index-aligned booleans, every span the
extractor emits indexes the *original* string — offsets stay byte-exact, which
is what lets an accepted rewrite be spliced back precisely.

Marker grammar (each marker is a tuple whose first element is a tag):

* ``("begin", name, start, token_end)``       transparent ``\\begin{name}[..]``
* ``("end", name, start, token_end)``          transparent ``\\end{name}``
* ``("opaque_env", name, start, end)``         whole masked opaque environment
* ``("display", start, end)``                  display math ``\\[ ... \\]``
* ``("caption", cmd_start, cstart, cend)``     ``\\caption{...}`` content range
* ``("footnote", cmd_start, cstart, cend)``    ``\\footnote{...}`` content range
* ``("section", level, cmd_start, cstart, cend)`` ``\\section{...}`` etc.
* ``("item", start, content_start)``           ``\\item[..]``
* ``("appendix", start)``                      ``\\appendix``
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Commands whose entire argument group(s) are non-prose and masked wholesale.
OPAQUE_CMDS: frozenset[str] = frozenset(
    {
        "ref",
        "eqref",
        "pageref",
        "autoref",
        "cref",
        "Cref",
        "vref",
        "nameref",
        "cite",
        "citep",
        "citet",
        "citeauthor",
        "citeyear",
        "citealp",
        "citealt",
        "citenum",
        "citeyearpar",
        "label",
        "url",
        "href",
        "includegraphics",
        "input",
        "include",
        "usepackage",
        "documentclass",
        "bibliography",
        "bibliographystyle",
        "nocite",
        "hspace",
        "vspace",
        "rule",
        "S",
        "P",
        "dots",
        "ldots",
        "cdots",
        "vdots",
        "hfill",
        "vfill",
        "toprule",
        "midrule",
        "bottomrule",
    }
)

# No-argument formatting/spacing commands: mask just the token, keep scanning.
BARE_CMDS: frozenset[str] = frozenset(
    {
        "noindent",
        "centering",
        "raggedright",
        "raggedleft",
        "clearpage",
        "newpage",
        "smallskip",
        "medskip",
        "bigskip",
        "maketitle",
        "par",
        "bfseries",
        "itshape",
        "ttfamily",
        "scshape",
        "normalfont",
        "small",
        "footnotesize",
        "large",
        "Large",
        "LARGE",
        "huge",
        "tiny",
    }
)

# Formatting wrappers: the command token and braces are masked, the wrapped text
# stays visible prose (recursed into by the main loop).
STYLE_CMDS: frozenset[str] = frozenset(
    {
        "emph",
        "textbf",
        "textit",
        "texttt",
        "textsc",
        "textrm",
        "textsf",
        "textnormal",
        "underline",
        "mbox",
        "text",
        "uline",
        "so",
        "textup",
        "textmd",
        "boldsymbol",
    }
)

# Environments whose whole body is non-prose: fast-forward to matching \end.
OPAQUE_ENVS: frozenset[str] = frozenset(
    {
        "tikzpicture",
        "tabular",
        "tabular*",
        "tabularx",
        "tabulary",
        "array",
        "aligned",
        "align",
        "align*",
        "alignat",
        "alignat*",
        "gather",
        "gather*",
        "multline",
        "multline*",
        "cases",
        "matrix",
        "pmatrix",
        "bmatrix",
        "vmatrix",
        "Vmatrix",
        "smallmatrix",
        "equation",
        "equation*",
        "eqnarray",
        "eqnarray*",
        "verbatim",
        "Verbatim",
        "lstlisting",
        "minted",
        "tabbing",
        "thebibliography",
        "split",
    }
)

SECTION_CMDS: frozenset[str] = frozenset(
    {
        "section",
        "subsection",
        "subsubsection",
        "paragraph",
        "subparagraph",
        "part",
        "chapter",
    }
)

Marker = tuple


@dataclass
class ScanResult:
    text: str
    visible: bytearray
    markers: list[Marker]
    unknown_cmds: dict[str, int] = field(default_factory=dict)


def _read_control_word(text: str, i: int) -> tuple[str, int]:
    n = len(text)
    j = i + 1
    while j < n and text[j].isalpha():
        j += 1
    return text[i + 1 : j], j


def _find_matching_brace(text: str, open_idx: int) -> int:
    n = len(text)
    depth = 0
    k = open_idx
    while k < n:
        c = text[k]
        if c == "\\":
            k += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return k
        k += 1
    return n


def _find_unescaped_dollar(text: str, start: int) -> int:
    n = len(text)
    k = start
    while k < n:
        if text[k] == "\\":
            k += 2
            continue
        if text[k] == "$":
            return k
        k += 1
    return -1


def _consume_arg_groups(text: str, i: int) -> int:
    n = len(text)
    k = i
    while k < n:
        if text[k] == "{":
            k = _find_matching_brace(text, k) + 1
        elif text[k] == "[":
            close = text.find("]", k)
            k = (close + 1) if close != -1 else n
        else:
            break
    return k


def _mask(visible: bytearray, start: int, end: int) -> None:
    for k in range(max(start, 0), min(end, len(visible))):
        visible[k] = 0


def scan(text: str) -> ScanResult:
    n = len(text)
    visible = bytearray(n)
    markers: list[Marker] = []
    unknown: dict[str, int] = {}

    # Brace frames entered char-by-char. Frame = (kind, content_start, level,
    # cmd_start); kind in {plain, style, caption, footnote, section}.
    stack: list[tuple[str, int, str, int]] = []

    i = 0
    while i < n:
        c = text[i]

        if c == "\\":
            nxt = text[i + 1] if i + 1 < n else ""
            if nxt.isalpha():
                name, after = _read_control_word(text, i)
                star = after < n and text[after] == "*"
                token_end = after + 1 if star else after
                _mask(visible, i, token_end)
                i = _handle_command(
                    text, visible, markers, unknown, stack, name, i, token_end
                )
                continue
            if nxt == "[":  # display math
                close = text.find("\\]", i + 2)
                end = (close + 2) if close != -1 else n
                _mask(visible, i, end)
                markers.append(("display", i, end))
                i = end
                continue
            if nxt == "(":  # inline math (backslash form)
                close = text.find("\\)", i + 2)
                end = (close + 2) if close != -1 else n
                _mask(visible, i, end)
                i = end
                continue
            _mask(visible, i, i + 2)  # escaped control symbol
            i += 2
            continue

        if c == "$":
            if i + 1 < n and text[i + 1] == "$":
                close = text.find("$$", i + 2)
                end = (close + 2) if close != -1 else n
            else:
                close = _find_unescaped_dollar(text, i + 1)
                end = (close + 1) if close != -1 else n
            _mask(visible, i, end)
            i = end
            continue

        if c == "%":
            nl = text.find("\n", i)
            end = nl if nl != -1 else n
            _mask(visible, i, end)
            i = end
            continue

        if c == "{":
            stack.append(("plain", i + 1, "", -1))
            visible[i] = 0
            i += 1
            continue
        if c == "}":
            visible[i] = 0
            if stack:
                kind, cstart, level, cmd_start = stack.pop()
                if kind == "caption":
                    markers.append(("caption", cmd_start, cstart, i))
                elif kind == "footnote":
                    markers.append(("footnote", cmd_start, cstart, i))
                elif kind == "section":
                    markers.append(("section", level, cmd_start, cstart, i))
            i += 1
            continue

        visible[i] = 1  # literal prose (opaque envs are fast-forwarded past)
        i += 1

    return ScanResult(text=text, visible=visible, markers=markers, unknown_cmds=unknown)


def _handle_command(
    text, visible, markers, unknown, stack, name, cmd_start, after_name
) -> int:
    n = len(text)

    if name == "begin":
        brace = after_name
        if brace < n and text[brace] == "{":
            close = _find_matching_brace(text, brace)
            env = text[brace + 1 : close]
            _mask(visible, brace, close + 1)
            if env in OPAQUE_ENVS:
                end = _skip_opaque_env(text, visible, env, close + 1)
                markers.append(("opaque_env", env, cmd_start, end))
                return end
            after = close + 1
            if after < n and text[after] == "[":
                rb = text.find("]", after)
                after = (rb + 1) if rb != -1 else n
                _mask(visible, close + 1, after)
            markers.append(("begin", env, cmd_start, after))
            return after
        return after_name

    if name == "end":
        brace = after_name
        if brace < n and text[brace] == "{":
            close = _find_matching_brace(text, brace)
            env = text[brace + 1 : close]
            _mask(visible, brace, close + 1)
            markers.append(("end", env, cmd_start, close + 1))
            return close + 1
        return after_name

    if name == "caption":
        return _open_special(text, visible, stack, after_name, "caption", "", cmd_start)
    if name == "footnote":
        return _open_special(
            text, visible, stack, after_name, "footnote", "", cmd_start
        )
    if name in SECTION_CMDS:
        return _open_special(
            text, visible, stack, after_name, "section", name, cmd_start
        )

    if name == "item":
        after = after_name
        if after < n and text[after] == "[":
            rb = text.find("]", after)
            after = (rb + 1) if rb != -1 else n
            _mask(visible, after_name, after)
        markers.append(("item", cmd_start, after))
        return after

    if name == "appendix":
        markers.append(("appendix", cmd_start))
        return after_name

    if name in STYLE_CMDS:
        brace = after_name
        if brace < n and text[brace] == "{":
            visible[brace] = 0
            stack.append(("style", brace + 1, "", -1))
            return brace + 1
        return after_name

    if name in OPAQUE_CMDS:
        end = _consume_arg_groups(text, after_name)
        _mask(visible, after_name, end)
        return end

    if name in BARE_CMDS:
        return after_name

    unknown[name] = unknown.get(name, 0) + 1
    return after_name


def _open_special(text, visible, stack, after_name, kind, level, cmd_start) -> int:
    n = len(text)
    after = after_name
    if after < n and text[after] == "[":
        rb = text.find("]", after)
        after = (rb + 1) if rb != -1 else n
        _mask(visible, after_name, after)
    if after < n and text[after] == "{":
        visible[after] = 0
        stack.append((kind, after + 1, level, cmd_start))
        return after + 1
    return after


# Inside these, % is a literal character, not a comment — TeX recognises the
# \end token even on a % line, so their spans must NOT be comment-aware.
_VERBATIM_ENVS: frozenset[str] = frozenset({"verbatim", "Verbatim", "lstlisting", "minted"})


def _commented(text: str, pos: int) -> bool:
    """True if ``pos`` is preceded on its line by an unescaped ``%``."""
    k = text.rfind("\n", 0, pos) + 1
    while k < pos:
        if text[k] == "\\":
            k += 2
            continue
        if text[k] == "%":
            return True
        k += 1
    return False


def _find_active(text: str, tok: str, start: int, comment_aware: bool) -> int:
    """``text.find`` that (when comment-aware) skips matches on commented text."""
    idx = text.find(tok, start)
    while comment_aware and idx != -1 and _commented(text, idx):
        idx = text.find(tok, idx + 1)
    return idx


def _scan_env_end(
    text: str, begin_tok: str, end_tok: str, start: int, aware: bool
) -> int | None:
    """Index just past the matching end token, or None if never closed."""
    n = len(text)
    depth = 1
    k = start
    while k < n and depth > 0:
        nb = _find_active(text, begin_tok, k, aware)
        ne = _find_active(text, end_tok, k, aware)
        if ne == -1:
            return None
        if nb != -1 and nb < ne:
            depth += 1
            k = nb + len(begin_tok)
        else:
            depth -= 1
            k = ne + len(end_tok)
    return k


def _skip_opaque_env(text: str, visible: bytearray, env: str, start: int) -> int:
    begin_tok = "\\begin{" + env + "}"
    end_tok = "\\end{" + env + "}"
    aware = env not in _VERBATIM_ENVS
    k = _scan_env_end(text, begin_tok, end_tok, start, aware)
    if k is None and aware:
        # Every end token sat after a % on its line. A literal percent
        # (\verb|%|, \url{..%..}) is far likelier than a document whose SOLE
        # \end is commented out (that would not compile), so degrade to the
        # comment-blind match rather than masking to end-of-file.
        k = _scan_env_end(text, begin_tok, end_tok, start, False)
    if k is None:
        k = len(text)
    _mask(visible, start, k)
    return k

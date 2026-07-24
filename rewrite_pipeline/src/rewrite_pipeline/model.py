"""Data model for the sentence-rewrite pipeline.

A ``Record`` is one extracted prose sentence with a *byte-exact* span into the
source ``.tex`` file, so that an accepted rewrite can be spliced back into the
exact original location. The ``Manifest`` binds a set of records to a specific
file hash; offsets are only valid against that hash.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

EXTRACTOR_VERSION = "1"

# The kinds of prose container a sentence can live in.
KINDS = ("abstract", "body", "caption", "footnote", "list-item", "heading")


@dataclass(frozen=True)
class Record:
    """One extracted sentence.

    ``text`` is exactly ``source[abs_start:abs_end]`` — the reinserter asserts
    this before it will splice, so the offsets are the sole source of truth.
    """

    id: str
    text: str
    abs_start: int
    abs_end: int
    line_no: int
    kind: str
    in_scope: bool
    n_dollars: int  # count of UNescaped '$' in text; must be even to be in scope
    n_brace_delta: int  # unescaped '{' minus '}'; must be 0 to be in scope
    has_terminal: bool  # ends in [.?!] (+ optional closer)
    contains_footnote: bool  # span overlaps a \footnote{...} (parent excluded)
    excluded_reason: str | None = None  # why not in_scope (None iff in_scope)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Manifest:
    file_path: str
    file_sha256: str
    git_commit: str | None
    extractor_version: str
    records: list[Record]

    def to_json(self) -> str:
        return json.dumps(
            {
                "file_path": self.file_path,
                "file_sha256": self.file_sha256,
                "git_commit": self.git_commit,
                "extractor_version": self.extractor_version,
                "records": [r.to_dict() for r in self.records],
            },
            indent=2,
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, s: str) -> Manifest:
        data = json.loads(s)
        return cls(
            file_path=data["file_path"],
            file_sha256=data["file_sha256"],
            git_commit=data.get("git_commit"),
            extractor_version=data["extractor_version"],
            records=[Record(**r) for r in data["records"]],
        )

    def by_id(self) -> dict[str, Record]:
        return {r.id: r for r in self.records}


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_id(ordinal: int, text: str) -> str:
    """Stable, human-scannable id: ``0007-1a2b3c4d`` (ordinal + content hash)."""
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"{ordinal:04d}-{digest}"

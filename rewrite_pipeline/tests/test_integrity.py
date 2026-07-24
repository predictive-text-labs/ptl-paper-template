"""Compile-gate tests: every failure mode must return (False, log), never raise.

cmd_apply writes the rewritten .tex BEFORE calling compile_check and reverts
only on a falsy return — an escaping exception would skip the revert and leave
the manuscript modified.
"""

from __future__ import annotations

import subprocess

from rewrite_pipeline import integrity


def test_compile_check_timeout_returns_false(monkeypatch, tmp_path):
    monkeypatch.setattr(integrity.shutil, "which", lambda _: "/usr/bin/latexmk")

    def boom(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=["latexmk"], timeout=600, output="partial latexmk output"
        )

    monkeypatch.setattr(integrity.subprocess, "run", boom)
    tex = tmp_path / "paper.tex"
    tex.write_text("x", encoding="utf-8")
    ok, log = integrity.compile_check(tex, tmp_path / "build")
    assert ok is False
    assert "timed out" in log
    assert "partial latexmk output" in log


def test_compile_check_oserror_returns_false(monkeypatch, tmp_path):
    monkeypatch.setattr(integrity.shutil, "which", lambda _: "/usr/bin/latexmk")

    def boom(*args, **kwargs):
        raise OSError("exec format error")

    monkeypatch.setattr(integrity.subprocess, "run", boom)
    tex = tmp_path / "paper.tex"
    tex.write_text("x", encoding="utf-8")
    ok, log = integrity.compile_check(tex, tmp_path / "build")
    assert ok is False
    assert "failed to start" in log


def test_compile_check_missing_latexmk(monkeypatch, tmp_path):
    monkeypatch.setattr(integrity.shutil, "which", lambda _: None)
    ok, log = integrity.compile_check(tmp_path / "paper.tex", tmp_path / "build")
    assert ok is False
    assert "not found" in log


def test_compile_check_unmakeable_outdir_returns_false(monkeypatch, tmp_path):
    monkeypatch.setattr(integrity.shutil, "which", lambda _: "/usr/bin/latexmk")
    tex = tmp_path / "paper.tex"
    tex.write_text("x", encoding="utf-8")
    build = tmp_path / "build"
    build.write_text("i am a file, not a directory", encoding="utf-8")
    ok, log = integrity.compile_check(tex, build)
    assert ok is False
    assert "failed to start" in log


def test_compile_check_non_utf8_output_returns_false(tmp_path, monkeypatch):
    # pdftex logs echo raw source/package bytes; decoding must never raise.
    fake = tmp_path / "latexmk"
    fake.write_bytes(b"#!/bin/sh\nprintf 'bad: \\377\\376'\nexit 1\n")
    fake.chmod(0o755)
    monkeypatch.setattr(integrity.shutil, "which", lambda _: str(fake))
    tex = tmp_path / "paper.tex"
    tex.write_text("x", encoding="utf-8")
    ok, log = integrity.compile_check(tex, tmp_path / "build")
    assert ok is False
    assert "bad:" in log

"""Gemini fan-out — fire every sentence at once.

Sends each in-scope sentence to ``gemini-3.1-pro-preview`` with the *verbatim*
rewrite prompt (no system instruction — faithful to "just this single prompt and
just this single sentence"). Extraction of a clean snippet from the raw response
is deferred to the Claude judging stage.

Dispatch is dead simple, because the account limits (22K RPM / 28M TPM) and the
fd limit (~1M) dwarf a ~250-call job: fire all sentences concurrently, no
stagger, no semaphore. Only two things are load-bearing:
  * a per-call wall-clock timeout. The one real failure mode is a STALLED HTTP
    response — the request is accepted but the server never returns and never
    errors, so httpx waits on a dead socket forever. "Retry on error" can't fix
    that (there is no error); the timeout is what turns a hang into a retry.
  * infinite retry on any error (a stall is random per-connection, so a retry on
    a fresh socket almost always works), with a bounded give-up on repeated
    timeouts so one cursed sentence can't stall the whole run. 400/403/404 are
    terminal (bad request / auth / missing model).
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path

from .model import Manifest, Record

PROMPT_TEMPLATE = (
    "Can the below be rewritten to lower cognitive complexity when reading "
    "**using less words**? If so, give me a rewrite. Keep every defined "
    "term, hedge, quantifier, citation, and math exactly as written.\n"
    "{sentence}"
)

MODEL = "gemini-3.1-pro-preview"
TERMINAL_CODES = frozenset({400, 403, 404})
# A single call is aborted after this long. Legit high-thinking calls run up to
# ~55s under load, so this is generous headroom — anything longer is a stall.
REQUEST_TIMEOUT_S = 120.0
# Brief pause before retrying a failed call (limits are effectively unlimited, so
# this only paces retries; a 503 "high load" wants a moment for the model to
# recover before the next attempt).
RETRY_DELAY_S = 2.0
# Errors retry forever; a call that keeps *timing out* is rare (needs many
# independent stalls in a row) — retry hard (20x), then give up and record
# status="timeout" so the run can always finish.
MAX_TIMEOUTS = 20


@dataclass
class GeminiRecord:
    id: str
    kind: str
    original_sentence: str
    gemini_raw_response: str | None
    status: str  # ok | terminal_<code> | blocked_* | empty_* | no_candidates
    context_before: str
    context_after: str
    attempts: int

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "original_sentence": self.original_sentence,
            "gemini_raw_response": self.gemini_raw_response,
            "status": self.status,
            "context_before": self.context_before,
            "context_after": self.context_after,
            "attempts": self.attempts,
        }


def load_env(env_path: Path | None = None) -> None:
    """Load the repo-root ``.env`` so ``GEMINI_API_KEY`` reaches the SDK."""
    from dotenv import load_dotenv

    if env_path is not None:
        load_dotenv(env_path)
    else:
        # search upward from CWD, then try the repo root next to this package.
        from dotenv import find_dotenv

        found = find_dotenv(usecwd=True)
        load_dotenv(found or None)


def _neighbors(records: list[Record]) -> dict[str, tuple[str, str]]:
    """Adjacent sentence text (document order) for grammatical drop-in context."""
    ctx: dict[str, tuple[str, str]] = {}
    for idx, r in enumerate(records):
        before = records[idx - 1].text if idx > 0 else ""
        after = records[idx + 1].text if idx + 1 < len(records) else ""
        ctx[r.id] = (before, after)
    return ctx


def _extract_text(resp: object) -> tuple[str | None, str]:
    """Pull the model text out of a response, returning (text_or_None, status).

    Treats ``.text`` as optional (safety block, MAX_TOKENS, thinking-only)."""
    pf = getattr(resp, "prompt_feedback", None)
    if pf is not None and getattr(pf, "block_reason", None):
        return None, f"blocked_{getattr(pf.block_reason, 'name', pf.block_reason)}"
    cands = getattr(resp, "candidates", None) or []
    if not cands:
        return None, "no_candidates"
    cand = cands[0]
    text: str | None
    try:
        text = resp.text  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - SDK .text can raise on odd responses; treat as empty
        text = None
    if not text:
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", None) or []
        collected = "".join(
            p.text
            for p in parts
            if getattr(p, "text", None) and not getattr(p, "thought", False)
        )
        text = collected or None
    if not text:
        fr = getattr(cand, "finish_reason", None)
        return None, f"empty_{getattr(fr, 'name', fr)}"
    return text, "ok"


async def _call_one(
    client,
    cfg,
    rec: Record,
    ctx: tuple[str, str],
    model: str,
) -> GeminiRecord:
    from google.genai import errors as gerrors

    try:
        import httpx

        transient_http: tuple[type[BaseException], ...] = (
            httpx.TimeoutException,
            httpx.TransportError,
        )
    except ImportError:  # pragma: no cover
        transient_http = ()

    before, after = ctx
    prompt = PROMPT_TEMPLATE.format(sentence=rec.text)

    def record(status: str, text: str | None, attempt: int) -> GeminiRecord:
        return GeminiRecord(
            id=rec.id,
            kind=rec.kind,
            original_sentence=rec.text,
            gemini_raw_response=text,
            status=status,
            context_before=before,
            context_after=after,
            attempts=attempt,
        )

    attempt = 0
    timeouts = 0
    while True:
        attempt += 1
        try:
            resp = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=model, contents=prompt, config=cfg
                ),
                timeout=REQUEST_TIMEOUT_S,
            )
            text, status = _extract_text(resp)
            return record(status, text, attempt)
        except TimeoutError:
            # A stalled response. Retry hard, but give up after MAX_TIMEOUTS so
            # one cursed sentence can't stall the whole run.
            timeouts += 1
            if timeouts >= MAX_TIMEOUTS:
                return record("timeout", None, attempt)
        except gerrors.APIError as e:
            # ClientError (4xx) and ServerError (5xx) both subclass APIError.
            code = getattr(e, "code", None)
            if code in TERMINAL_CODES:
                return record(f"terminal_{code}", None, attempt)
            # 429 / 5xx / everything else: retry forever.
        except transient_http:
            pass  # network blip: retry forever.
        await asyncio.sleep(RETRY_DELAY_S)


def preflight(client, model: str = MODEL) -> bool:
    """Confirm the model id is live for this key. Returns True if listed; prints
    a warning (but does not raise) if not — preview ids are not always listed."""
    target = model if model.startswith("models/") else f"models/{model}"
    found = []
    g3 = []
    try:
        for m in client.models.list():
            name = getattr(m, "name", "")
            found.append(name)
            if "gemini-3" in name:
                g3.append(name)
    except Exception as e:  # noqa: BLE001 - network/API errors are non-fatal for preflight
        print(f"[preflight] could not list models: {e}")
        return False
    ok = target in found
    if ok:
        print(f"[preflight] OK: {target} is available.")
    else:
        print(
            f"[preflight] WARNING: {target} not in ListModels. Gemini-3 models seen: {g3}"
        )
        print(
            "[preflight] proceeding anyway; missing-model calls will return terminal_404."
        )
    return ok


async def fanout(
    records: list[Record],
    all_records: list[Record],
    *,
    model: str = MODEL,
) -> list[GeminiRecord]:
    from google import genai
    from google.genai import types

    client = genai.Client()  # reads GEMINI_API_KEY / GOOGLE_API_KEY
    preflight(client, model)

    cfg = types.GenerateContentConfig(
        # Gemini 3 Pro reasons at the highest setting; output budget left unset
        # so it uses the model's full maximum (thinking tokens draw from it).
        thinking_config=types.ThinkingConfig(thinking_level="high"),
    )
    ctx = _neighbors(all_records)

    done = 0
    total = len(records)
    results: list[GeminiRecord] = []

    async def worker(rec: Record) -> None:
        nonlocal done
        gr = await _call_one(client, cfg, rec, ctx.get(rec.id, ("", "")), model)
        results.append(gr)
        done += 1
        if done % 25 == 0 or done == total:
            print(f"[fanout] {done}/{total} done")

    # Every sentence in flight at once — no stagger, no semaphore.
    await asyncio.gather(*(worker(r) for r in records))
    return results


def run(
    manifest: Manifest,
    out_path: Path,
    *,
    model: str = MODEL,
    env_path: Path | None = None,
) -> list[GeminiRecord]:
    load_env(env_path)
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        raise SystemExit(
            "GEMINI_API_KEY not found. Put it in .env (GEMINI_API_KEY=...)."
        )
    in_scope = [r for r in manifest.records if r.in_scope]
    print(f"[fanout] {len(in_scope)} in-scope sentences → {model}")
    results = asyncio.run(fanout(in_scope, manifest.records, model=model))
    results.sort(key=lambda g: g.id)
    out_path.write_text(
        json.dumps([g.to_dict() for g in results], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    ok = sum(1 for g in results if g.status == "ok")
    print(f"[fanout] wrote {out_path}  (ok={ok}/{len(results)})")
    return results

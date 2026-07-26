"""Gemini fan-out — fire every sentence at once.

Sends each in-scope sentence to ``gemini-3.6-flash`` with the *verbatim*
rewrite prompt (no system instruction — faithful to "just this single prompt and
just this single sentence"). Extraction of a clean snippet from the raw response
is deferred to the Claude judging stage.

Dispatch is dead simple, because the account limits (22K RPM / 28M TPM) and the
fd limit (~1M) dwarf the job: fire all sentences concurrently, no stagger, no
semaphore. Staggering was measured and rejected — a 950-call instantaneous burst
drew zero 429s and zero 503s, so there is no server-side pushback to pace away,
and a 0.1s stagger would add ~95s to the floor before the last call even starts.
Only two things are load-bearing:
  * a per-copy hedge timer. The one real failure mode is a STALLED HTTP response
    — the request is accepted but the server never returns and never errors, so
    httpx waits on a dead socket forever. "Retry on error" cannot fix that
    (there is no error), and elapsed time is the only available detector. But a
    slow call and a stalled call look identical until one of them answers, so
    instead of killing a slow copy and starting from zero, a DUPLICATE is raced
    against it and the first answer wins: a stall is per-connection, so a fresh
    socket almost always works, while the original stays alive in case it was
    merely queued.
  * infinite retry on any error, bounded by one hard per-call deadline.
    400/401/403/404 are terminal (bad request / auth / missing model — never
    transient, so a retry loop would hang every worker).
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from .model import Manifest, Record

PROMPT_TEMPLATE = (
    "Can the below be rewritten to lower cognitive complexity when reading "
    "**using less words**? If so, give me a rewrite. Keep every defined "
    "term, hedge, quantifier, citation, and math exactly as written.\n"
    "{sentence}"
)

MODEL = "gemini-3.6-flash"
TERMINAL_CODES = frozenset({400, 401, 403, 404})
# How long one copy of a call may be outstanding before a duplicate is raced
# against it. Measured, not guessed: a burst this deep drains steadily out to
# ~111s (in a 950-sentence reference run, 802 done by 75s and 900 by 111s), so a
# tighter trigger would hedge calls that are merely queued — 16% of them at 75s.
# Past ~120s a call is a genuine outlier: 96% of that run answered on the first
# copy, so hedging stays cheap. Those numbers come from gemini-3.1-pro-preview;
# they are a ceiling for the Flash default, which is faster per call at the same
# thinking_level, so the trigger stays conservative rather than needing a re-tune.
HEDGE_AFTER_S = 120.0
# Ceiling on concurrent copies of ONE logical call, so a pathological sentence
# cannot fan out without bound.
MAX_HEDGES = 4
# Brief pause before replacing a failed call (limits are effectively unlimited,
# so this only paces retries; a 503 "high load" wants a moment for the model to
# recover before the next attempt).
RETRY_DELAY_S = 2.0
# Hard give-up for one logical call — all copies — after which the record lands
# as status="timeout" so the run can always finish. This replaces a per-attempt
# timeout cap, which multiplied the wall clock (20 attempts x 120s = 40 min for
# a single cursed sentence) precisely because each attempt discarded a copy that
# might still have answered.
CALL_DEADLINE_S = 600.0
# The real ceiling on "fire everything at once" is NOT the account rate limit
# (22K RPM / 28M TPM) or the fd limit (~1M soft) — it is httpx's own connection
# pool, which defaults to max_connections=100. Above that, every extra call
# blocks in pool acquisition; a queued call that its caller then cancels is
# cancelled *mid-acquire* and does not reliably hand its slot back, so the pool
# bleeds slots into CLOSE_WAIT until every worker is parked on a pool that can
# never refill — a permanent hang with 0 sockets in flight. Sizing the pool
# above the job means nothing ever queues, so there is no cancel-during-acquire
# to leak in the first place.
MAX_CONNECTIONS = 1500
# Progress is reprinted at least this often even when nothing completes, so a
# stalled run looks stalled instead of looking like a silent hang.
HEARTBEAT_S = 15.0


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


@dataclass
class _Stats:
    """Live fan-out counters, shared by every worker and the progress printer."""

    total: int
    done: int = 0
    ok: int = 0
    inflight: int = 0
    retries: int = 0
    hedges: int = 0
    timeouts: int = 0

    def line(self, elapsed: float) -> str:
        bad = self.done - self.ok
        return (
            f"[fanout] {self.done}/{self.total} done"
            f" | {self.ok} ok, {bad} failed"
            f" | {self.inflight} in flight"
            f" | {self.retries} retries, {self.hedges} hedged, {self.timeouts} timeouts"
            f" | {elapsed:.0f}s"
        )


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
    stats: _Stats | None = None,
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
    inflight: set[asyncio.Task] = set()
    started = time.monotonic()

    def spawn() -> None:
        """Put one more copy of this call on the wire."""
        nonlocal attempt
        attempt += 1
        inflight.add(
            asyncio.create_task(
                client.aio.models.generate_content(
                    model=model, contents=prompt, config=cfg
                )
            )
        )

    try:
        spawn()
        while True:
            left = CALL_DEADLINE_S - (time.monotonic() - started)
            if left <= 0:
                if stats is not None:
                    stats.timeouts += 1
                return record("timeout", None, attempt)

            # Wake on the first copy to answer, or when the next hedge is due.
            done, _ = await asyncio.wait(
                inflight,
                timeout=min(HEDGE_AFTER_S, left),
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in done:
                inflight.discard(task)
                exc = task.exception()
                if exc is None:
                    text, status = _extract_text(task.result())
                    return record(status, text, attempt)
                if isinstance(exc, gerrors.APIError):
                    # ClientError (4xx) and ServerError (5xx) both subclass it.
                    code = getattr(exc, "code", None)
                    if code in TERMINAL_CODES:
                        return record(f"terminal_{code}", None, attempt)
                    # 429 / 5xx / everything else: retry forever.
                elif not isinstance(exc, transient_http):
                    raise exc  # a real bug, not a network condition
                if stats is not None:
                    stats.retries += 1

            if not inflight:
                # Every copy failed — pause, then start a fresh one.
                await asyncio.sleep(min(RETRY_DELAY_S, left))
                spawn()
            elif not done and len(inflight) < MAX_HEDGES:
                # Nobody answered within HEDGE_AFTER_S and the survivors may be
                # stalled on dead sockets. Race a duplicate rather than killing
                # them: whichever returns first wins, so a copy that was merely
                # slow still counts.
                if stats is not None:
                    stats.hedges += 1
                spawn()
    finally:
        # Losing copies must be cancelled AND awaited, or asyncio reports their
        # results as never-retrieved once they eventually settle.
        for task in inflight:
            task.cancel()
        if inflight:
            await asyncio.gather(*inflight, return_exceptions=True)


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


def make_client(max_connections: int = MAX_CONNECTIONS):
    """A genai client whose httpx pool is sized above the whole job.

    Without this the pool caps at 100 and the fan-out deadlocks (see
    ``MAX_CONNECTIONS``). ``google-genai`` only uses aiohttp when it is
    installed; the default async path is ``httpx.AsyncClient``, so the pool is
    set through ``async_client_args``.
    """
    import httpx
    from google import genai
    from google.genai import types

    limits = httpx.Limits(
        max_connections=max_connections,
        max_keepalive_connections=max_connections,
    )
    # reads GEMINI_API_KEY / GOOGLE_API_KEY
    return genai.Client(
        http_options=types.HttpOptions(async_client_args={"limits": limits})
    )


async def fanout(
    records: list[Record],
    all_records: list[Record],
    *,
    model: str = MODEL,
) -> list[GeminiRecord]:
    from google.genai import types

    client = make_client()
    preflight(client, model)

    cfg = types.GenerateContentConfig(
        # thinking_level is an EFFORT dial (minimal/low/medium/high), not a token
        # allowance: the model still spends dynamically inside it — the same call
        # at "high" measured 6.6k / 8.1k / 11.7k thinking tokens on three
        # identical tries. Asking for "high" is not redundant, because
        # gemini-3.6-flash defaults to "medium": ~8.8k mean at high vs ~4.8k
        # unset. Leave thinking_budget alone — it is the legacy numeric knob,
        # deprecated for Gemini 3.x, and setting both is a hard 400 ("You can
        # only set only one of thinking budget and thinking level"). No
        # max_output_tokens either, so the model keeps its full ceiling.
        thinking_config=types.ThinkingConfig(thinking_level="high"),
    )
    ctx = _neighbors(all_records)

    stats = _Stats(total=len(records))
    results: list[GeminiRecord] = []
    started = time.monotonic()

    def emit() -> None:
        # flush=True is load-bearing: stdout is block-buffered whenever it is
        # not a TTY (a pipe, a log file, a background task), so an unflushed
        # print can sit in an 8 KB buffer for the entire run and appear only at
        # exit — which is indistinguishable from a hang.
        print(stats.line(time.monotonic() - started), flush=True)

    async def heartbeat() -> None:
        while True:
            await asyncio.sleep(HEARTBEAT_S)
            emit()

    async def worker(rec: Record) -> None:
        stats.inflight += 1
        try:
            gr = await _call_one(
                client, cfg, rec, ctx.get(rec.id, ("", "")), model, stats
            )
        finally:
            stats.inflight -= 1
        results.append(gr)
        stats.done += 1
        if gr.status == "ok":
            stats.ok += 1
        if stats.done % 25 == 0 or stats.done == stats.total:
            emit()

    # Every sentence in flight at once — no stagger, no semaphore.
    emit()
    beat = asyncio.create_task(heartbeat())
    try:
        await asyncio.gather(*(worker(r) for r in records))
    finally:
        beat.cancel()
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
    print(
        f"[fanout] {len(in_scope)} in-scope sentences → {model} "
        f"(pool={MAX_CONNECTIONS}, hedge={HEDGE_AFTER_S:.0f}s"
        f"×{MAX_HEDGES}, deadline={CALL_DEADLINE_S:.0f}s)",
        flush=True,
    )
    results = asyncio.run(fanout(in_scope, manifest.records, model=model))
    results.sort(key=lambda g: g.id)
    out_path.write_text(
        json.dumps([g.to_dict() for g in results], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    ok = sum(1 for g in results if g.status == "ok")
    if ok != len(results):
        from collections import Counter

        bad = Counter(g.status for g in results if g.status != "ok")
        print(f"[fanout] non-ok statuses: {dict(bad)}", flush=True)
    print(f"[fanout] wrote {out_path}  (ok={ok}/{len(results)})", flush=True)
    return results

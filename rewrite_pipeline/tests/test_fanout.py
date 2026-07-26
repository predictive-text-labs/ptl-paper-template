"""Fan-out dispatch tests: the hedge race, on a fake client (no network).

A stalled HTTP response is the fan-out's only real failure mode, and it is the
one thing that cannot be reproduced against the live API on demand — so it is
faked here. Every test drives ``_call_one`` with scripted per-copy behaviours and
shrinks the timers, so the whole file runs in well under a second.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from google.genai import errors as gerrors

from rewrite_pipeline import gemini_fanout as gf
from rewrite_pipeline.model import Record


class FakeResp:
    """The minimum shape ``_extract_text`` reads off a real response."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.candidates = [SimpleNamespace(content=None, finish_reason="STOP")]
        self.prompt_feedback = None


async def stall() -> FakeResp:
    """A request the server accepted and will never answer."""
    await asyncio.sleep(3600)
    raise AssertionError("unreachable")  # pragma: no cover


async def answers(text: str = "Short rewrite.") -> FakeResp:
    return FakeResp(text)


class FakeClient:
    """Serves ``behaviours[i]`` to the i-th copy; the last one repeats."""

    def __init__(self, *behaviours) -> None:
        self.behaviours = list(behaviours)
        self.calls = 0
        self.cancelled = 0
        self.aio = SimpleNamespace(models=self)

    async def generate_content(self, *, model, contents, config):
        which = self.behaviours[min(self.calls, len(self.behaviours) - 1)]
        self.calls += 1
        try:
            return await which()
        except asyncio.CancelledError:
            self.cancelled += 1
            raise


@pytest.fixture(autouse=True)
def fast_timers(monkeypatch):
    """Real timings are 120s/600s; scale them down so tests stay instant."""
    monkeypatch.setattr(gf, "HEDGE_AFTER_S", 0.05)
    monkeypatch.setattr(gf, "CALL_DEADLINE_S", 0.6)
    monkeypatch.setattr(gf, "RETRY_DELAY_S", 0.01)


def record(text: str = "A sentence that is long enough here.") -> Record:
    return Record(
        id="s0001",
        text=text,
        abs_start=0,
        abs_end=len(text),
        line_no=1,
        kind="body",
        in_scope=True,
        n_dollars=0,
        n_brace_delta=0,
        has_terminal=True,
        contains_footnote=False,
    )


async def call(client, stats=None):
    return await gf._call_one(client, None, record(), ("", ""), gf.MODEL, stats)


def test_hedge_wins_when_the_first_copy_stalls():
    """The whole point: a stalled copy must not cost the call its result."""
    client = FakeClient(stall, answers)
    stats = gf._Stats(total=1)
    got = asyncio.run(call(client, stats))
    assert got.status == "ok"
    assert got.gemini_raw_response == "Short rewrite."
    assert got.attempts == 2  # original + one hedge
    assert stats.hedges == 1
    assert stats.timeouts == 0  # a hedged win is not a timeout


def test_stalled_copies_are_cancelled_not_leaked():
    """The loser of the race is cancelled and awaited, so nothing dangles."""
    client = FakeClient(stall, answers)

    async def go():
        got = await call(client)
        # Nothing left running but the test's own task.
        assert len(asyncio.all_tasks()) == 1
        return got

    assert asyncio.run(go()).status == "ok"
    assert client.cancelled == 1


def test_hedges_are_capped_and_the_deadline_gives_up():
    """Every copy stalls: bounded fan-out, then status='timeout'."""
    client = FakeClient(stall)
    stats = gf._Stats(total=1)
    got = asyncio.run(call(client, stats))
    assert got.status == "timeout"
    assert got.gemini_raw_response is None
    assert client.calls <= gf.MAX_HEDGES
    assert stats.timeouts == 1


def test_terminal_error_does_not_retry():
    """400/401/403/404 are permanent — retrying them would hang the worker."""

    async def bad_request():
        raise gerrors.ClientError(400, {"error": {"message": "nope"}})

    client = FakeClient(bad_request)
    got = asyncio.run(call(client))
    assert got.status == "terminal_400"
    assert client.calls == 1


def test_transient_error_is_replaced_by_a_fresh_copy():
    """A 503 costs one copy, not the call."""

    async def overloaded():
        raise gerrors.ServerError(503, {"error": {"message": "overloaded"}})

    client = FakeClient(overloaded, answers)
    stats = gf._Stats(total=1)
    got = asyncio.run(call(client, stats))
    assert got.status == "ok"
    assert stats.retries == 1
    assert stats.hedges == 0  # a replacement is not a hedge


def test_fast_call_is_never_hedged():
    """96% of a real run answers on the first copy — it must stay one call."""
    client = FakeClient(answers)
    stats = gf._Stats(total=1)
    got = asyncio.run(call(client, stats))
    assert got.status == "ok"
    assert got.attempts == 1
    assert client.calls == 1
    assert stats.hedges == 0

"""CPU tests for ``slime.agent.sandbox.E2BSandbox.read_file`` error handling.

``read_file`` returned ``""`` for every exception, so a sandbox that died or a
read whose transient retries were exhausted looked exactly like an empty file.
Graders parse that empty content as "no tests passed" / "not resolved" and
record reward 0 for the model, which then trains on an infrastructure failure.
Only a missing file reads as ``""`` now; every other failure propagates so the
rollout is aborted instead of scored.

The SDK is not imported: the fakes below reproduce its exception class names,
which is what ``E2BSandbox`` dispatches on.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from slime.agent.sandbox import E2BSandbox

NUM_GPUS = 0


class SandboxException(Exception):
    pass


class NotFoundException(SandboxException):
    pass


class FileNotFoundException(NotFoundException):
    pass


class SandboxNotFoundException(NotFoundException):
    pass


class TimeoutException(SandboxException):
    pass


class ReadError(Exception):
    pass


class _FakeFiles:
    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = 0

    async def read(self, path, *, user="root"):
        self.calls += 1
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


def _sandbox(outcome, rpc_retries=3):
    sb = E2BSandbox("fake-image", rpc_retries=rpc_retries)
    sb.rpc_backoff_base_sec = 0.0  # no real backoff sleeps between retries
    files = _FakeFiles(outcome)
    sb._sb = SimpleNamespace(files=files)
    return sb, files


@pytest.mark.unit
def test_read_file_returns_content():
    sb, _ = _sandbox('{"tests": []}')
    assert asyncio.run(sb.read_file("/tmp/result.json")) == '{"tests": []}'


@pytest.mark.unit
@pytest.mark.parametrize("exc_cls", [FileNotFoundException, NotFoundException])
def test_missing_file_reads_as_empty(exc_cls):
    sb, files = _sandbox(exc_cls("path not found"))
    assert asyncio.run(sb.read_file("/tmp/missing.json")) == ""
    assert files.calls == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    "exc",
    [
        SandboxNotFoundException("sandbox abc not found"),
        TimeoutException("sandbox timed out"),
        SandboxException("502: bad gateway"),
    ],
    ids=["sandbox-gone", "sandbox-timeout", "envd-error"],
)
def test_sandbox_failure_is_not_read_as_empty(exc):
    sb, _ = _sandbox(exc)
    with pytest.raises(type(exc)):
        asyncio.run(sb.read_file("/tmp/result.json"))


@pytest.mark.unit
def test_exhausted_transient_retries_are_not_read_as_empty():
    sb, files = _sandbox(ReadError("connection reset"), rpc_retries=3)
    with pytest.raises(ReadError):
        asyncio.run(sb.read_file("/tmp/result.json"))
    assert files.calls == 3


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

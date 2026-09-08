import os
import threading
import time

from app.runtime.model_process import DirectRunner, ModelProcessRunner


def _return_pid_target() -> int:
    return os.getpid()


def _typed_target() -> tuple[int, str]:
    return (7, "result")


def _fail_target() -> None:
    raise RuntimeError("secret token leaked in message")


def _sleep_target(seconds: float) -> str:
    time.sleep(seconds)
    return "done"


def _exit_without_envelope_target() -> None:
    os._exit(1)


def test_model_factory_runs_in_child_pid() -> None:
    runner = ModelProcessRunner(timeout_seconds=10)

    outcome = runner.run(target=_return_pid_target)

    assert outcome.ok
    assert outcome.result == outcome.child_pid
    assert outcome.child_pid != os.getpid()
    assert outcome.exit_code == 0


def test_typed_result_returns_to_parent() -> None:
    runner = ModelProcessRunner(timeout_seconds=10)

    outcome = runner.run(target=_typed_target)

    assert outcome.ok
    assert outcome.result == (7, "result")


def test_child_exception_returns_bounded_envelope_without_traceback() -> None:
    runner = ModelProcessRunner(timeout_seconds=10)

    outcome = runner.run(target=_fail_target)

    assert outcome.ok is False
    assert outcome.error is not None
    assert "RuntimeError" in outcome.error
    assert "Traceback" not in outcome.error


def test_timeout_terminates_only_the_exact_child() -> None:
    runner = ModelProcessRunner(timeout_seconds=1)

    outcome = runner.run(target=_sleep_target, args=(30,))

    assert outcome.ok is False
    assert "timed out" in (outcome.error or "")
    assert outcome.child_pid is not None


def test_parent_reaps_child_before_replying() -> None:
    runner = ModelProcessRunner(timeout_seconds=10)

    outcome = runner.run(target=_return_pid_target)

    assert outcome.exit_code is not None


def test_direct_runner_executes_in_process() -> None:
    runner = DirectRunner()

    outcome = runner.run(target=_return_pid_target)

    assert outcome.ok
    assert outcome.result == os.getpid()


def test_child_exit_without_envelope_is_detected_promptly() -> None:
    """SIGKILL/OOM child death is reaped promptly, not after the full timeout."""

    runner = ModelProcessRunner(timeout_seconds=60)
    started = time.monotonic()

    outcome = runner.run(target=_exit_without_envelope_target)
    elapsed = time.monotonic() - started

    assert outcome.ok is False
    assert "without a result" in (outcome.error or "")
    assert outcome.exit_code == 1
    assert elapsed < 10.0


def test_child_peak_rss_is_reported() -> None:
    runner = ModelProcessRunner(timeout_seconds=10)

    outcome = runner.run(target=_return_pid_target)

    assert outcome.ok
    assert outcome.child_peak_rss_bytes > 0


def test_timeout_reports_unknown_peak_rather_than_invented_zero() -> None:
    """A child that cannot report its peak yields UNKNOWN (None), never 0."""

    runner = ModelProcessRunner(timeout_seconds=1)

    outcome = runner.run(target=_sleep_target, args=(30,))

    assert outcome.ok is False
    assert "timed out" in (outcome.error or "")
    assert outcome.child_peak_rss_bytes is None


def test_abnormal_exit_reports_unknown_peak_rather_than_invented_zero() -> None:
    """A child that dies without an envelope yields UNKNOWN (None), never 0."""

    runner = ModelProcessRunner(timeout_seconds=60)

    outcome = runner.run(target=_exit_without_envelope_target)

    assert outcome.ok is False
    assert outcome.child_peak_rss_bytes is None


def test_cancel_event_terminates_and_reaps_child_immediately() -> None:
    """Lease-loss cancellation stops the active child and reports the reason."""

    runner = ModelProcessRunner(timeout_seconds=60)
    cancel = threading.Event()

    def trigger() -> None:
        time.sleep(0.2)
        cancel.set()

    threading.Thread(target=trigger).start()
    started = time.monotonic()
    outcome = runner.run(target=_sleep_target, args=(60,), cancel_event=cancel)
    elapsed = time.monotonic() - started

    assert outcome.ok is False
    assert "lease loss" in (outcome.error or "")
    assert outcome.child_peak_rss_bytes is None
    assert elapsed < 10.0

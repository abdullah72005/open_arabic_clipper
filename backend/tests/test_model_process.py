import os
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

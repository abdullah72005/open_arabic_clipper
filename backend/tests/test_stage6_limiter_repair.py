"""Stage 6 bounded peak-limiter repair: real FFmpeg regression.

A hot source makes the ideal linear -16 LUFS target infeasible without exceeding
the true-peak guard, so the render fails closed with ``AUDIO_PEAK_REPAIR_NEEDED``
(and a partial measurement manifest). The single authorized bounded limiter
repair then starts from the *lossless* unmastered mix, produces a different
limited intermediate, and a re-run with the authorized reduction satisfies the
final decoded loudness/true-peak QC. Synthetic fixture media, not acceptance.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from test_stage6_authored_render import (
    _AUTHORED_ASS,
    _CANONICAL_ASS,
    _run,
    _runtime,
    _spec,
)

from app.core.settings import get_settings
from app.render.execution.audio import LIMITED_RELATIVE_NAME
from app.render.execution.compiler import compile_render
from app.render.execution.qc import check_render_artifact
from app.render.execution.runner import RenderProcessError, run_compiled_render
from app.render.execution.types import AttemptContext


@pytest.fixture  # type: ignore[untyped-decorator]
def hot_workspace(tmp_path: Path) -> Path:
    ffmpeg = get_settings().ffmpeg_binary
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    source = attempt / "source.mp4"
    # A deliberately hot source: a full-scale tone boosted so the ideal linear
    # target cannot be met under the true-peak guard.
    _run(
        [
            ffmpeg,
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=360x640:rate=30:duration=6",
            "-f",
            "lavfi",
            "-i",
            "aevalsrc='0.9*sin(2*PI*1000*t)*lt(mod(t,1),0.025)':s=48000:d=6",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(source),
        ]
    )
    narration = attempt / "narr.wav"
    _run(
        [
            ffmpeg,
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=700:sample_rate=48000:duration=1.0",
            "-c:a",
            "pcm_s16le",
            str(narration),
        ]
    )
    (attempt / "captions.ass").write_text(_CANONICAL_ASS, encoding="utf-8")
    return attempt


def _context(workspace: Path) -> AttemptContext:
    return AttemptContext(
        attempt_directory=workspace,
        ass_bytes=_CANONICAL_ASS.encode("utf-8"),
        authored_ass_bytes=_AUTHORED_ASS.encode("utf-8"),
        timeout_seconds=180.0,
    )


def _with_repair(spec: Any, reduction_db: float) -> Any:
    payload = spec.authored
    assert payload is not None
    mastering = replace(payload.mastering, repair_peak_limit_db=reduction_db)
    return replace(spec, authored=replace(payload, mastering=mastering))


def test_bounded_limiter_repair_survives_to_final_qc(hot_workspace: Path) -> None:
    source = hot_workspace / "source.mp4"
    narration = hot_workspace / "narr.wav"
    base = _spec(source, narration)

    # 1. Without the repair the hot source is infeasible: fail closed, no MP4,
    #    carrying the partial measurement manifest the finalizer needs.
    compiled = compile_render(base, _runtime(source, hot_workspace))
    with pytest.raises(RenderProcessError) as raised:
        run_compiled_render(compiled, _context(hot_workspace))
    assert raised.value.reason_code == "AUDIO_PEAK_REPAIR_NEEDED"
    assert "mix_measurement" in raised.value.manifest

    # 2. The authorized bounded repair starts from the lossless unmastered mix and
    #    changes the intermediate identity, then satisfies the final decoded QC.
    repaired = _with_repair(base, 3.0)
    compiled_repaired = compile_render(repaired, _runtime(source, hot_workspace))
    artifacts = run_compiled_render(compiled_repaired, _context(hot_workspace))

    limited = hot_workspace / LIMITED_RELATIVE_NAME
    assert limited.is_file()
    unmastered = hot_workspace / "mix-unmastered.wav"
    assert unmastered.is_file()
    assert limited.read_bytes() != unmastered.read_bytes()

    qc = check_render_artifact(
        artifacts, artifacts.manifest, get_settings().stage52_config(), source_path=source
    )
    assert qc.status in {"PASS", "WARN"}, qc.as_dict()
    assert -17.0 <= float(qc.measured["final_integrated_lufs"]) <= -15.0
    assert float(qc.measured["final_true_peak_dbtp"]) <= -1.5 + 0.05

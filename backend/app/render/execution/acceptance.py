"""Stage 5.2 synthetic-regression and production manual-acceptance builders.

Two explicitly separated modes, each with its own entry point so accidental
mislabelling is difficult:

``SYNTHETIC_REGRESSION``
    Hand-built :class:`RenderSpec` objects with invented text/timings. Useful
    only for deterministic rendering demonstrations. Written under
    ``storage/benchmarks/stage-5-2/synthetic-regression`` and clearly labelled
    synthetic; never counted toward the final product gate.

``PRODUCTION_MANUAL_ACCEPTANCE``
    Genuine persisted upstream evidence only. Requires a current retained
    candidate, a current executable Stage 5.0 render contract, a current ready
    Stage 5.1 visual-composition plan with its canonical ASS, and an
    audio-backed ``FINAL_CLIP`` Stage 3.5 refinement. The request is built by the
    production adapter (``build_render_spec``), executed through the normal
    queue/executor, and exported with full lineage under
    ``storage/benchmarks/stage-5-2/manual-acceptance``. It fails closed when
    prerequisites are absent and never falls back to synthetic evidence.

This module is an operational script, not a pipeline stage. It uses the
production adapter/queue/executor and never mutates upstream rows.

Usage::

    python -m app.render.execution.acceptance synthetic-regression \
        --source <media> --output <dir>
    python -m app.render.execution.acceptance production-acceptance \
        --candidate <uuid> --output <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from app.composition.ass import serialize_ass
from app.composition.captions import CaptionPlan
from app.composition.policy import CaptionStyle, safe_zone_for
from app.composition.types import CaptionEvent, CaptionWordTiming
from app.core.settings import Settings, get_settings
from app.render.execution.compiler import compile_render
from app.render.execution.policy import (
    COMPILER_VERSION,
    CORE_SOURCE_VALIDATION,
    DEFAULT_DELIVERY_PROFILE_KEY,
    Stage52Config,
)
from app.render.execution.qc import check_render_artifact
from app.render.execution.runner import run_compiled_render
from app.render.execution.service import (
    build_render_spec,
    get_render_execution,
    resolve_runtime_identity,
    validate_candidate_for_render,
)
from app.render.execution.types import (
    AssAsset,
    AttemptContext,
    CaptionEventSpec,
    CompiledRender,
    CropKeyframeSpec,
    RenderSpec,
    RuntimeIdentity,
    SceneSpec,
    TimelineOccurrence,
)
from app.services.storage import StorageService

SYNTHETIC_REGRESSION = "SYNTHETIC_REGRESSION"
PRODUCTION_MANUAL_ACCEPTANCE = "PRODUCTION_MANUAL_ACCEPTANCE"

_PLACEHOLDER_IDENTIFIERS = frozenset(
    {
        "",
        "manual",
        "manual-acceptance",
        "candidate",
        "source",
        "contract",
        "plan",
        "render-execution",
        "demo",
        "synthetic",
        "test",
        "placeholder",
    }
)
_FINGERPRINT_PATTERN = re.compile(r"^[0-9a-f]{16,128}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_FINAL_REFINEMENT_STATUSES = frozenset({"CANDIDATE_REFINED", "FINAL_TRANSCRIPT_READY"})
_AUDIO_COMPONENT_HINTS = ("audio", "local", "asr", "whisper", "transcription")


class AcceptanceValidationError(ValueError):
    """A production acceptance request is not backed by genuine evidence."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


# ---------------------------------------------------------------------------
# Synthetic regression (unchanged behaviour, explicitly labelled)
# ---------------------------------------------------------------------------


def _quote_spec(
    *,
    source_path: Path,
    execution_id: str,
    occurrences: tuple[TimelineOccurrence, ...],
    caption_events: tuple[CaptionEventSpec, ...],
    source_duration: float,
    ass_bytes: bytes,
    ass_event_count: int,
    display_width: int,
    display_height: int,
    frame_rate: Fraction,
) -> RenderSpec:
    return RenderSpec(
        render_execution_id=execution_id,
        candidate_id="manual-acceptance",
        source_id="manual-acceptance",
        render_contract_id="manual-acceptance",
        visual_plan_id="manual-acceptance",
        artifact_purpose="CORE_SOURCE_VALIDATION",
        source_media_relative_path=str(source_path),
        source_content_hash="",
        source_size_bytes=source_path.stat().st_size,
        source_mtime_ns=source_path.stat().st_mtime_ns,
        source_duration=source_duration,
        source_frame_rate=frame_rate,
        display_width=display_width,
        display_height=display_height,
        encoded_width=display_width,
        encoded_height=display_height,
        rotation_degrees=0,
        pixel_aspect_ratio=1.0,
        output_width=1080,
        output_height=1920,
        output_frame_rate=frame_rate,
        output_profile={"profile_key": "SHORTS_1080X1920", "width": 1080, "height": 1920},
        delivery_profile_key="MP4_H264_AAC_1080X1920_V1",
        plan_input_fingerprint="manual",
        plan_output_fingerprint="manual",
        ass=AssAsset(
            relative_path="manual/captions.ass",
            sha256="",
            event_count=ass_event_count,
            line_count=ass_event_count,
            policy_version="stage5.1-ass-v5",
        ),
        caption_events=caption_events,
        occurrences=occurrences,
        audio_channels=2,
    )


def _scene(
    index: int,
    block: int,
    start: float,
    end: float,
    mode: str,
    keyframes: tuple[CropKeyframeSpec, ...] = (),
) -> SceneSpec:
    return SceneSpec(
        scene_index=index,
        block_index=block,
        source_start=start,
        source_end=end,
        framing_mode=mode,
        interpolation_policy="smoothstep-ease",
        crop_keyframes=keyframes,
    )


def _occurrence(
    block: int, start: float, end: float, output_start: float, scenes: tuple[SceneSpec, ...]
) -> TimelineOccurrence:
    return TimelineOccurrence(
        occurrence_id=f"block-{block}",
        block_index=block,
        source_start=start,
        source_end=end,
        output_start=output_start,
        output_end=output_start + (end - start),
        source_role="HERO" if block == 0 else "SUPPORT",
        is_hero=block == 0,
        scenes=scenes,
    )


def _mixed_caption_ass() -> tuple[bytes, CaptionEvent, int]:
    """Build the mixed Arabic/English caption via production ASS serialization."""

    text = "أنا كنت content creator لمدة سنتين"
    tokens = text.split(" ")
    start = 85.5
    word_timings = tuple(
        CaptionWordTiming(
            index=index, text=token, start=start + index * 0.42, end=start + index * 0.42 + 0.40
        )
        for index, token in enumerate(tokens)
    )
    event = CaptionEvent(
        event_id="manual-mixed",
        block_index=0,
        word_start_index=0,
        word_end_index=len(tokens) - 1,
        start=start,
        end=start + len(tokens) * 0.42,
        text=text,
        lines=(text,),
        placement_zone="LOWER",
        placement_reason="MANUAL_ACCEPTANCE",
        word_timings=word_timings,
    )
    plan = CaptionPlan(policy_version="stage5.1-caption-layout-v4", events=(event,))
    style = CaptionStyle()
    safe_zone = safe_zone_for("SHORTS_VERTICAL_SAFE_ZONE_V1")
    return serialize_ass(plan, style, safe_zone), event, len(tokens)


def _runtime(source: Path, attempt: Path) -> RuntimeIdentity:
    return RuntimeIdentity(
        ffmpeg_version="ffmpeg",
        ffprobe_version="ffprobe",
        libavformat_version="",
        libass_version="",
        font_family="Noto Sans Arabic",
        font_match="",
        compiler_version=COMPILER_VERSION,
        policy_version="stage5.2-v3",
        source_absolute_path=str(source),
        attempt_directory=str(attempt),
    )


def build_synthetic_regression(
    source: Path, output: Path, config: Stage52Config
) -> dict[str, object]:
    """Build the deterministic synthetic-regression demonstration package."""

    output.mkdir(parents=True, exist_ok=True)
    display_width, display_height, frame_rate, source_duration = _probe_media(source)
    cases: list[dict[str, object]] = []

    tracked = _scene(
        0,
        0,
        85.2,
        88.2,
        "TRACKED_CROP",
        (
            CropKeyframeSpec(85.2, 0.45, 0.48, 0.95),
            CropKeyframeSpec(86.2, 0.50, 0.46, 0.82),
            CropKeyframeSpec(87.2, 0.55, 0.44, 0.70),
            CropKeyframeSpec(88.2, 0.52, 0.45, 0.78),
        ),
    )
    cases.append(
        _render_case(
            name="01-tracked-speaker.mp4",
            output=output,
            source=source,
            config=config,
            spec=_quote_spec(
                source_path=source,
                execution_id="manual-01",
                occurrences=(_occurrence(0, 85.2, 88.2, 0.0, (tracked,)),),
                caption_events=(),
                source_duration=source_duration,
                ass_bytes=b"",
                ass_event_count=0,
                display_width=display_width,
                display_height=display_height,
                frame_rate=frame_rate,
            ),
            ass_bytes=_empty_ass(),
            label=(
                "SYNTHETIC: real source A/V; invented smoothstep tracked crop path "
                "(not a frozen Stage 5.1 plan)"
            ),
        )
    )

    fill = _scene(0, 0, 71.78, 74.28, "BACKGROUND_FILL")
    cases.append(
        _render_case(
            name="02-multi-person-background-fill.mp4",
            output=output,
            source=source,
            config=config,
            spec=_quote_spec(
                source_path=source,
                execution_id="manual-02",
                occurrences=(_occurrence(0, 71.78, 74.28, 0.0, (fill,)),),
                caption_events=(),
                source_duration=source_duration,
                ass_bytes=b"",
                ass_event_count=0,
                display_width=display_width,
                display_height=display_height,
                frame_rate=frame_rate,
            ),
            ass_bytes=_empty_ass(),
            label="SYNTHETIC: real source A/V; accepted BACKGROUND_FILL treatment",
        )
    )

    ass_bytes, event, token_count = _mixed_caption_ass()
    static = _scene(0, 0, 84.8, 88.4, "STATIC_CROP", (CropKeyframeSpec(84.8, 0.5, 0.5, 0.95),))
    cases.append(
        _render_case(
            name="03-mixed-bidi-highlight.mp4",
            output=output,
            source=source,
            config=config,
            spec=_quote_spec(
                source_path=source,
                execution_id="manual-03",
                occurrences=(_occurrence(0, 84.8, 88.4, 0.0, (static,)),),
                caption_events=(CaptionEventSpec("manual-mixed", 0, event.start, event.end),),
                source_duration=source_duration,
                ass_bytes=ass_bytes,
                ass_event_count=1,
                display_width=display_width,
                display_height=display_height,
                frame_rate=frame_rate,
            ),
            ass_bytes=ass_bytes,
            label=(
                "SYNTHETIC: real source A/V; INVENTED caption text/timings "
                "(not the speaker's words)"
            ),
        )
    )

    scene_a = _scene(0, 0, 30.0, 32.0, "SOURCE_AS_IS")
    scene_b = _scene(1, 1, 85.2, 88.2, "SOURCE_AS_IS")
    cases.append(
        _render_case(
            name="04-multi-span-av-sync.mp4",
            output=output,
            source=source,
            config=config,
            spec=_quote_spec(
                source_path=source,
                execution_id="manual-04",
                occurrences=(
                    _occurrence(0, 30.0, 32.0, 0.0, (scene_a,)),
                    _occurrence(1, 85.2, 88.2, 2.0, (scene_b,)),
                ),
                caption_events=(),
                source_duration=source_duration,
                ass_bytes=b"",
                ass_event_count=0,
                display_width=display_width,
                display_height=display_height,
                frame_rate=frame_rate,
            ),
            ass_bytes=_empty_ass(),
            label=(
                "SYNTHETIC: real source A/V across two spans; invented SOURCE_AS_IS "
                "framing (no genuine Stage 5.1 decisions)"
            ),
        )
    )

    manifest = {
        "stage": "5.2",
        "classification": SYNTHETIC_REGRESSION,
        "counts_toward_final_gate": False,
        "purpose": "DETERMINISTIC_RENDERING_DEMONSTRATION",
        "source": str(source),
        "cases": cases,
        "publication_ready": False,
        "stage6_implemented": False,
        "note": (
            "Hand-built specs with invented text/timings. Useful for regression "
            "and rendering demonstrations only; never final product acceptance."
        ),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "README.md").write_text(_readme(manifest), encoding="utf-8")
    return manifest


def _render_case(
    *,
    name: str,
    output: Path,
    source: Path,
    config: Stage52Config,
    spec: RenderSpec,
    ass_bytes: bytes,
    label: str,
) -> dict[str, object]:
    case_dir = output / name.replace(".mp4", "")
    case_dir.mkdir(parents=True, exist_ok=True)
    compiled = compile_render(spec, _runtime(source, case_dir))
    artifacts = run_compiled_render(
        compiled,
        AttemptContext(
            attempt_directory=case_dir,
            timeout_seconds=600,
            ass_bytes=ass_bytes if ass_bytes else _empty_ass(),
        ),
    )
    qc = check_render_artifact(artifacts, artifacts.manifest, config, source_path=source)
    (case_dir / "qc.json").write_text(json.dumps(qc.as_dict(), indent=2), encoding="utf-8")
    produced = case_dir / "output.mp4"
    target = output / name
    produced.replace(target)
    return {
        "name": name,
        "path": str(target),
        "classification": SYNTHETIC_REGRESSION,
        "evidence": label,
        "probe": artifacts.probe,
        "duration_seconds": artifacts.duration_seconds,
        "wc": qc.status,
        "qc_reason_codes": list(qc.reason_codes),
        "source_intervals": [
            {
                "source_start": occ.source_start,
                "source_end": occ.source_end,
                "output_start": occ.output_start,
                "output_end": occ.output_end,
            }
            for occ in spec.occurrences
        ],
    }


def _probe_media(source: Path) -> tuple[int, int, Fraction, float]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate:format=duration",
            "-of",
            "json",
            str(source),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    stream = payload["streams"][0]
    duration = float(payload.get("format", {}).get("duration") or 300.0)
    return (
        int(stream["width"]),
        int(stream["height"]),
        _fraction(stream.get("r_frame_rate")),
        duration,
    )


def _fraction(text: object) -> Fraction:
    if isinstance(text, str) and "/" in text:
        numerator, denominator = text.split("/", 1)
        try:
            return Fraction(int(numerator), int(denominator))
        except (ValueError, ZeroDivisionError):
            pass
    return Fraction(30, 1)


def _empty_ass() -> bytes:
    return (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "PlayResX: 1080\n"
        "PlayResY: 1920\n"
        "WrapStyle: 2\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: CaptionLower,Noto Sans Arabic,88,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,"
        "0,0,0,0,100,100,0,0,1,7,3,2,54,162,504,1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    ).encode("utf-8")


def _readme(manifest: dict[str, object]) -> str:
    classification = manifest.get("classification", SYNTHETIC_REGRESSION)
    lines = [
        f"# Stage 5.2 {classification}",
        "",
    ]
    if classification == SYNTHETIC_REGRESSION:
        lines += [
            "Deterministic synthetic-regression MP4s. Hand-built specs with",
            "invented text/timings. NOT final product acceptance.",
            "",
        ]
    for case in _case_list(manifest):
        lines.append(f"- `{case.get('name')}` — {case.get('evidence')} (QC {case.get('wc')})")
    lines += [
        "",
        "Automatic QC proves structural/timing correctness only; it does not prove",
        "speech transcription, caption-to-speech sync, lip sync, composition, or",
        "readability.",
        "",
    ]
    return "\n".join(lines)


def _case_list(manifest: dict[str, object]) -> list[dict[str, object]]:
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        return []
    return [case for case in cases if isinstance(case, dict)]


# ---------------------------------------------------------------------------
# Production manual acceptance
# ---------------------------------------------------------------------------


def _require_real_uuid(value: object, reason_code: str) -> str:
    text = str(value or "")
    if text.lower() in _PLACEHOLDER_IDENTIFIERS:
        raise AcceptanceValidationError(reason_code)
    try:
        uuid.UUID(text)
    except (ValueError, AttributeError, TypeError) as error:
        raise AcceptanceValidationError(reason_code) from error
    return text


def _require_fingerprint(value: object, reason_code: str) -> str:
    text = str(value or "")
    if not _FINGERPRINT_PATTERN.match(text):
        raise AcceptanceValidationError(reason_code)
    return text


def _require_sha256(value: object, reason_code: str) -> str:
    text = str(value or "")
    if not _SHA256_PATTERN.match(text):
        raise AcceptanceValidationError(reason_code)
    return text


def validate_production_evidence(
    *,
    spec: RenderSpec,
    contract_payload: Mapping[str, object],
    plan_payload: Mapping[str, object],
    refinement: object,
) -> None:
    """Fail closed unless the request is backed by genuine persisted evidence.

    Rejects placeholder contract/plan/candidate identities, empty plan
    fingerprints, a missing/mismatched canonical ASS, seeded or demonstration
    (non-audio-backed) ``FINAL_CLIP`` evidence, and occurrences without a real
    per-occurrence Stage 5.1 visual decision.
    """

    _require_real_uuid(spec.candidate_id, "CANDIDATE_IDENTITY_MISSING")
    _require_real_uuid(spec.source_id, "SOURCE_IDENTITY_MISSING")
    _require_real_uuid(spec.render_contract_id, "CONTRACT_IDENTITY_MISSING")
    _require_real_uuid(spec.visual_plan_id, "VISUAL_PLAN_IDENTITY_MISSING")
    _require_fingerprint(spec.plan_input_fingerprint, "PLAN_INPUT_FINGERPRINT_MISSING")
    _require_fingerprint(spec.plan_output_fingerprint, "PLAN_OUTPUT_FINGERPRINT_MISSING")
    _require_sha256(spec.ass.sha256, "ASS_HASH_MISSING")
    if not spec.caption_events and spec.ass.event_count:
        raise AcceptanceValidationError("ASS_EVENT_MISMATCH")

    _validate_final_refinement(refinement)
    _validate_occurrence_decisions(spec, contract_payload, plan_payload)


def _validate_final_refinement(refinement: object) -> None:
    if refinement is None:
        raise AcceptanceValidationError("FINAL_CLIP_REFINEMENT_MISSING")
    priority = getattr(getattr(refinement, "priority", None), "value", None)
    if priority != "FINAL_CLIP":
        raise AcceptanceValidationError("FINAL_CLIP_REFINEMENT_MISSING")
    status = getattr(getattr(refinement, "status", None), "value", None)
    if status not in _FINAL_REFINEMENT_STATUSES:
        raise AcceptanceValidationError("FINAL_CLIP_NOT_READY")
    audio_path = getattr(refinement, "audio_relative_path", None)
    audio_hash = getattr(refinement, "audio_content_hash", None)
    if not audio_path or not audio_hash:
        raise AcceptanceValidationError("FINAL_CLIP_NOT_AUDIO_BACKED")
    _require_sha256(audio_hash, "FINAL_CLIP_NOT_AUDIO_BACKED")
    if not list(getattr(refinement, "word_timestamps", None) or []):
        raise AcceptanceValidationError("FINAL_CLIP_WORD_TIMESTAMPS_MISSING")
    if not str(getattr(refinement, "final_transcript", "") or "").strip():
        raise AcceptanceValidationError("FINAL_CLIP_TRANSCRIPT_MISSING")
    components = getattr(refinement, "component_fingerprints", None) or {}
    keys = (
        " ".join(str(key).lower() for key in components) if isinstance(components, Mapping) else ""
    )
    if not any(hint in keys for hint in _AUDIO_COMPONENT_HINTS):
        raise AcceptanceValidationError("FINAL_CLIP_NOT_AUDIO_BACKED")


def _validate_occurrence_decisions(
    spec: RenderSpec,
    contract_payload: Mapping[str, object],
    plan_payload: Mapping[str, object],
) -> None:
    if not spec.occurrences:
        raise AcceptanceValidationError("NO_SOURCE_OCCURRENCES")
    scenes = plan_payload.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise AcceptanceValidationError("MULTI_SPAN_VISUAL_DECISIONS_MISSING")
    scene_blocks = {
        int(scene.get("block_index", -1))
        for scene in scenes
        if isinstance(scene, Mapping) and scene.get("framing_mode")
    }
    for occurrence in spec.occurrences:
        if not occurrence.scenes:
            raise AcceptanceValidationError("MULTI_SPAN_VISUAL_DECISIONS_MISSING")
        if occurrence.block_index not in scene_blocks:
            raise AcceptanceValidationError("MULTI_SPAN_VISUAL_DECISIONS_MISSING")
        for scene in occurrence.scenes:
            if not scene.framing_mode:
                raise AcceptanceValidationError("MULTI_SPAN_VISUAL_DECISIONS_MISSING")


@dataclass(frozen=True)
class _ProductionContext:
    candidate: object
    contract: object
    plan: object
    refinement: object
    spec: RenderSpec


def _load_production_context(
    session: object, candidate_id: str, settings: Settings
) -> _ProductionContext:
    from app.models import ClipCandidate

    candidate_uuid = uuid.UUID(candidate_id)
    candidate = session.get(ClipCandidate, candidate_uuid)  # type: ignore[attr-defined]
    if candidate is None:
        raise AcceptanceValidationError("CANDIDATE_NOT_FOUND")
    prerequisites = validate_candidate_for_render(session, candidate_uuid, settings=settings)
    contract = prerequisites.contract
    plan = prerequisites.plan
    contract_payload = dict(contract.contract_payload or {})
    plan_payload = dict(plan.plan_payload or {})
    storage = StorageService(settings.storage_root)
    spec = build_render_spec(
        session,
        candidate,
        artifact_purpose=CORE_SOURCE_VALIDATION,
        delivery_profile_key=DEFAULT_DELIVERY_PROFILE_KEY,
        settings=settings,
        storage=storage,
    )
    refinement = None
    refinement_id = getattr(plan, "final_refinement_id", None)
    if refinement_id is not None:
        from app.models import CandidateRefinement

        refinement = session.get(CandidateRefinement, refinement_id)  # type: ignore[attr-defined]
    validate_production_evidence(
        spec=spec,
        contract_payload=contract_payload,
        plan_payload=plan_payload,
        refinement=refinement,
    )
    return _ProductionContext(
        candidate=candidate, contract=contract, plan=plan, refinement=refinement, spec=spec
    )


def build_production_acceptance(
    candidate_id: str,
    output: Path,
    *,
    settings: Settings | None = None,
    session_factory: Callable[[], Any] | None = None,
    executor: Callable[[str, str | None], Mapping[str, object]] | None = None,
    queue: bool = True,
) -> dict[str, object]:
    """Render one genuine candidate and export its full lineage evidence.

    Uses the production adapter to build the request (which revalidates
    currentness/ASS/source identity), the production queue, and the production
    render task. Raises :class:`AcceptanceValidationError` when genuine evidence
    is absent; it never substitutes synthetic inputs.
    """

    resolved = settings or get_settings()
    if session_factory is None:
        from app.db.session import create_session_factory

        session_factory = create_session_factory()
    session = session_factory()
    try:
        context = _load_production_context(session, candidate_id, resolved)
        spec = context.spec
        storage = StorageService(resolved.storage_root)

        execution_id, job_id = _queue_and_execute(
            session, candidate_id, resolved, queue=queue, executor=executor
        )
        session.expire_all()
        row = get_render_execution(session, execution_id)
        if row is None:
            raise AcceptanceValidationError("RENDER_EXECUTION_MISSING")
        if row.lifecycle.value != "COMPLETE":
            raise AcceptanceValidationError("RENDER_EXECUTION_NOT_COMPLETE")
        if row.qc_status is None or row.qc_status.value not in {"PASS", "WARN"}:
            raise AcceptanceValidationError("RENDER_QC_NOT_ACCEPTABLE")

        case = _export_production_case(
            session=session,
            output=output,
            candidate_id=candidate_id,
            spec=spec,
            context=context,
            row=row,
            storage=storage,
            settings=resolved,
        )
    finally:
        session.close()

    manifest = {
        "stage": "5.2",
        "classification": PRODUCTION_MANUAL_ACCEPTANCE,
        "counts_toward_final_gate": True,
        "purpose": "CORE_SOURCE_VALIDATION",
        "candidate_id": candidate_id,
        "cases": [case],
        "publication_ready": False,
        "stage6_implemented": False,
        "fingerprint_limitation": (
            "Fingerprint/hash checks prove identity and lineage only, not speech "
            "correctness, caption sync, lip sync, composition, or readability."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "README.md").write_text(_readme(manifest), encoding="utf-8")
    return manifest


def _queue_and_execute(
    session: object,
    candidate_id: str,
    settings: Settings,
    *,
    queue: bool,
    executor: Callable[[str, str | None], Mapping[str, object]] | None,
) -> tuple[str, str | None]:
    from app.render.execution.queue import queue_render_execution

    outcome = queue_render_execution(
        session,
        candidate_id,
        delivery_profile_key=DEFAULT_DELIVERY_PROFILE_KEY,
        artifact_purpose=CORE_SOURCE_VALIDATION,
        settings=settings,
        dispatcher=lambda *args: None,
    )
    execution_id = str(outcome.render_execution_id)
    job_id = str(outcome.job_id) if outcome.job_id else None
    if queue and outcome.queued:
        _run_production_execution(execution_id, job_id, executor)
    return execution_id, job_id


def _run_production_execution(
    execution_id: str,
    job_id: str | None,
    executor: Callable[[str, str | None], Mapping[str, object]] | None,
) -> None:
    if executor is not None:
        executor(execution_id, job_id)
        return
    from app.workers.tasks import run_render_execution

    run_render_execution.apply(args=[execution_id, job_id, False]).get()


def _export_production_case(
    *,
    session: object,
    output: Path,
    candidate_id: str,
    spec: RenderSpec,
    context: _ProductionContext,
    row: object,
    storage: StorageService,
    settings: Settings,
) -> dict[str, object]:
    case_dir = output / candidate_id
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    artifact = dict(getattr(row, "artifact_reference", {}) or {})
    relative = str(artifact.get("relative_path") or "")
    if not relative:
        raise AcceptanceValidationError("ARTIFACT_REFERENCE_MISSING")
    managed_path = (storage.storage_root / relative).resolve()
    if not managed_path.is_file():
        raise AcceptanceValidationError("ARTIFACT_MISSING")
    output_sha = _sha256_file(managed_path)
    if artifact.get("sha256") and artifact["sha256"] != output_sha:
        raise AcceptanceValidationError("ARTIFACT_HASH_MISMATCH")

    ass_path = (storage.storage_root / spec.ass.relative_path).resolve()
    ass_bytes = ass_path.read_bytes()
    ass_sha = hashlib.sha256(ass_bytes).hexdigest()
    if ass_sha != spec.ass.sha256:
        raise AcceptanceValidationError("ASS_HASH_MISMATCH")

    source_path = (storage.storage_root / spec.source_media_relative_path).resolve()
    source_sha = _sha256_file(source_path)

    runtime = resolve_runtime_identity(
        settings,
        source_absolute_path=str(source_path),
        attempt_directory=str(case_dir),
    )
    compiled = compile_render(spec, runtime)
    filtergraph = compiled.filtergraph
    if filtergraph != _read_attempt_filtergraph(managed_path.parent):
        raise AcceptanceValidationError("FILTERGRAPH_MISMATCH")

    review_copy = case_dir / "output.mp4"
    shutil.copyfile(managed_path, review_copy)

    _write_json(case_dir / "render_spec.json", spec.as_dict())
    _write_json(
        case_dir / "contract_snapshot.json", getattr(context.contract, "contract_payload", {})
    )
    _write_json(case_dir / "plan_snapshot.json", getattr(context.plan, "plan_payload", {}))
    _write_json(case_dir / "final_refinement.json", _refinement_snapshot(context.refinement))
    _write_json(
        case_dir / "render_execution.json",
        {
            "render_execution_id": str(getattr(row, "id", "")),
            "lifecycle": getattr(getattr(row, "lifecycle", None), "value", None),
            "qc_status": getattr(getattr(row, "qc_status", None), "value", None),
            "input_fingerprint": getattr(row, "input_fingerprint", ""),
            "output_fingerprint": getattr(row, "output_fingerprint", ""),
            "compiler_fingerprint": getattr(row, "compiler_fingerprint", ""),
            "runtime_fingerprint": getattr(row, "runtime_fingerprint", ""),
            "qc_fingerprint": getattr(row, "qc_fingerprint", ""),
            "artifact_reference": artifact,
            "execution_manifest": getattr(row, "execution_manifest", {}),
            "qc_result": getattr(row, "qc_result", {}),
            "reason_codes": list(getattr(row, "reason_codes", []) or []),
            "managed_artifact_relative_path": relative,
        },
    )
    _write_json(case_dir / "qc.json", getattr(row, "qc_result", {}))
    (case_dir / "filtergraph.txt").write_text(filtergraph, encoding="utf-8")
    (case_dir / "captions.ass").write_bytes(ass_bytes)
    _write_json(case_dir / "geometry.json", _geometry_evidence(compiled))
    _write_json(
        case_dir / "identity.json",
        {
            "candidate_id": candidate_id,
            "source_id": spec.source_id,
            "render_contract_id": spec.render_contract_id,
            "visual_plan_id": spec.visual_plan_id,
            "final_refinement_id": str(getattr(context.refinement, "id", "")),
            "source_media_relative_path": spec.source_media_relative_path,
            "source_content_hash": spec.source_content_hash,
            "source_sha256_observed": source_sha,
            "ass_sha256": ass_sha,
            "output_sha256": output_sha,
            "plan_input_fingerprint": spec.plan_input_fingerprint,
            "plan_output_fingerprint": spec.plan_output_fingerprint,
        },
    )

    return {
        "name": "output.mp4",
        "path": str(review_copy),
        "classification": PRODUCTION_MANUAL_ACCEPTANCE,
        "managed_artifact_relative_path": relative,
        "evidence_dir": str(case_dir),
        "candidate_id": candidate_id,
        "source_id": spec.source_id,
        "render_contract_id": spec.render_contract_id,
        "visual_plan_id": spec.visual_plan_id,
        "final_refinement_id": str(getattr(context.refinement, "id", "")),
        "render_execution_id": str(getattr(row, "id", "")),
        "output_sha256": output_sha,
        "ass_sha256": ass_sha,
        "qc_status": getattr(getattr(row, "qc_status", None), "value", None),
        "qc_reason_codes": list(getattr(row, "reason_codes", []) or []),
        "framing_modes": sorted(
            {scene.framing_mode for occurrence in spec.occurrences for scene in occurrence.scenes}
        ),
        "occurrences": [
            {
                "block_index": occurrence.block_index,
                "source_start": occurrence.source_start,
                "source_end": occurrence.source_end,
                "output_start": occurrence.output_start,
                "output_end": occurrence.output_end,
                "framing_modes": sorted({scene.framing_mode for scene in occurrence.scenes}),
            }
            for occurrence in spec.occurrences
        ],
    }


def _read_attempt_filtergraph(attempt_directory: Path) -> str:
    path = attempt_directory / "filtergraph.txt"
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def _geometry_evidence(compiled: CompiledRender) -> list[dict[str, object]]:
    scenes = compiled.diagnostics.get("scenes")
    if not isinstance(scenes, list):
        return []
    evidence: list[dict[str, object]] = []
    for scene in scenes:
        if isinstance(scene, Mapping) and isinstance(scene.get("geometry"), Mapping):
            evidence.append(dict(scene["geometry"]))
    return evidence


def _refinement_snapshot(refinement: object) -> dict[str, object]:
    return {
        "id": str(getattr(refinement, "id", "")),
        "priority": getattr(getattr(refinement, "priority", None), "value", None),
        "status": getattr(getattr(refinement, "status", None), "value", None),
        "audio_relative_path": getattr(refinement, "audio_relative_path", None),
        "audio_content_hash": getattr(refinement, "audio_content_hash", None),
        "component_fingerprints": getattr(refinement, "component_fingerprints", {}),
        "final_transcript": getattr(refinement, "final_transcript", ""),
        "word_timestamps": list(getattr(refinement, "word_timestamps", []) or []),
        "confidence": getattr(refinement, "confidence", None),
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 5.2 acceptance package builders.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    synthetic = subparsers.add_parser(
        "synthetic-regression", help="Deterministic synthetic rendering demonstration."
    )
    synthetic.add_argument("--source", required=True, type=Path)
    synthetic.add_argument("--output", required=True, type=Path)

    production = subparsers.add_parser(
        "production-acceptance",
        help="Genuine persisted-evidence production acceptance (requires a database).",
    )
    production.add_argument("--candidate", required=True)
    production.add_argument("--output", required=True, type=Path)
    production.add_argument(
        "--no-queue",
        action="store_true",
        help="Reuse an existing cache-eligible execution instead of executing.",
    )

    args = parser.parse_args()
    if args.mode == "synthetic-regression":
        manifest = build_synthetic_regression(args.source, args.output, Stage52Config())
    else:
        manifest = build_production_acceptance(args.candidate, args.output, queue=not args.no_queue)
    print(
        json.dumps(
            {
                "classification": manifest.get("classification"),
                "output": str(args.output),
                "cases": [case.get("name") for case in _case_list(manifest)],
            }
        )
    )


if __name__ == "__main__":
    main()

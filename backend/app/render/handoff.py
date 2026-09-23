"""Typed read-only Stage 5.0 -> Stage 5.1 handoff builder.

Stage 5.1 must never rediscover or recompute anything: this handoff exposes the
exact persisted contract, ordered blocks, bindings, slots, caption source, and
readiness. It performs no probing, provider call, rendering, captioning, or TTS,
and it never claims platform safety, monetization, or final-render readiness.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy.orm import Session

from app.models import ClipCandidate
from app.render.service import get_current_render_contract, read_render_contract

_STAGE_FLAGS = {
    "stage5_1_implemented": True,
    "stage5_2_implemented": False,
    "stage6_implemented": False,
}


def _as_uuid(value: uuid.UUID | str) -> uuid.UUID:
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


def build_stage5_1_handoff(
    session: Session, candidate_id: uuid.UUID | str
) -> dict[str, Any] | None:
    candidate_uuid = _as_uuid(candidate_id)
    candidate = session.get(ClipCandidate, candidate_uuid)
    if candidate is None:
        return None

    row = get_current_render_contract(session, candidate.id)
    view = read_render_contract(session, candidate.id)
    live_freshness = view.live_freshness if view is not None else "NOT_CURRENT"
    effective = bool(view.effective) if view is not None else False
    base: dict[str, Any] = {
        "candidate": {
            "id": str(candidate.id),
            "candidate_key": candidate.candidate_key,
            "source_id": str(candidate.source_video_id),
            "disposition": candidate.disposition.value,
        },
        "contract": None,
        "selection": None,
        "selected_plan": None,
        "source_media": None,
        "output_profile": None,
        "compatibility": None,
        "blocks": [],
        "hero": None,
        "preservation_constraints": [],
        "retention": None,
        "narration": None,
        "materialization": {"required": False, "slots": []},
        "caption_input": None,
        "framing_input": None,
        "language_and_dialect": None,
        "verification": None,
        "governance": None,
        "readiness": None,
        "reason_codes": [],
        **_STAGE_FLAGS,
    }
    if row is None:
        base["reason"] = "NO_RENDER_CONTRACT"
        return base

    payload = dict(row.contract_payload or {})
    identity = dict(payload.get("identity") or {})
    selection_section = dict(identity.get("selection") or {}) if isinstance(identity, dict) else {}
    selected_plan_section = (
        dict(identity.get("selected_plan") or {}) if isinstance(identity, dict) else {}
    )
    source_media = dict(payload.get("source_media") or {})
    probe = dict(source_media.get("probe") or {})
    compatibility_section = dict(payload.get("compatibility") or {})
    governance = dict(payload.get("governance") or {})

    base["contract"] = {
        "id": str(row.id),
        "input_fingerprint": row.input_fingerprint,
        "output_fingerprint": row.output_fingerprint,
        "status": row.status.value,
        "compatibility_outcome": (
            row.compatibility_outcome.value if row.compatibility_outcome else None
        ),
        "contract_ready": bool(row.contract_ready),
        "is_current": bool(row.is_current),
        "live_freshness": live_freshness,
        "effective": effective,
        "policy_version": row.policy_version,
        "schema_version": row.schema_version,
        "fingerprint_version": row.fingerprint_version,
        "profile_key": row.profile_key,
        "profile_version": row.profile_version,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }
    base["selection"] = {
        "id": selection_section.get("id"),
        "status": selection_section.get("status"),
        "selected_with_caution": selection_section.get("selected_with_caution"),
    }
    base["selected_plan"] = {
        "id": selected_plan_section.get("id"),
        "plan_key": selected_plan_section.get("plan_key"),
        "plan_output_fingerprint": selected_plan_section.get("plan_output_fingerprint"),
        "strategy": dict(identity.get("strategy") or {}),
        "governance": dict(identity.get("governance") or {}),
    }
    base["source_media"] = {
        "identity": dict(source_media.get("identity") or row.source_media_identity or {}),
        "managed_relative_path": source_media.get("managed_relative_path"),
        "content_hash": source_media.get("content_hash"),
        "duration": probe.get("duration_seconds"),
        "video_stream": {
            "codec": probe.get("video_codec"),
            "width": probe.get("width"),
            "height": probe.get("height"),
            "frame_rate": probe.get("frames_per_second"),
        },
        "audio_stream": {
            "codec": probe.get("audio_codec"),
            "sample_rate": probe.get("audio_sample_rate"),
        },
        "streams": {
            "video_streams": probe.get("video_streams", 0),
            "audio_streams": probe.get("audio_streams", 0),
        },
        "probe_fingerprint": row.probe_fingerprint,
        "source_media_fingerprint": row.source_media_fingerprint,
    }
    base["output_profile"] = dict(payload.get("output_profile") or {})
    base["compatibility"] = {
        "outcome": compatibility_section.get("outcome"),
        "exact_match": compatibility_section.get("exact_match"),
        "per_block_verdicts": list(compatibility_section.get("per_block_verdicts") or []),
        "unresolved_spans": list(compatibility_section.get("unresolved_spans") or []),
        "recovered_code_switch_tokens": list(
            compatibility_section.get("recovered_code_switch_tokens") or []
        ),
        "compatibility_policy_version": compatibility_section.get("compatibility_policy_version"),
        "evidence": dict(row.compatibility_evidence or {}),
    }
    base["blocks"] = list(payload.get("blocks") or [])
    base["hero"] = payload.get("hero")
    base["preservation_constraints"] = list(payload.get("preservation_constraints") or [])
    base["retention"] = payload.get("retention")
    base["narration"] = payload.get("narration")
    base["materialization"] = dict(
        payload.get("materialization") or {"required": False, "slots": []}
    )
    base["caption_input"] = payload.get("caption_input")
    base["framing_input"] = payload.get("framing_input")
    base["language_and_dialect"] = payload.get("language_and_dialect")
    base["verification"] = payload.get("verification")
    base["governance"] = {
        "dimensions": dict(governance.get("dimensions") or {}),
        "warnings": list(governance.get("warnings") or []),
        "reason_codes": list(governance.get("reason_codes") or []),
        "provider_evidence": dict(governance.get("provider_evidence") or {}),
        "platform_risk": dict(governance.get("platform_risk") or {}),
    }
    base["readiness"] = dict(row.readiness or {})
    base["reason_codes"] = list(row.reason_codes or [])
    return base


__all__ = ["build_stage5_1_handoff"]

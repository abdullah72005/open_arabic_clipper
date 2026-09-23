"""Typed read-only Stage 5.1 -> Stage 5.2 handoff builder.

Exposes the exact persisted visual-composition plan, its currentness/effectiveness,
ordered blocks, bound source spans, scenes, captions, ASS asset, overlays, and
materialization slots. It performs no probing, rendering, captioning, TTS, or
provider call, and it never claims publication or final-render readiness.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from sqlalchemy.orm import Session

from app.composition.geometry import DisplayProbe
from app.composition.policy import Stage51Config
from app.composition.service import read_visual_composition
from app.models import ClipCandidate
from app.render.handoff import build_stage5_1_handoff

_NOT_READY_FLAGS = {
    "source_framing_ready": False,
    "source_captions_ready": False,
    "authored_assets_pending": True,
    "stage5_2_handoff_eligible": False,
}


def _as_uuid(value: uuid.UUID | str) -> uuid.UUID:
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _sequence(value: object) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _bound_spans(blocks: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    spans: list[dict[str, object]] = []
    for block in blocks:
        if str(block.get("block_type") or "") != "SOURCE_EXCERPT":
            continue
        binding = block.get("source_binding")
        if not isinstance(binding, Mapping):
            continue
        spans.append(
            {
                "block_index": block.get("block_index"),
                "slot_kind": block.get("slot_kind"),
                "source_binding_verbatim": dict(binding),
            }
        )
    return spans


def build_stage5_2_handoff(
    session: Session,
    candidate_id: uuid.UUID | str,
    *,
    display_probe: DisplayProbe,
    config: Stage51Config,
) -> dict[str, Any] | None:
    candidate = session.get(ClipCandidate, _as_uuid(candidate_id))
    if candidate is None:
        return None
    stage51 = build_stage5_1_handoff(session, candidate.id)
    view = read_visual_composition(
        session, candidate.id, display_probe=display_probe, config=config
    )
    base: dict[str, Any] = {
        "candidate": {
            "id": str(candidate.id),
            "candidate_key": candidate.candidate_key,
            "source_id": str(candidate.source_video_id),
            "disposition": candidate.disposition.value,
        },
        "contract": (stage51 or {}).get("contract"),
        "plan": None,
        "source_media": None,
        "display_geometry": {},
        "frames_per_second": None,
        "output_profile": {},
        "blocks": [],
        "bound_source_spans": [],
        "scenes": [],
        "captions": {},
        "ass": {},
        "overlays": [],
        "materialization": {"required": False, "slots": []},
        "safe_zone": {},
        "protection_markers": [],
        "readiness": dict(_NOT_READY_FLAGS),
        "reason_codes": [],
        "warnings": [],
        "metrics": {},
        "final_timeline_frozen": False,
        "publication_ready": False,
        "render_ready": False,
        "stage5_2_implemented": False,
        "stage6_implemented": False,
    }
    if view is None:
        base["reason"] = "NO_VISUAL_COMPOSITION_PLAN"
        return base

    row = view.row
    payload = dict(row.plan_payload or {})
    geometry = _mapping(payload.get("geometry"))
    # Ordered blocks and bound source spans come from the executable Stage 5.0
    # contract; the Stage 5.1 plan carries scenes/captions/ASS/overlays.
    contract_blocks = _sequence((stage51 or {}).get("blocks"))
    readiness = _mapping(payload.get("readiness"))
    if not view.effective:
        readiness = {**readiness, **_NOT_READY_FLAGS}
    source_media = _mapping((stage51 or {}).get("source_media"))
    display_geometry = {
        key: value
        for key, value in geometry.items()
        if key not in {"output", "safe_zone", "frames_per_second"}
    }
    base["plan"] = {
        "id": str(row.id),
        "status": row.status.value,
        "execution_status": row.execution_status.value,
        "plan_ready": bool(row.plan_ready),
        "is_current": bool(row.is_current),
        "live_freshness": view.live_freshness,
        "effective": bool(view.effective),
        "input_fingerprint": row.input_fingerprint,
        "output_fingerprint": row.output_fingerprint,
        "contract_input_fingerprint": row.contract_input_fingerprint,
        "contract_output_fingerprint": row.contract_output_fingerprint,
        "caption_source_fingerprint": row.caption_source_fingerprint,
        "source_media_fingerprint": row.source_media_fingerprint,
        "analysis_fingerprint": row.analysis_fingerprint,
        "framing_fingerprint": row.framing_fingerprint,
        "ass_fingerprint": row.ass_fingerprint,
        "policy_version": row.policy_version,
        "schema_version": row.schema_version,
        "fingerprint_version": row.fingerprint_version,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }
    base["source_media"] = {
        "identity": _mapping(source_media.get("identity")) or dict(row.source_media_identity),
        "managed_relative_path": source_media.get("managed_relative_path"),
        "content_hash": source_media.get("content_hash"),
        "source_media_fingerprint": row.source_media_fingerprint,
    }
    base["display_geometry"] = display_geometry
    base["frames_per_second"] = geometry.get("frames_per_second")
    base["output_profile"] = _mapping(payload.get("output_profile"))
    base["blocks"] = contract_blocks
    base["bound_source_spans"] = _bound_spans(contract_blocks)
    base["scenes"] = _sequence(payload.get("scenes"))
    base["captions"] = _mapping(payload.get("captions"))
    base["ass"] = _mapping(payload.get("ass"))
    base["overlays"] = _sequence(payload.get("overlays"))
    base["materialization"] = _mapping(payload.get("materialization")) or {
        "required": False,
        "slots": [],
    }
    base["safe_zone"] = _mapping(geometry.get("safe_zone"))
    base["protection_markers"] = _sequence(
        _mapping(payload.get("protection")).get("protection_markers")
    )
    base["readiness"] = readiness
    base["reason_codes"] = list(row.reason_codes or [])
    base["warnings"] = _sequence(payload.get("warnings"))
    base["metrics"] = dict(row.metrics or {})
    return base


__all__ = ["build_stage5_2_handoff"]

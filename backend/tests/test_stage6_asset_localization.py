"""Stage 6 authored-asset localization through StorageService (SOL defect C).

The production executor must resolve every referenced authored asset through
StorageService, verify its immutable hash/size, and hand the compiler a
controlled attempt-local copy -- never a provider path, URL, or traversal.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from app.render.execution.authored import (
    ASSET_ROLE_NARRATION,
    ASSET_ROLE_SOURCE_VIDEO,
    MediaAssetRef,
)
from app.render.execution.executor import localize_authored_assets
from app.render.execution.service import RenderExecutionError
from app.services.storage import StorageService


class _Payload:
    def __init__(self, assets: tuple[MediaAssetRef, ...]) -> None:
        self.assets = assets


def _asset(relative: str, data: bytes, *, role: str = ASSET_ROLE_NARRATION) -> MediaAssetRef:
    return MediaAssetRef(
        asset_id="narr-1",
        role=role,
        storage_relative_path=relative,
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        duration_seconds=1.0,
        sample_rate=24000,
        channels=1,
    )


def test_localizes_and_verifies_narration_asset(tmp_path: Path) -> None:
    storage_root = tmp_path / "managed"
    storage_root.mkdir()
    managed = storage_root / "sources" / "s1" / "narration.wav"
    managed.parent.mkdir(parents=True, exist_ok=True)
    payload_bytes = b"RIFF....WAVEfmt " + b"\x00" * 16
    managed.write_bytes(payload_bytes)
    storage = StorageService(storage_root)
    attempt = tmp_path / "attempt"

    mapping = localize_authored_assets(
        storage,
        _Payload((_asset("sources/s1/narration.wav", payload_bytes),)),
        attempt,
    )
    assert mapping == {"narr-1": "asset-narr-1.wav"}
    assert (attempt / "asset-narr-1.wav").read_bytes() == payload_bytes


def test_source_video_asset_is_not_localized(tmp_path: Path) -> None:
    storage_root = tmp_path / "managed"
    storage_root.mkdir()
    storage = StorageService(storage_root)
    mapping = localize_authored_assets(
        storage,
        _Payload(
            (
                _asset(
                    "sources/s1/source.mp4",
                    b"data",
                    role=ASSET_ROLE_SOURCE_VIDEO,
                ),
            )
        ),
        tmp_path / "attempt",
    )
    assert mapping == {}


def test_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    storage_root = tmp_path / "managed"
    storage_root.mkdir()
    managed = storage_root / "sources" / "s1" / "narration.wav"
    managed.parent.mkdir(parents=True, exist_ok=True)
    managed.write_bytes(b"actual-bytes")
    storage = StorageService(storage_root)
    asset = _asset("sources/s1/narration.wav", b"different-bytes")
    with pytest.raises(RenderExecutionError) as error:
        localize_authored_assets(storage, _Payload((asset,)), tmp_path / "attempt")
    assert error.value.reason_code == "ASSET_HASH_MISMATCH"


def test_traversal_path_is_rejected(tmp_path: Path) -> None:
    storage_root = tmp_path / "managed"
    storage_root.mkdir()
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"secret")
    storage = StorageService(storage_root)
    asset = _asset("../outside.wav", b"secret")
    with pytest.raises(RenderExecutionError) as error:
        localize_authored_assets(storage, _Payload((asset,)), tmp_path / "attempt")
    assert error.value.reason_code in {"ASSET_PATH_INVALID", "ASSET_MISSING"}

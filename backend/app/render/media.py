"""Bounded read-only source-media identity and ffprobe preflight.

The only external process Stage 5.0 may invoke is a single bounded read-only
ffprobe metadata probe, and only through the injectable ``prober`` seam so tests
never require ffprobe. No full-file hashing, no decoding, no frame extraction,
no rendering.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, cast

from app.media.ffprobe import (
    MediaMetadata,
    ProbeExecutionError,
    ProbeParseError,
)
from app.render.policy import (
    SOURCE_AUDIO_STREAM_MISSING,
    SOURCE_MEDIA_CHANGED_DURING_PREFLIGHT,
    SOURCE_MEDIA_CORRUPT,
    SOURCE_MEDIA_INVALID_DURATION,
    SOURCE_MEDIA_MISSING,
    SOURCE_MEDIA_NO_VIDEO_STREAM,
    SOURCE_MEDIA_NOT_INGESTED,
    SOURCE_MEDIA_UNMANAGED_PATH,
    SOURCE_MEDIA_ZERO_BYTES,
)
from app.render.types import SourceMediaFacts, SourceMediaIdentity
from app.services.storage import StorageService, StorageValidationError


class Prober(Protocol):
    """Injectable ffprobe seam."""

    def probe(self, path: Path) -> MediaMetadata: ...


class SourceMediaFailure(Exception):
    """A bounded, reason-coded source-media preflight failure."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


@dataclass(frozen=True)
class SourceMediaPreflight:
    """Result of source-media identity + probe preflight."""

    identity: SourceMediaIdentity | None
    facts: SourceMediaFacts | None
    reason_code: str | None = None
    probe_reused: bool = False
    managed_relative_path: str = ""
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return self.facts is not None and self.reason_code is None


def resolve_managed_source(
    source_id: object, source_uri: str | None, *, storage: StorageService
) -> tuple[Path, str]:
    """Resolve a managed source path or raise ``SourceMediaFailure``.

    Rejects empty/remote URIs and any path outside the managed source directory.
    """

    if not source_uri or source_uri.startswith(("http://", "https://")):
        raise SourceMediaFailure(SOURCE_MEDIA_NOT_INGESTED)
    candidate = Path(source_uri).expanduser()
    source_directory = storage.source_directory(str(source_id))
    try:
        resolved = candidate.resolve()
    except OSError as error:  # pragma: no cover - defensive
        raise SourceMediaFailure(SOURCE_MEDIA_UNMANAGED_PATH) from error
    try:
        resolved.relative_to(source_directory)
    except ValueError as error:
        raise SourceMediaFailure(SOURCE_MEDIA_UNMANAGED_PATH) from error
    try:
        relative = str(resolved.relative_to(storage.storage_root))
    except ValueError:
        relative = str(resolved)
    return resolved, relative


def source_media_identity(
    source_id: object, source_uri: str | None, *, storage: StorageService
) -> SourceMediaIdentity:
    """Cheap managed-file identity (stat only, never a full-file hash)."""

    path, relative = resolve_managed_source(source_id, source_uri, storage=storage)
    try:
        stat = path.stat()
    except FileNotFoundError as error:
        raise SourceMediaFailure(SOURCE_MEDIA_MISSING) from error
    except OSError as error:
        raise SourceMediaFailure(SOURCE_MEDIA_CORRUPT) from error
    if not path.is_file():
        raise SourceMediaFailure(SOURCE_MEDIA_MISSING)
    if stat.st_size <= 0:
        raise SourceMediaFailure(SOURCE_MEDIA_ZERO_BYTES)
    return SourceMediaIdentity(
        source_id=str(source_id),
        content_hash="",
        relative_path=relative,
        size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
    )


def _same_media_identity(cached: Mapping[str, object], identity: SourceMediaIdentity) -> bool:
    """Identity equality excluding the stored content hash (never recomputed)."""

    cached_body = {key: value for key, value in cached.items() if key != "content_hash"}
    return cached_body == {
        key: value for key, value in identity.as_dict().items() if key != "content_hash"
    }


def facts_from_metadata(metadata: MediaMetadata) -> SourceMediaFacts:
    return SourceMediaFacts(
        duration_seconds=metadata.duration_seconds,
        video_codec=metadata.video_codec,
        width=metadata.width,
        height=metadata.height,
        frames_per_second=metadata.frames_per_second,
        audio_codec=metadata.audio_codec,
        audio_sample_rate=metadata.audio_sample_rate,
    )


def facts_from_mapping(payload: Mapping[str, object]) -> SourceMediaFacts | None:
    """Rehydrate cached probe facts; never trusts malformed stored JSON."""

    try:
        duration = payload["duration_seconds"]
        codec = payload["video_codec"]
        width = payload["width"]
        height = payload["height"]
        fps = payload["frames_per_second"]
        audio_codec = payload.get("audio_codec")
        sample_rate = payload.get("audio_sample_rate")
        return SourceMediaFacts(
            duration_seconds=float(cast(float, duration)),
            video_codec=str(codec),
            width=int(cast(int, width)),
            height=int(cast(int, height)),
            frames_per_second=float(cast(float, fps)),
            audio_codec=str(audio_codec) if audio_codec is not None else None,
            audio_sample_rate=int(cast(int, sample_rate)) if sample_rate is not None else None,
        )
    except (KeyError, TypeError, ValueError):
        return None


def run_source_media_preflight(
    source_id: object,
    source_uri: str | None,
    *,
    storage: StorageService,
    prober: Prober,
    probe_reuse_enabled: bool = True,
    cached_identity: Mapping[str, object] | None = None,
    cached_facts: Mapping[str, object] | None = None,
) -> SourceMediaPreflight:
    """Identity + bounded probe preflight with cached reuse and re-stat."""

    try:
        identity = source_media_identity(source_id, source_uri, storage=storage)
    except SourceMediaFailure as failure:
        return SourceMediaPreflight(identity=None, facts=None, reason_code=failure.reason_code)
    except StorageValidationError:
        return SourceMediaPreflight(
            identity=None, facts=None, reason_code=SOURCE_MEDIA_UNMANAGED_PATH
        )

    if (
        probe_reuse_enabled
        and cached_facts
        and cached_identity is not None
        and _same_media_identity(cached_identity, identity)
    ):
        facts = facts_from_mapping(cached_facts)
        if facts is not None and facts.audio_codec is not None and facts.duration_seconds > 0:
            return SourceMediaPreflight(
                identity=identity,
                facts=facts,
                probe_reused=True,
                managed_relative_path=identity.relative_path,
            )

    path, relative = resolve_managed_source(source_id, source_uri, storage=storage)
    facts, reason = _probe_once(path, prober)
    if reason is not None:
        return SourceMediaPreflight(identity=identity, facts=None, reason_code=reason)

    try:
        restat = source_media_identity(source_id, source_uri, storage=storage)
    except SourceMediaFailure as failure:
        return SourceMediaPreflight(identity=identity, facts=None, reason_code=failure.reason_code)
    if restat.as_dict() != identity.as_dict():
        facts, reason = _probe_once(path, prober)
        if reason is not None:
            return SourceMediaPreflight(identity=identity, facts=None, reason_code=reason)
        try:
            final = source_media_identity(source_id, source_uri, storage=storage)
        except SourceMediaFailure as failure:
            return SourceMediaPreflight(
                identity=identity, facts=None, reason_code=failure.reason_code
            )
        if final.as_dict() != restat.as_dict():
            return SourceMediaPreflight(
                identity=final,
                facts=None,
                reason_code=SOURCE_MEDIA_CHANGED_DURING_PREFLIGHT,
            )
        identity = final

    return SourceMediaPreflight(
        identity=identity,
        facts=facts,
        managed_relative_path=relative,
    )


def _probe_once(path: Path, prober: Prober) -> tuple[SourceMediaFacts | None, str | None]:
    try:
        metadata = prober.probe(path)
    except ProbeParseError:
        return None, SOURCE_MEDIA_NO_VIDEO_STREAM
    except (ProbeExecutionError, OSError, ValueError):
        return None, SOURCE_MEDIA_CORRUPT
    facts = facts_from_metadata(metadata)
    if facts.duration_seconds <= 0:
        return None, SOURCE_MEDIA_INVALID_DURATION
    if facts.audio_codec is None:
        return None, SOURCE_AUDIO_STREAM_MISSING
    return facts, None


__all__ = [
    "Prober",
    "SourceMediaFailure",
    "SourceMediaPreflight",
    "facts_from_mapping",
    "facts_from_metadata",
    "resolve_managed_source",
    "run_source_media_preflight",
    "source_media_identity",
]

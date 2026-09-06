"""Stable content fingerprints for pipeline dependencies."""
import hashlib
import json
from collections.abc import Mapping


def canonical_fingerprint(namespace: str, version: str, payload: Mapping[str, object]) -> str:
    body = {"namespace": namespace, "version": version, "payload": payload}
    return hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.settings import Settings


def test_settings_uses_storage_root_from_explicit_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    storage_root = tmp_path / "clipfactory-storage"
    monkeypatch.setenv("CLIPFACTORY_STORAGE_ROOT", str(storage_root))

    settings = Settings()

    assert settings.storage_root == storage_root


def test_settings_rejects_non_positive_upload_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CLIPFACTORY_STORAGE_ROOT", str(tmp_path / "storage"))
    monkeypatch.setenv("CLIPFACTORY_MAX_UPLOAD_BYTES", "0")

    with pytest.raises(ValidationError, match="greater than 0"):
        Settings()

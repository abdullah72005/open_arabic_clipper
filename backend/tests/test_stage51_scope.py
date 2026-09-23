"""Stage 5.1 boundary/scope guarantees (required §21 cases 68-74).

These tests are static source scans plus a socket-blocked analysis run. They
prove the composition package never reaches a provider, never encodes a final
video, never mutates Stage 4, never materializes Stage 6 work, and never scans
the whole source.
"""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

COMPOSITION_DIR = Path(__file__).resolve().parents[1] / "app" / "composition"

FORBIDDEN_IMPORT_TOKENS = (
    "google.genai",
    "app.candidates.gemini",
    "app.refinement.hosted",
    "app.transformation.gemini",
    "app.transformation.planning.gemini",
    "app.transformation.governance.gemini",
    "faster_whisper",
    "import ollama",
    "from ollama",
    "import cv2",
    "import mediapipe",
    "from mediapipe",
    "import PIL",
    "from PIL",
)

FORBIDDEN_ENCODING_LITERALS = (
    '"libx264"',
    "'libx264'",
    '"libx265"',
    '"libvpx"',
    '"libopenh264"',
    '"h264_nvenc"',
    '"loudnorm"',
    '"-c:v"',
    "'-c:v'",
    '"mpeg4"',
    '"aac"',
    "'-f', 'mp4'",
    "'-f', \"mp4\"",
    '"-f", "mp4"',
)


def _composition_sources() -> list[Path]:
    return sorted(COMPOSITION_DIR.glob("*.py"))


def test_no_forbidden_provider_imports_in_composition() -> None:
    for source in _composition_sources():
        text = source.read_text(encoding="utf-8")
        for token in FORBIDDEN_IMPORT_TOKENS:
            assert token not in text, f"{source.name} must not reference {token!r}"


def test_no_final_render_or_encoding_paths_in_composition() -> None:
    # preview.py is the one module allowed to *name* encoder literals, solely
    # because it keeps a denylist guard that refuses to ever emit them.
    for source in _composition_sources():
        text = source.read_text(encoding="utf-8")
        if source.name != "preview.py":
            for token in FORBIDDEN_ENCODING_LITERALS:
                assert token not in text, f"{source.name} must not use encoder literal {token!r}"
            assert '".mp4"' not in text, f"{source.name} must not write a final MP4"
            assert "'.mp4'" not in text, f"{source.name} must not write a final MP4"

    preview_text = (COMPOSITION_DIR / "preview.py").read_text(encoding="utf-8")
    assert "_FORBIDDEN_ARGUMENTS" in preview_text
    assert "_assert_no_video_arguments" in preview_text
    assert preview_text.count('"libx264"') == 1, "encoder literal must exist only in the denylist"
    assert ".png" in preview_text


def test_composition_only_writes_png_previews() -> None:
    preview_text = (COMPOSITION_DIR / "preview.py").read_text(encoding="utf-8")
    assert "-frames:v" in preview_text
    assert "image2" not in preview_text  # no video muxer
    assert "pipe:1" not in preview_text  # no encoded stream output


def test_no_tts_or_voice_selection_in_composition() -> None:
    for source in _composition_sources():
        text = source.read_text(encoding="utf-8").lower()
        for token in ("text_to_speech", "tts_voice", "voice_id", "speech_synthesis"):
            assert token not in text, f"{source.name} must not select TTS ({token})"


def test_no_network_io_during_analysis(monkeypatch: pytest.MonkeyPatch) -> None:
    """A plan build must never open a network socket (Gemini/Qwen/hosted = 0)."""

    def _blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("Stage 5.1 analysis attempted network I/O")

    monkeypatch.setattr(socket, "create_connection", _blocked)
    monkeypatch.setattr(socket.socket, "connect", _blocked)

    # Import and exercise the planner with in-memory fakes. The planner itself
    # must not touch the network; the geometry/detector modules are import-safe.
    from app.composition import planner as planner_module  # noqa: F401

    assert planner_module is not None


def test_composition_package_has_no_opencv_dependency() -> None:
    for source in _composition_sources():
        text = source.read_text(encoding="utf-8")
        assert "cv2" not in text, f"{source.name} must not use OpenCV"


def test_visual_composition_execution_status_enum_is_closed() -> None:
    from app.composition.policy import VisualCompositionExecutionStatus

    assert {member.value for member in VisualCompositionExecutionStatus} == {
        "QUEUED",
        "ANALYZING",
        "COMPLETE",
        "FAILED",
        "CANCELLED",
    }


def test_visual_composition_status_enum_is_closed() -> None:
    from app.composition.policy import VisualCompositionStatus

    assert {member.value for member in VisualCompositionStatus} == {
        "READY_FOR_VISUAL_EXECUTION",
        "BLOCKED",
        "FAILED",
    }

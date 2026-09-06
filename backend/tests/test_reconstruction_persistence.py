from app.core.enums import JobKind, PipelineStage
from app.models import Transcript


def test_transcript_declares_separate_stage_2_7_derived_fields() -> None:
    """Reconstruction persistence cannot replace raw or Stage 2.5 transcript evidence."""

    columns = Transcript.__table__.c

    assert "contextual_reconstructed_text" in columns
    assert "reconstruction_fingerprint" in columns
    assert "reconstruction_metadata" in columns
    assert "reconstruction_status" in columns
    assert PipelineStage.CONTEXTUAL_RECONSTRUCTION.value == "CONTEXTUAL_RECONSTRUCTION"
    assert JobKind.RECONSTRUCTION.value == "RECONSTRUCTION"

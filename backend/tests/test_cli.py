from typer.testing import CliRunner

from app.cli import app


def test_stage_2_transcript_commands_are_exposed() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "transcribe" in result.stdout
    assert "retranscribe" in result.stdout
    assert "reconstruct" in result.stdout
    assert "benchmark-reconstruction" in result.stdout
    assert "transcript" in result.stdout

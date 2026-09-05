import json

import pytest

from app.transcription.correction import ContextualCorrector
from app.transcription.providers import (
    CorrectionRequest,
    OpenAICompatibleCorrectionProvider,
    ProviderCorrection,
    ProviderResponseError,
    validate_provider_results,
)


def test_contextual_provider_receives_bounded_windows_and_can_correct_target() -> None:
    """Provider sees local context but returns an annotation only for each stable ID."""

    class RecordingProvider:
        def __init__(self) -> None:
            self.requests: list[object] = []

        def correct_batch(self, requests: list[object]) -> list[ProviderCorrection]:
            self.requests = requests
            return [
                ProviderCorrection(
                    segment_index=request.segment_index,
                    corrected_text=("خلي بالك" if request.segment_index == 2 else request.raw_text),
                    changed=request.segment_index == 2,
                    confidence=0.96,
                    changes=[],
                )
                for request in requests
            ]

    provider = RecordingProvider()
    corrections = ContextualCorrector.from_default_lexicon(provider=provider).correct(
        [
            {"start": float(index), "end": float(index + 1), "text": text}
            for index, text in enumerate(["عامل إيه", "يا جماعة", "خطي بالك", "الموضوع", "مش سهل"])
        ]
    )

    request = provider.requests[2]
    assert request.previous == ("عامل إيه", "يا جماعة")
    assert request.raw_text == "خطي بالك"
    assert request.following == ("الموضوع", "مش سهل")
    assert corrections[2].corrected_text == "خلي بالك"
    assert corrections[2].method == "llm+lexicon"


def test_validator_rejects_missing_duplicate_or_unrequested_segment_ids() -> None:
    """A provider cannot merge, reorder, or silently omit persistent transcript segments."""

    response = [
        ProviderCorrection(0, "أهلا", False, 0.99, []),
        ProviderCorrection(0, "أهلا", False, 0.99, []),
    ]

    with pytest.raises(ProviderResponseError, match="duplicate"):
        validate_provider_results({0, 1}, response)


def test_openai_compatible_provider_uses_structured_json_prompt() -> None:
    """Local compatible endpoints receive the dialect-preserving correction contract."""

    captured: dict[str, object] = {}

    def request(url: str, body: bytes, headers: dict[str, str], timeout: float) -> bytes:
        captured.update(url=url, body=body, headers=headers, timeout=timeout)
        return json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "corrections": [
                                        {
                                            "segment_id": 0,
                                            "corrected_text": "خلي بالك",
                                            "changed": True,
                                            "confidence": 0.96,
                                            "changes": [
                                                {
                                                    "from": "خطي بالك",
                                                    "to": "خلي بالك",
                                                    "reason": "phonetic ASR correction",
                                                }
                                            ],
                                        }
                                    ]
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            },
            ensure_ascii=False,
        ).encode()

    provider = OpenAICompatibleCorrectionProvider(
        base_url="http://ollama:11434",
        model="qwen-local",
        api_key=None,
        timeout_seconds=12.0,
        request=request,
    )
    result = provider.correct_batch(
        [
            CorrectionRequest(
                segment_index=0,
                previous=(),
                raw_text="خطي بالك",
                following=(),
            )
        ]
    )

    assert result[0].corrected_text == "خلي بالك"
    assert captured["url"] == "http://ollama:11434/v1/chat/completions"
    payload = json.loads(captured["body"])
    assert payload["response_format"] == {"type": "json_object"}
    assert "Preserve the speaker's exact meaning and dialect." in payload["messages"][0]["content"]

from app.transcription.reconstruction.entities import build_entity_memory


def test_entity_memory_never_accepts_unseen_canonical_name() -> None:
    """Provider fluency cannot introduce a longer person name absent from source evidence."""

    memory = build_entity_memory(
        [
            {"text": "قابلت جاكوب امبارح"},
            {"text": "United Fruit Company سنة 1950"},
        ]
    )

    assert memory.supports_change("جاكوب", "جاكوب أربنز", (0,)) is False
    assert memory.contains_exact("United Fruit Company") is True


def test_entity_memory_accepts_observed_repeated_form_only() -> None:
    """Consistency uses a source form with real supporting segment IDs."""

    segments = [
        {"text": "الرئيس جاكوب أربنز وصل"},
        {"text": "قال جاكوب أربنز إن التجربة مستمرة"},
    ]
    memory = build_entity_memory(segments)

    assert memory.supports_change("جاكوب ارينز", "جاكوب أربنز", (0, 1)) is True
    assert memory.with_observed_nomination("رئيس غير موجود", (0,), segments) is memory

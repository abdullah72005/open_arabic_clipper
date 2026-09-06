from app.transcription.reconstruction.routing import RoutingConfig, RoutingPriority, route_segment
def test_multiple_weak_words_force_reconstruction_priority():
    decision = route_segment({"text": "كلام", "words": [{"word": "كلام", "probability": .52}, {"word": "مصري", "probability": .99}, {"word": "جديد", "probability": .53}]}, RoutingConfig())
    assert decision.priority is RoutingPriority.RECONSTRUCT
    assert {span.text for span in decision.focus_spans} == {"كلام", "جديد"}
def test_high_confidence_arabic_gets_context_check():
    assert route_segment({"text": "ده كلام", "words": [{"word": "ده", "probability": .98}, {"word": "كلام", "probability": .98}]}, language="ar").priority is RoutingPriority.CONTEXT_CHECK

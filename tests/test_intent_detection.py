from app.intent.intent_detector import detect_intent


def test_detect_intent_transfer():
    assert detect_intent("123456789", 500000) == "TRANSFER"


def test_detect_intent_unknown_when_missing_account():
    assert detect_intent(None, 500000) == "UNKNOWN"

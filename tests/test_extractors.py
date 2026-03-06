from app.extraction.amount_detector import detect_largest_amount, extract_amount_candidates
from app.extraction.bank_detector import detect_bank
from app.extraction.entity_extractor import extract_entities
from app.intent.intent_detector import detect_intent


def test_bank_detection_normalization():
    text = "Chuyen khoan toi MB de thanh toan"
    assert detect_bank(text) == "MB BANK"


def test_amount_detection_largest():
    text = "So tien 50,000 va tong 2.500.000 VND"
    assert detect_largest_amount(text) == 2500000
    assert 50000 in extract_amount_candidates(text)


def test_intent_detection_rule():
    assert detect_intent("12345678", 100000) == "TRANSFER"
    assert detect_intent(None, 100000) == "UNKNOWN"


def test_entity_extraction_account_and_amount():
    raw = "MB BANK\nSo TK 123456789\nSo tien 3,300,000\nNoi dung: Thanh toan"
    lines = raw.split("\n")
    entities = extract_entities(raw_text=raw, lines=lines)
    assert entities["accountNumber"] == "123456789"
    assert entities["amount"] == 3300000
    assert entities["bank"] == "MB BANK"

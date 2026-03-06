from app.nlp.ner_model import TransferNER


def test_ner_extract_fallback_fields():
    ner = TransferNER(model_path="data/models/model_not_exist")
    text = "Chuyen 2,500,000 vao stk 123456789 MB Bank noi dung thanh toan"
    entities = ner.extract(text)

    assert entities["bank"] is not None
    assert entities["accountNumber"] == "123456789"
    assert entities["amount"] == 2500000

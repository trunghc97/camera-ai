from app.evaluation.metrics import compute_metrics


def test_compute_metrics_basic():
    gold = [{"BANK": "MB", "ACCOUNT_NUMBER": "12345678", "ACCOUNT_NAME": "NGUYEN VAN A", "AMOUNT": "2000000", "DESCRIPTION": "thanh toan"}]
    pred = [{"BANK": "MB", "ACCOUNT_NUMBER": "12345678", "ACCOUNT_NAME": "NGUYEN VAN A", "AMOUNT": "2,000,000", "DESCRIPTION": "thanh toan"}]
    metrics = compute_metrics(gold, pred)
    assert metrics["overall"]["f1"] == 1.0

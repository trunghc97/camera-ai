import json
import sys
from pathlib import Path

# Allow running as: python app/nlp/evaluate_ner.py
sys.path.append(str(Path(__file__).resolve().parents[2]))

from app.evaluation.metrics import LABELS, compute_metrics, extract_gold_values
from app.nlp.ner_model import TransferNER

TEST_PATH = Path("data/dataset/vietnamese_banking/test.json")
MODEL_PATH = Path("data/models/ner_model")
OUT_PATH = Path("data/benchmark/ner_evaluation.json")


def evaluate_ner(test_path: str = TEST_PATH.as_posix(), model_path: str = MODEL_PATH.as_posix()) -> dict:
    path = Path(test_path)
    if not path.exists():
        raise FileNotFoundError(f"Test dataset not found: {path}")

    samples = json.loads(path.read_text(encoding="utf-8"))
    ner = TransferNER(model_path=model_path)

    gold_rows = []
    pred_rows = []

    for sample in samples:
        gold = extract_gold_values(sample)
        pred = ner.extract(sample["text"])
        pred_rows.append(
            {
                "BANK": pred.get("bank"),
                "ACCOUNT_NUMBER": pred.get("accountNumber"),
                "ACCOUNT_NAME": pred.get("accountName"),
                "AMOUNT": pred.get("amount"),
                "DESCRIPTION": pred.get("description"),
            }
        )
        gold_rows.append(gold)

    metrics = compute_metrics(gold_rows, pred_rows)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("NER Evaluation")
    print("---------------------------------------")
    for label in LABELS:
        m = metrics["per_entity"][label]
        print(f"{label:<16} P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}")
    print("---------------------------------------")
    print(f"OVERALL F1: {metrics['overall']['f1']:.4f}")

    return metrics


if __name__ == "__main__":
    evaluate_ner()

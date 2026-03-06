import json
import sys
from pathlib import Path

# Allow running as: python app/evaluation/benchmark.py
sys.path.append(str(Path(__file__).resolve().parents[2]))

from app.evaluation.metrics import LABELS, compute_metrics, extract_gold_values
from app.extraction.entity_extractor import extract_entities
from app.nlp.ner_model import TransferNER

TEST_PATH = Path("data/dataset/vietnamese_banking/test.json")
OUT_PATH = Path("data/benchmark/results.json")
MODEL_PATH = Path("data/models/ner_model")


def _regex_predict(text: str) -> dict:
    ent = extract_entities(raw_text=text, lines=[text])
    return {
        "BANK": ent.get("bank"),
        "ACCOUNT_NUMBER": ent.get("accountNumber"),
        "ACCOUNT_NAME": ent.get("accountName"),
        "AMOUNT": ent.get("amount"),
        "DESCRIPTION": ent.get("description"),
    }


def _ner_predict(ner: TransferNER, text: str) -> dict:
    ent = ner.extract(text)
    return {
        "BANK": ent.get("bank"),
        "ACCOUNT_NUMBER": ent.get("accountNumber"),
        "ACCOUNT_NAME": ent.get("accountName"),
        "AMOUNT": ent.get("amount"),
        "DESCRIPTION": ent.get("description"),
    }


def _hybrid_predict(regex_pred: dict, ner_pred: dict) -> dict:
    merged = dict(regex_pred)
    for key, value in ner_pred.items():
        if value not in (None, ""):
            merged[key] = value

    # Sanity: if amount equals account number, prefer regex amount.
    if (
        merged.get("AMOUNT") is not None
        and merged.get("ACCOUNT_NUMBER") is not None
        and str(merged["AMOUNT"]) == str(merged["ACCOUNT_NUMBER"])
        and regex_pred.get("AMOUNT") is not None
    ):
        merged["AMOUNT"] = regex_pred.get("AMOUNT")

    return merged


def _summary_row(name: str, metrics: dict) -> dict:
    return {
        "model": name,
        "precision": metrics["overall"]["precision"],
        "recall": metrics["overall"]["recall"],
        "f1": metrics["overall"]["f1"],
    }


def run_benchmark(test_path: str = TEST_PATH.as_posix(), model_path: str = MODEL_PATH.as_posix()) -> dict:
    path = Path(test_path)
    if not path.exists():
        raise FileNotFoundError(f"Test dataset not found: {path}")

    samples = json.loads(path.read_text(encoding="utf-8"))
    ner = TransferNER(model_path=model_path)

    gold_rows = []
    regex_rows = []
    ner_rows = []
    hybrid_rows = []

    for sample in samples:
        text = sample["text"]
        gold_rows.append(extract_gold_values(sample))

        regex_pred = _regex_predict(text)
        ner_pred = _ner_predict(ner, text)
        hybrid_pred = _hybrid_predict(regex_pred, ner_pred)

        regex_rows.append(regex_pred)
        ner_rows.append(ner_pred)
        hybrid_rows.append(hybrid_pred)

    regex_metrics = compute_metrics(gold_rows, regex_rows)
    ner_metrics = compute_metrics(gold_rows, ner_rows)
    hybrid_metrics = compute_metrics(gold_rows, hybrid_rows)

    result = {
        "summary": [
            _summary_row("Regex", regex_metrics),
            _summary_row("NER", ner_metrics),
            _summary_row("Hybrid", hybrid_metrics),
        ],
        "details": {
            "Regex": regex_metrics,
            "NER": ner_metrics,
            "Hybrid": hybrid_metrics,
        },
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("MODEL            PRECISION   RECALL   F1")
    print("-----------------------------------------")
    for row in result["summary"]:
        print(f"{row['model']:<16} {row['precision']:<10.4f} {row['recall']:<8.4f} {row['f1']:<.4f}")

    return result


if __name__ == "__main__":
    run_benchmark()

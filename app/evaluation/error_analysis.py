import json
import sys
from collections import defaultdict
from pathlib import Path

# Allow running as: python app/evaluation/error_analysis.py
sys.path.append(str(Path(__file__).resolve().parents[2]))

from app.evaluation.benchmark import _hybrid_predict, _ner_predict, _regex_predict
from app.evaluation.metrics import LABELS, extract_gold_values, normalize_value
from app.nlp.ner_model import TransferNER

TEST_PATH = Path("data/dataset/vietnamese_banking/test.json")
OUT_PATH = Path("data/benchmark/error_report.json")
MODEL_PATH = Path("data/models/ner_model")


def generate_error_report(test_path: str = TEST_PATH.as_posix(), model_path: str = MODEL_PATH.as_posix()) -> dict:
    samples = json.loads(Path(test_path).read_text(encoding="utf-8"))
    ner = TransferNER(model_path=model_path)

    issue_counts = defaultdict(int)
    examples = defaultdict(list)

    for sample in samples:
        text = sample["text"]
        gold = extract_gold_values(sample)

        regex_pred = _regex_predict(text)
        ner_pred = _ner_predict(ner, text)
        pred = _hybrid_predict(regex_pred, ner_pred)

        def add_issue(name: str):
            issue_counts[name] += 1
            if len(examples[name]) < 20:
                examples[name].append(
                    {
                        "text": text,
                        "gold": gold,
                        "pred": pred,
                    }
                )

        g_bank = normalize_value("BANK", gold.get("BANK"))
        p_bank = normalize_value("BANK", pred.get("BANK"))
        if g_bank and not p_bank:
            add_issue("missing_bank")

        g_amount = normalize_value("AMOUNT", gold.get("AMOUNT"))
        p_amount = normalize_value("AMOUNT", pred.get("AMOUNT"))
        if g_amount and p_amount and g_amount != p_amount:
            add_issue("incorrect_amount_detection")

        g_acc = normalize_value("ACCOUNT_NUMBER", gold.get("ACCOUNT_NUMBER"))
        p_acc = normalize_value("ACCOUNT_NUMBER", pred.get("ACCOUNT_NUMBER"))
        if g_acc and p_acc and g_acc != p_acc:
            add_issue("account_number_confusion")

        g_name = normalize_value("ACCOUNT_NAME", gold.get("ACCOUNT_NAME"))
        p_name = normalize_value("ACCOUNT_NAME", pred.get("ACCOUNT_NAME"))
        if g_name and not p_name:
            add_issue("missing_account_name")

        g_desc = normalize_value("DESCRIPTION", gold.get("DESCRIPTION"))
        p_desc = normalize_value("DESCRIPTION", pred.get("DESCRIPTION"))
        if g_desc and not p_desc:
            add_issue("missing_description")

    report = {
        "issue_counts": dict(sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)),
        "sample_count": len(samples),
        "examples": dict(examples),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Error analysis saved:", OUT_PATH)
    for key, value in report["issue_counts"].items():
        print(f"- {key}: {value}")

    return report


if __name__ == "__main__":
    generate_error_report()

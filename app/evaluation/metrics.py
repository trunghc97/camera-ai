import re
from collections import defaultdict

LABELS = ["BANK", "ACCOUNT_NUMBER", "ACCOUNT_NAME", "AMOUNT", "DESCRIPTION"]


def normalize_text(value: str | int | None) -> str | None:
    if value is None:
        return None
    return " ".join(str(value).strip().split())


def normalize_amount(value: str | int | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower().replace("vnd", "").replace("vnđ", "")
    text = text.replace(" ", "")

    if not text:
        return None

    # Vietnamese shorthands.
    multiplier = 1
    if text.endswith("trieu"):
        multiplier = 1_000_000
        text = text[:-5]
    elif text.endswith("tr"):
        multiplier = 1_000_000
        text = text[:-2]
    elif text.endswith("k"):
        multiplier = 1_000
        text = text[:-1]

    digits = re.sub(r"\D", "", text)
    if not digits:
        return None

    return str(int(digits) * multiplier)


def normalize_value(label: str, value: str | int | None) -> str | None:
    if value is None:
        return None

    if label == "AMOUNT":
        return normalize_amount(value)

    if label == "ACCOUNT_NUMBER":
        digits = re.sub(r"\D", "", str(value))
        return digits or None

    normalized = normalize_text(value)
    if normalized is None:
        return None

    if label == "BANK":
        return normalized.upper()

    if label == "ACCOUNT_NAME":
        return normalized.upper()

    return normalized.lower()


def extract_gold_values(sample: dict) -> dict[str, str | None]:
    text = sample["text"]
    out = {label: None for label in LABELS}
    for ent in sample.get("entities", []):
        label = ent.get("label")
        if label not in out:
            continue
        start, end = ent.get("start", 0), ent.get("end", 0)
        if 0 <= start < end <= len(text):
            out[label] = text[start:end]
    return out


def compute_metrics(gold_rows: list[dict[str, str | None]], pred_rows: list[dict[str, str | int | None]]) -> dict:
    counts = {label: defaultdict(int) for label in LABELS}
    overall = defaultdict(int)

    for gold, pred in zip(gold_rows, pred_rows):
        for label in LABELS:
            g = normalize_value(label, gold.get(label))
            p = normalize_value(label, pred.get(label))

            if p is not None and g is not None and p == g:
                counts[label]["tp"] += 1
                overall["tp"] += 1
            else:
                if p is not None:
                    counts[label]["fp"] += 1
                    overall["fp"] += 1
                if g is not None:
                    counts[label]["fn"] += 1
                    overall["fn"] += 1

    per_entity = {}
    for label in LABELS:
        tp, fp, fn = counts[label]["tp"], counts[label]["fp"], counts[label]["fn"]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_entity[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }

    otp, ofp, ofn = overall["tp"], overall["fp"], overall["fn"]
    op = otp / (otp + ofp) if (otp + ofp) else 0.0
    or_ = otp / (otp + ofn) if (otp + ofn) else 0.0
    of1 = (2 * op * or_ / (op + or_)) if (op + or_) else 0.0

    return {
        "per_entity": per_entity,
        "overall": {
            "precision": round(op, 4),
            "recall": round(or_, 4),
            "f1": round(of1, 4),
            "tp": otp,
            "fp": ofp,
            "fn": ofn,
        },
    }

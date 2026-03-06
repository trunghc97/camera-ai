import re
from typing import Any

from app.extraction.amount_detector import detect_largest_amount, extract_amount_candidates
from app.extraction.bank_detector import detect_bank

ACCOUNT_PATTERN = re.compile(r"(?<!\d)(\d{8,20})(?!\d)")


def _extract_account_name(lines: list[str]) -> str | None:
    for i, line in enumerate(lines):
        upper_line = line.upper()
        if any(key in upper_line for key in ["CHU TAI KHOAN", "TEN TAI KHOAN", "ACCOUNT NAME"]):
            # Prefer text after ':' in the same line.
            if ":" in line:
                right = line.split(":", 1)[1].strip()
                if right:
                    return right.upper()
            # Fallback to next non-empty line.
            if i + 1 < len(lines) and lines[i + 1].strip():
                return lines[i + 1].strip().upper()

    # Heuristic fallback: uppercase alpha line with at least 2 words.
    for line in lines:
        text = line.strip()
        if len(text.split()) >= 2 and re.fullmatch(r"[A-Z\s.]+", text.upper()):
            return text.upper()
    return None


def _extract_description(lines: list[str]) -> str | None:
    for i, line in enumerate(lines):
        upper_line = line.upper()
        if any(key in upper_line for key in ["NOI DUNG", "DIEN GIAI", "DESCRIPTION", "REMARK"]):
            if ":" in line:
                right = line.split(":", 1)[1].strip()
                if right:
                    return right
            if i + 1 < len(lines) and lines[i + 1].strip():
                return lines[i + 1].strip()
    return None


def extract_entities(raw_text: str, lines: list[str]) -> dict[str, Any]:
    account_matches = ACCOUNT_PATTERN.findall(raw_text)
    account_number = max(account_matches, key=len) if account_matches else None

    amount = detect_largest_amount(raw_text)
    amount_candidates = extract_amount_candidates(raw_text)

    return {
        "accountNumber": account_number,
        "accountName": _extract_account_name(lines),
        "bank": detect_bank(raw_text),
        "amount": amount,
        "amountCandidates": amount_candidates,
        "description": _extract_description(lines),
    }

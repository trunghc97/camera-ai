import re
from typing import List, Optional


AMOUNT_PATTERN = re.compile(r"(?<!\d)(\d{1,3}(?:[.,\s]\d{3})+|\d{4,15})(?!\d)")
AMOUNT_HINTS = ("SO TIEN", "AMOUNT", "VND", "VNĐ", "TRANSFER", "CHUYEN", "THANH TOAN")
ACCOUNT_HINTS = ("STK", "SO TK", "TAI KHOAN", "ACCOUNT", "ACC")


def _to_int(value: str) -> Optional[int]:
    digits = re.sub(r"\D", "", value)
    if not digits:
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def extract_amount_candidates(raw_text: str) -> List[int]:
    candidates = []
    for token in AMOUNT_PATTERN.findall(raw_text):
        parsed = _to_int(token)
        if parsed is not None:
            candidates.append(parsed)

    # Remove very small values that are likely noise, keep meaningful transfer amounts.
    return sorted({x for x in candidates if x >= 1000})


def _extract_amount_candidates_from_line(line: str) -> list[int]:
    values: list[int] = []
    upper = line.upper()
    for match in AMOUNT_PATTERN.finditer(line):
        token = match.group(1)
        left_context = upper[max(0, match.start() - 18):match.start()]
        # Skip tokens likely tied to account identifiers.
        if any(h in left_context for h in ACCOUNT_HINTS):
            continue
        parsed = _to_int(token)
        if parsed is not None and parsed >= 1000:
            values.append(parsed)
    return values


def detect_largest_amount(raw_text: str) -> Optional[int]:
    # Prefer values on lines that semantically indicate amount.
    hinted_candidates: list[int] = []
    for line in raw_text.splitlines():
        if any(h in line.upper() for h in AMOUNT_HINTS):
            hinted_candidates.extend(_extract_amount_candidates_from_line(line))

    if hinted_candidates:
        return max(hinted_candidates)

    candidates = extract_amount_candidates(raw_text)
    return max(candidates) if candidates else None

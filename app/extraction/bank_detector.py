from typing import Optional

BANK_ALIASES = {
    "MB": "MB BANK",
    "MB BANK": "MB BANK",
    "VIETCOMBANK": "VIETCOMBANK",
    "TECHCOMBANK": "TECHCOMBANK",
    "BIDV": "BIDV",
    "AGRIBANK": "AGRIBANK",
    "ACB": "ACB",
    "SACOMBANK": "SACOMBANK",
    "VPBANK": "VPBANK",
    "TPBANK": "TPBANK",
    "SHB": "SHB",
    "OCB": "OCB",
    "VIB": "VIB",
    "SCB": "SCB",
    "SEABANK": "SEABANK",
}


def detect_bank(raw_text: str) -> Optional[str]:
    upper_text = raw_text.upper()
    for alias, canonical in BANK_ALIASES.items():
        if alias in upper_text:
            return canonical
    return None

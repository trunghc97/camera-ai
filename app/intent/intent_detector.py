def detect_intent(account_number: str | None, amount: int | None) -> str:
    # Transfer intent is considered valid only when both account and amount exist.
    if account_number and amount:
        return "TRANSFER"
    return "UNKNOWN"

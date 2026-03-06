def plan_action(intent: str, entities: dict) -> dict | None:
    if intent != "TRANSFER":
        return None

    return {
        "screen": "TRANSFER_SCREEN",
        "fields": {
            "account": entities.get("accountNumber"),
            "bank": entities.get("bank"),
            "amount": entities.get("amount"),
            "description": entities.get("description"),
        },
    }

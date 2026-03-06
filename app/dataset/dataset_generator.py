import json
import random
from pathlib import Path

BANKS = [
    "MB Bank",
    "Vietcombank",
    "Techcombank",
    "BIDV",
    "Agribank",
    "ACB",
    "Sacombank",
]

DESCRIPTIONS = [
    "thanh toan hoa don",
    "chuyen tien hoc phi",
    "tra no",
    "mua hang online",
    "chuyen khoan gia dinh",
    "payment for invoice",
    "monthly rent",
    "service payment",
    "refund order",
    "utility bill",
]

TEMPLATES = [
    "Chuyen {amount_text} vao stk {account} {bank} {description}",
    "Thanh toan {description} {amount_text} VND toi {bank} {account}",
    "Transfer {amount_text} to account {account} {bank} for {description}",
    "Xin chuyen khoan {amount_text} den tai khoan {account} ngan hang {bank} {description}",
    "Please transfer {amount_text} VND to {bank} {account} note {description}",
]


def _random_amount() -> int:
    return random.randint(50_000, 50_000_000)


def _amount_text(amount: int) -> str:
    style = random.choice(["plain", "comma", "dot", "word"])
    if style == "plain":
        return str(amount)
    if style == "comma":
        return f"{amount:,}"
    if style == "dot":
        return f"{amount:,}".replace(",", ".")

    million = amount // 1_000_000
    if million >= 1:
        return f"{million} trieu"
    return f"{amount // 1000} nghin"


def _account_number() -> str:
    length = random.randint(8, 14)
    return "".join(random.choices("0123456789", k=length))


def generate_record() -> dict:
    bank = random.choice(BANKS)
    account = _account_number()
    amount = _random_amount()
    description = random.choice(DESCRIPTIONS)

    text = random.choice(TEMPLATES).format(
        amount_text=_amount_text(amount),
        account=account,
        bank=bank,
        description=description,
    )

    return {
        "text": text,
        "entities": {
            "bank": bank,
            "account_number": account,
            "amount": str(amount),
            "description": description,
        },
    }


def generate_dataset(output_path: str = "data/dataset/train.json", size: int = 10_000, seed: int = 42) -> Path:
    random.seed(seed)
    records = [generate_record() for _ in range(size)]

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


if __name__ == "__main__":
    path = generate_dataset()
    print(f"Generated dataset at: {path}")

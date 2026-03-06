import json
import random
from pathlib import Path

OUTPUT_DIR = Path("data/dataset/vietnamese_banking")
BANKS_PATH = Path("data/banks_vietnam.json")

NAMES = [
    "Nguyen Van A",
    "Tran Thi B",
    "Le Minh C",
    "Pham Gia Huy",
    "Doan Thu Trang",
    "Vo Quoc Bao",
    "Bui Khanh Linh",
    "Hoang Anh Tuan",
    "Dinh Thi Mai",
    "Nguyen Hoang Nam",
]

DESCRIPTIONS = [
    "thanh toan hoa don",
    "tra no tien hang",
    "hoc phi thang 3",
    "dat coc don hang",
    "phi dich vu",
    "payment invoice 2026",
    "rent payment",
    "ho tro gia dinh",
    "chuyen tien mua may",
    "refund order",
]

STYLES = ["chat", "invoice", "request", "mixed"]


def _load_banks() -> list[str]:
    if BANKS_PATH.exists():
        return json.loads(BANKS_PATH.read_text(encoding="utf-8"))
    return ["MB", "VIETCOMBANK", "TECHCOMBANK", "BIDV"]


def _account_number() -> str:
    return "".join(random.choices("0123456789", k=random.randint(8, 14)))


def _amount_value() -> int:
    return random.randint(50_000, 90_000_000)


def _amount_text(amount: int, style: str) -> str:
    if style == "chat":
        chat_type = random.choice(["k", "tr", "raw"])
        if chat_type == "k":
            return f"{max(1, amount // 1000)}k"
        if chat_type == "tr":
            return f"{max(1, amount // 1_000_000)} tr"
        return str(amount)

    if style == "mixed":
        return random.choice([
            f"{amount:,}",
            f"{amount:,}".replace(",", "."),
            f"{amount} VND",
        ])

    if style == "invoice":
        return f"{amount:,}".replace(",", ".")

    return f"{amount:,}"


def _compose_sentence(style: str, amount_text: str, account: str, bank: str, name: str, description: str) -> str:
    templates = {
        "chat": [
            "Chuyen {amount} vao stk {account} {bank} {name} {description}",
            "ck {amount} den tk {account} {bank} {name} {description}",
            "chuyen gap {amount} qua {bank} {account} cho {name} {description}",
        ],
        "invoice": [
            "Thanh toan hoa don: {description}. So tien: {amount}. Ngan hang: {bank}. So TK: {account}. Ten TK: {name}",
            "Yeu cau thanh toan {description} | Amount {amount} | Bank {bank} | Account {account} | Beneficiary {name}",
        ],
        "request": [
            "Ban vui long chuyen khoan {amount} toi tai khoan {account} ngan hang {bank} chu TK {name}, noi dung {description}",
            "De nghi transfer {amount} to {bank} account {account} account name {name} for {description}",
        ],
        "mixed": [
            "Transfer {amount} to {bank} {account} {name} note {description}",
            "Thanh toan {amount} VND toi {bank} - {account} ({name}) noi dung {description}",
        ],
    }
    return random.choice(templates[style]).format(
        amount=amount_text,
        account=account,
        bank=bank,
        name=name,
        description=description,
    )


def _find_span(text: str, value: str) -> tuple[int, int]:
    start = text.lower().find(value.lower())
    if start < 0:
        raise ValueError(f"Cannot find span for value '{value}' in '{text}'")
    return start, start + len(value)


def generate_record(banks: list[str]) -> dict:
    style = random.choice(STYLES)
    bank = random.choice(banks)
    account = _account_number()
    amount_int = _amount_value()
    amount_text = _amount_text(amount_int, style)
    name = random.choice(NAMES)
    description = random.choice(DESCRIPTIONS)

    text = _compose_sentence(style, amount_text, account, bank, name, description)

    entities = []
    for label, value in [
        ("AMOUNT", amount_text),
        ("ACCOUNT_NUMBER", account),
        ("BANK", bank),
        ("ACCOUNT_NAME", name),
        ("DESCRIPTION", description),
    ]:
        start, end = _find_span(text, value)
        entities.append({"label": label, "start": start, "end": end})

    return {"text": text, "entities": entities}


def _split_dataset(records: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    n = len(records)
    train_end = int(n * 0.70)
    val_end = train_end + int(n * 0.15)
    return records[:train_end], records[train_end:val_end], records[val_end:]


def generate_vietnamese_dataset(
    size: int = 20_000,
    seed: int = 42,
    output_dir: str = OUTPUT_DIR.as_posix(),
) -> dict[str, int]:
    random.seed(seed)
    banks = _load_banks()
    records = [generate_record(banks) for _ in range(size)]
    random.shuffle(records)

    train, val, test = _split_dataset(records)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train.json").write_text(json.dumps(train, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "validation.json").write_text(json.dumps(val, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "test.json").write_text(json.dumps(test, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"train": len(train), "validation": len(val), "test": len(test)}


if __name__ == "__main__":
    stats = generate_vietnamese_dataset()
    print(f"Generated Vietnamese banking dataset: {stats}")

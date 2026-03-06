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
DOCUMENT_TYPES = ["invoice", "debt", "text", "qr", "document"]

DEBT_TERMS = [
    "cong no ky truoc",
    "thanh toan cong no don hang",
    "doi soat cong no",
    "thu hoi cong no",
    "hoan ung cong no",
]


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


def _compose_document(
    doc_type: str,
    amount_text: str,
    account: str,
    bank: str,
    name: str,
    description: str,
) -> str:
    if doc_type == "invoice":
        return (
            "HOA DON THANH TOAN\n"
            f"So tien can chuyen: {amount_text}\n"
            f"Ngan hang huong thu: {bank}\n"
            f"So tai khoan: {account}\n"
            f"Chu tai khoan: {name}\n"
            f"Noi dung chuyen khoan: {description}"
        )

    if doc_type == "debt":
        debt_term = random.choice(DEBT_TERMS)
        return (
            "BIEN BAN DOI SOAT CONG NO\n"
            f"Khoan muc: {debt_term}\n"
            f"So tien thanh toan: {amount_text}\n"
            f"Thong tin nhan tien: {bank} - {account}\n"
            f"Ten nguoi nhan: {name}\n"
            f"Dien giai: {description}"
        )

    if doc_type == "qr":
        return (
            "THANH TOAN QR\n"
            f"payload=bank:{bank};acc:{account};name:{name};amount:{amount_text};desc:{description}\n"
            f"Quet ma QR va chuyen {amount_text}"
        )

    if doc_type == "document":
        return (
            "DE NGHI THANH TOAN CHUYEN KHOAN\n"
            "Kinh gui phong ke toan,\n"
            f"Vui long chuyen khoan so tien {amount_text} vao tai khoan {account} tai {bank}.\n"
            f"Nguoi thu huong: {name}.\n"
            f"Noi dung: {description}.\n"
            "Tran trong."
        )

    # text: generic free-form style.
    return _compose_sentence(
        style=random.choice(STYLES),
        amount_text=amount_text,
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


def _label_noise(text: str) -> str:
    replacements = {
        "so tai khoan": "stk",
        "ngan hang": random.choice(["ngan hang", "bank", "n.hang"]),
        "chu tai khoan": random.choice(["chu tk", "ten tk", "account name"]),
        "noi dung chuyen khoan": random.choice(["noi dung", "note", "description"]),
        "so tien": random.choice(["so tien", "amount", "tong tien"]),
    }
    noisy = text
    for src, dst in replacements.items():
        if random.random() < 0.35:
            noisy = noisy.replace(src, dst).replace(src.title(), dst)
    return noisy


def generate_record(banks: list[str], document_type: str | None = None) -> dict:
    style = random.choice(STYLES)
    bank = random.choice(banks)
    account = _account_number()
    amount_int = _amount_value()
    amount_text = _amount_text(amount_int, style)
    name = random.choice(NAMES)
    description = random.choice(DESCRIPTIONS)

    doc_type = document_type or random.choice(DOCUMENT_TYPES)
    text = _compose_document(
        doc_type=doc_type,
        amount_text=amount_text,
        account=account,
        bank=bank,
        name=name,
        description=description,
    )
    text = _label_noise(text)

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

    return {"text": text, "entities": entities, "document_type": doc_type}


def _split_dataset(records: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    n = len(records)
    train_end = int(n * 0.70)
    val_end = train_end + int(n * 0.15)
    return records[:train_end], records[train_end:val_end], records[val_end:]


def generate_vietnamese_dataset(
    size: int = 20_000,
    seed: int = 42,
    output_dir: str = OUTPUT_DIR.as_posix(),
    enforce_doc_balance: bool = True,
) -> dict[str, int]:
    random.seed(seed)
    banks = _load_banks()

    records: list[dict] = []
    if enforce_doc_balance and size >= len(DOCUMENT_TYPES):
        per_type = size // len(DOCUMENT_TYPES)
        for doc_type in DOCUMENT_TYPES:
            records.extend(generate_record(banks, document_type=doc_type) for _ in range(per_type))
        while len(records) < size:
            records.append(generate_record(banks))
    else:
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

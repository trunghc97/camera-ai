import logging
import re
from pathlib import Path

import spacy
from spacy.language import Language

from app.extraction.amount_detector import detect_largest_amount
from app.extraction.bank_detector import detect_bank
from config import settings

logger = logging.getLogger(__name__)

BANKS = [
    "MB",
    "MB BANK",
    "VIETCOMBANK",
    "TECHCOMBANK",
    "BIDV",
    "AGRIBANK",
    "ACB",
    "SACOMBANK",
    "VPBANK",
    "TPBANK",
    "SHB",
    "OCB",
    "VIB",
    "SCB",
    "SEABANK",
]
ACCOUNT_RE = re.compile(r"(?<!\d)(\d{8,20})(?!\d)")


class TransferNER:
    def __init__(self, model_path: str = settings.ner_model_path):
        self.model_path = Path(model_path)
        self.nlp = self._load_or_build()

    def _load_or_build(self) -> Language:
        if self.model_path.exists():
            logger.info("Loading trained NER model from %s", self.model_path)
            return spacy.load(self.model_path.as_posix())

        # Fallback runtime model for local POC when no trained artifact exists yet.
        nlp = spacy.blank("xx")
        ruler = nlp.add_pipe("entity_ruler")
        patterns = []
        for bank in BANKS:
            patterns.append({"label": "BANK", "pattern": bank})
        ruler.add_patterns(patterns)
        return nlp

    @staticmethod
    def _to_int(value: str) -> int | None:
        digits = re.sub(r"\D", "", value)
        return int(digits) if digits else None

    def extract(self, text: str) -> dict:
        doc = self.nlp(text)
        entities: dict[str, str | int | None] = {
            "bank": None,
            "accountNumber": None,
            "accountName": None,
            "amount": None,
            "description": None,
        }

        for ent in doc.ents:
            if ent.label_ == "BANK" and not entities["bank"]:
                entities["bank"] = ent.text.upper()
            elif ent.label_ == "ACCOUNT_NUMBER" and not entities["accountNumber"]:
                entities["accountNumber"] = ent.text
            elif ent.label_ == "ACCOUNT_NAME" and not entities["accountName"]:
                entities["accountName"] = ent.text.upper()
            elif ent.label_ == "AMOUNT" and not entities["amount"]:
                amount = self._to_int(ent.text)
                if amount:
                    entities["amount"] = amount
            elif ent.label_ == "DESCRIPTION" and not entities["description"]:
                entities["description"] = ent.text

        # Runtime safety net in case model is not trained yet.
        if not entities["bank"]:
            entities["bank"] = detect_bank(text)

        if not entities["accountNumber"]:
            match = ACCOUNT_RE.search(text)
            if match:
                entities["accountNumber"] = match.group(1)

        if not entities["amount"]:
            entities["amount"] = detect_largest_amount(text)

        if not entities["description"]:
            low = text.lower()
            for key in ["noi dung", "description", "dien giai", "note", "for"]:
                pos = low.find(key)
                if pos >= 0:
                    candidate = text[pos + len(key):].strip(" :.-")
                    if candidate:
                        entities["description"] = candidate
                        break

        return entities


ner_model = TransferNER()

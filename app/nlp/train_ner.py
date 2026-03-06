import json
import random
from pathlib import Path

import spacy
from spacy.training import Example

DATA_PATH = Path("data/dataset/vietnamese_banking/train.json")
MODEL_PATH = Path("data/models/ner_model")

LABEL_MAP = {
    "bank": "BANK",
    "account_number": "ACCOUNT_NUMBER",
    "account_name": "ACCOUNT_NAME",
    "amount": "AMOUNT",
    "description": "DESCRIPTION",
}


def _build_train_examples(dataset: list[dict]) -> list[tuple[str, dict]]:
    examples: list[tuple[str, dict]] = []

    for row in dataset:
        text = row["text"]
        spans: list[tuple[int, int, str]] = []

        # New span-format dataset support.
        if isinstance(row.get("entities"), list):
            for item in row["entities"]:
                label = item.get("label")
                start = int(item.get("start", -1))
                end = int(item.get("end", -1))
                if label in LABEL_MAP.values() and 0 <= start < end <= len(text):
                    spans.append((start, end, label))
        else:
            # Backward compatibility with key-value entity format.
            entities_block = row.get("entities", {})
            for key, label in LABEL_MAP.items():
                value = entities_block.get(key)
                if not value:
                    continue

                start = text.lower().find(str(value).lower())
                if start < 0:
                    continue
                end = start + len(str(value))
                spans.append((start, end, label))

        if spans:
            examples.append((text, {"entities": spans}))

    return examples


def train_ner(dataset_path: str = DATA_PATH.as_posix(), model_path: str = MODEL_PATH.as_posix(), n_iter: int = 15):
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    dataset = json.loads(path.read_text(encoding="utf-8"))
    train_data = _build_train_examples(dataset)
    if not train_data:
        raise ValueError("No trainable examples found in dataset")

    nlp = spacy.blank("xx")
    ner = nlp.add_pipe("ner")
    for label in LABEL_MAP.values():
        ner.add_label(label)

    optimizer = nlp.begin_training()

    for i in range(n_iter):
        random.shuffle(train_data)
        losses = {}
        for text, annotations in train_data:
            example = Example.from_dict(nlp.make_doc(text), annotations)
            nlp.update([example], drop=0.2, sgd=optimizer, losses=losses)
        print(f"Iteration {i + 1}/{n_iter} - losses: {losses}")

    out = Path(model_path)
    out.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(out)
    print(f"Saved NER model to: {out}")


if __name__ == "__main__":
    train_ner()

import json
from pathlib import Path

from app.dataset.vietnamese_dataset_generator import generate_vietnamese_dataset


def test_generate_vietnamese_dataset(tmp_path):
    out_dir = tmp_path / "vn"
    stats = generate_vietnamese_dataset(size=100, seed=11, output_dir=out_dir.as_posix())
    assert stats["train"] + stats["validation"] + stats["test"] == 100

    train = json.loads((Path(out_dir) / "train.json").read_text(encoding="utf-8"))
    assert len(train) == stats["train"]
    assert "text" in train[0]
    assert "entities" in train[0]

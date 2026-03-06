import json

from app.dataset.dataset_generator import generate_dataset


def test_generate_dataset(tmp_path):
    out = tmp_path / "train.json"
    generate_dataset(output_path=out.as_posix(), size=120, seed=7)

    data = json.loads(out.read_text(encoding="utf-8"))
    assert len(data) == 120
    sample = data[0]
    assert "text" in sample
    assert "entities" in sample
    assert {"bank", "account_number", "amount", "description"}.issubset(sample["entities"].keys())

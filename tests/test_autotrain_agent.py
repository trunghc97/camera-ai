import json
from pathlib import Path

from app.agent.autotrain_agent import AutoTrainAgent, AutoTrainConfig


def test_autotrain_agent_stops_when_target_met(tmp_path, monkeypatch):
    dataset_dir = tmp_path / "dataset"
    model_root = tmp_path / "models"
    report_path = tmp_path / "benchmark" / "report.json"

    def fake_generate(size, seed, output_dir, enforce_doc_balance=True):
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = [{"text": "chuyen 100000 toi 12345678 MB", "entities": []}]
        (out_dir / "train.json").write_text(json.dumps(payload), encoding="utf-8")
        (out_dir / "test.json").write_text(json.dumps(payload), encoding="utf-8")
        return {"train": 1, "validation": 0, "test": 1}

    trained_paths: list[str] = []

    def fake_train(dataset_path, model_path, n_iter, dropout, batch_size, seed):
        trained_paths.append(model_path)

    calls = {"n": 0}

    def fake_benchmark(test_path, model_path):
        calls["n"] += 1
        f1 = 0.88 if calls["n"] == 1 else 0.93
        return {
            "details": {
                "NER": {"overall": {"f1": f1 - 0.01}},
                "Hybrid": {"overall": {"f1": f1}},
            }
        }

    monkeypatch.setattr("app.agent.autotrain_agent.generate_vietnamese_dataset", fake_generate)
    monkeypatch.setattr("app.agent.autotrain_agent.train_ner", fake_train)
    monkeypatch.setattr("app.agent.autotrain_agent.run_benchmark", fake_benchmark)

    cfg = AutoTrainConfig(
        target_f1=0.92,
        max_rounds=5,
        dataset_dir=dataset_dir.as_posix(),
        model_root=model_root.as_posix(),
        report_path=report_path.as_posix(),
    )

    report = AutoTrainAgent(cfg).run()

    assert report["status"] == "success"
    assert len(report["rounds"]) == 2
    assert report["best"]["scores"]["hybrid_f1"] == 0.93
    assert len(trained_paths) == 2
    assert report_path.exists()


def test_autotrain_agent_marks_needs_improvement(tmp_path, monkeypatch):
    dataset_dir = tmp_path / "dataset"
    model_root = tmp_path / "models"
    report_path = tmp_path / "benchmark" / "report.json"

    def fake_generate(size, seed, output_dir, enforce_doc_balance=True):
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = [{"text": "stub", "entities": []}]
        (out_dir / "train.json").write_text(json.dumps(payload), encoding="utf-8")
        (out_dir / "test.json").write_text(json.dumps(payload), encoding="utf-8")
        return {"train": 1, "validation": 0, "test": 1}

    def fake_train(dataset_path, model_path, n_iter, dropout, batch_size, seed):
        return None

    def fake_benchmark(test_path, model_path):
        return {
            "details": {
                "NER": {"overall": {"f1": 0.85}},
                "Hybrid": {"overall": {"f1": 0.89}},
            }
        }

    monkeypatch.setattr("app.agent.autotrain_agent.generate_vietnamese_dataset", fake_generate)
    monkeypatch.setattr("app.agent.autotrain_agent.train_ner", fake_train)
    monkeypatch.setattr("app.agent.autotrain_agent.run_benchmark", fake_benchmark)

    cfg = AutoTrainConfig(
        target_f1=0.92,
        max_rounds=2,
        dataset_dir=dataset_dir.as_posix(),
        model_root=model_root.as_posix(),
        report_path=report_path.as_posix(),
    )

    report = AutoTrainAgent(cfg).run()

    assert report["status"] == "needs_improvement"
    assert len(report["rounds"]) == 2

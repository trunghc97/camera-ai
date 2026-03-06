import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

# Allow running as: python app/agent/autotrain_agent.py
sys.path.append(str(Path(__file__).resolve().parents[2]))

from app.dataset.vietnamese_dataset_generator import generate_vietnamese_dataset
from app.evaluation.benchmark import run_benchmark
from app.nlp.train_ner import train_ner

DEFAULT_DATASET_DIR = Path("data/dataset/vietnamese_banking")
DEFAULT_MODEL_ROOT = Path("data/models/auto_ner")
DEFAULT_REPORT_PATH = Path("data/benchmark/auto_train_report.json")


@dataclass
class RoundConfig:
    n_iter: int
    dropout: float
    batch_size: int
    seed: int


@dataclass
class AutoTrainConfig:
    target_f1: float = 0.92
    max_rounds: int = 6
    base_dataset_size: int = 20_000
    dataset_growth: int = 2_000
    dataset_seed: int = 42
    dataset_dir: str = DEFAULT_DATASET_DIR.as_posix()
    model_root: str = DEFAULT_MODEL_ROOT.as_posix()
    report_path: str = DEFAULT_REPORT_PATH.as_posix()


class AutoTrainAgent:
    def __init__(self, config: AutoTrainConfig):
        self.config = config

    @staticmethod
    def _round_profiles() -> list[RoundConfig]:
        return [
            RoundConfig(n_iter=16, dropout=0.2, batch_size=24, seed=42),
            RoundConfig(n_iter=20, dropout=0.18, batch_size=28, seed=77),
            RoundConfig(n_iter=24, dropout=0.15, batch_size=32, seed=111),
            RoundConfig(n_iter=28, dropout=0.12, batch_size=36, seed=150),
            RoundConfig(n_iter=30, dropout=0.10, batch_size=40, seed=201),
            RoundConfig(n_iter=32, dropout=0.08, batch_size=44, seed=301),
        ]

    def run(self) -> dict:
        dataset_dir = Path(self.config.dataset_dir)
        model_root = Path(self.config.model_root)
        report_path = Path(self.config.report_path)

        dataset_dir.mkdir(parents=True, exist_ok=True)
        model_root.mkdir(parents=True, exist_ok=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)

        profiles = self._round_profiles()
        round_results: list[dict] = []
        best: dict | None = None

        for i in range(self.config.max_rounds):
            round_no = i + 1
            profile = profiles[min(i, len(profiles) - 1)]
            dataset_size = self.config.base_dataset_size + (i * self.config.dataset_growth)
            round_model_path = model_root / f"round_{round_no}"

            stats = generate_vietnamese_dataset(
                size=dataset_size,
                seed=self.config.dataset_seed + i,
                output_dir=dataset_dir.as_posix(),
                enforce_doc_balance=True,
            )

            train_ner(
                dataset_path=(dataset_dir / "train.json").as_posix(),
                model_path=round_model_path.as_posix(),
                n_iter=profile.n_iter,
                dropout=profile.dropout,
                batch_size=profile.batch_size,
                seed=profile.seed,
            )

            bench = run_benchmark(
                test_path=(dataset_dir / "test.json").as_posix(),
                model_path=round_model_path.as_posix(),
            )

            hybrid_f1 = bench["details"]["Hybrid"]["overall"]["f1"]
            ner_f1 = bench["details"]["NER"]["overall"]["f1"]

            result = {
                "round": round_no,
                "dataset_size": dataset_size,
                "dataset_stats": stats,
                "train": asdict(profile),
                "scores": {
                    "ner_f1": ner_f1,
                    "hybrid_f1": hybrid_f1,
                },
                "model_path": round_model_path.as_posix(),
                "target_met": hybrid_f1 >= self.config.target_f1,
            }
            round_results.append(result)

            if best is None or hybrid_f1 > best["scores"]["hybrid_f1"]:
                best = result

            if result["target_met"]:
                break

        report = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": asdict(self.config),
            "best": best,
            "rounds": round_results,
            "status": "success" if best and best["scores"]["hybrid_f1"] >= self.config.target_f1 else "needs_improvement",
        }

        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Auto train/test agent for transfer extraction")
    parser.add_argument("--target-f1", type=float, default=0.92)
    parser.add_argument("--max-rounds", type=int, default=6)
    parser.add_argument("--base-size", type=int, default=20_000)
    parser.add_argument("--growth", type=int, default=2_000)
    parser.add_argument("--dataset-seed", type=int, default=42)
    parser.add_argument("--dataset-dir", type=str, default=DEFAULT_DATASET_DIR.as_posix())
    parser.add_argument("--model-root", type=str, default=DEFAULT_MODEL_ROOT.as_posix())
    parser.add_argument("--report-path", type=str, default=DEFAULT_REPORT_PATH.as_posix())
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = AutoTrainConfig(
        target_f1=args.target_f1,
        max_rounds=args.max_rounds,
        base_dataset_size=args.base_size,
        dataset_growth=args.growth,
        dataset_seed=args.dataset_seed,
        dataset_dir=args.dataset_dir,
        model_root=args.model_root,
        report_path=args.report_path,
    )
    report = AutoTrainAgent(cfg).run()
    best = report.get("best") or {}
    best_f1 = (best.get("scores") or {}).get("hybrid_f1")
    print("AutoTrain finished")
    print(f"status={report['status']}")
    print(f"best_hybrid_f1={best_f1}")
    print(f"report={cfg.report_path}")


if __name__ == "__main__":
    main()

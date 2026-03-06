# CAMERA AI - Transfer Detection System

Backend POC for extracting banking transfer information from images and text.

## Pipeline
Image -> Preprocessing -> OCR -> Layout detection -> NER extraction -> Regex fallback -> Intent detection -> Action planner -> JSON

## New Vietnamese NER Accuracy Stack
- Vietnamese banking dataset split: `data/dataset/vietnamese_banking/`
- NER evaluation: precision/recall/F1 per entity + overall F1
- Benchmark suite: Regex vs NER vs Hybrid
- Error analysis report for common extraction failures

## Key Files
- API: `main.py`, `app/api/extract_api.py`
- Vietnamese dataset generator: `app/dataset/vietnamese_dataset_generator.py`
- NER training: `app/nlp/train_ner.py`
- NER evaluation: `app/nlp/evaluate_ner.py`
- Auto train agent: `app/agent/autotrain_agent.py`
- Benchmark: `app/evaluation/benchmark.py`
- Error analysis: `app/evaluation/error_analysis.py`
- Bank dictionary: `data/banks_vietnam.json`

## Docker Workflow
1. Build containers
```bash
docker compose build
```

2. Start services
```bash
docker compose up -d
```

3. Generate Vietnamese dataset (20,000 samples)
```bash
docker compose exec app python app/dataset/vietnamese_dataset_generator.py
```

4. Train NER model
```bash
docker compose exec app python app/nlp/train_ner.py
```

5. Evaluate NER model
```bash
docker compose exec app python app/nlp/evaluate_ner.py
```

6. Run benchmark (Regex vs NER vs Hybrid)
```bash
docker compose exec app python app/evaluation/benchmark.py
```

7. Run error analysis
```bash
docker compose exec app python app/evaluation/error_analysis.py
```

8. Run auto-train agent (self-generate dataset, self-train, self-test until target)
```bash
docker compose exec app python app/agent/autotrain_agent.py --target-f1 0.92 --max-rounds 6 --base-size 20000 --growth 2000
```

9. Run API
```bash
docker compose exec app uvicorn main:app --host 0.0.0.0 --port 8000
```

10. Inference test
```bash
curl -F "file=@data/images/sample_transfer.jpg" http://localhost:8000/extract
```

## CLI Commands (Local)
```bash
python app/dataset/vietnamese_dataset_generator.py
python app/nlp/train_ner.py
python app/nlp/evaluate_ner.py
python app/evaluation/benchmark.py
python app/evaluation/error_analysis.py
python app/agent/autotrain_agent.py --target-f1 0.92 --max-rounds 6 --base-size 20000 --growth 2000
```

## Outputs
- Dataset: `data/dataset/vietnamese_banking/train.json`, `validation.json`, `test.json`
- Trained model: `data/models/ner_model`
- NER evaluation: `data/benchmark/ner_evaluation.json`
- Benchmark: `data/benchmark/results.json`
- Auto train report: `data/benchmark/auto_train_report.json`
- Error report: `data/benchmark/error_report.json`

## Benchmark Target
- Hybrid extraction target: `F1 > 0.90`

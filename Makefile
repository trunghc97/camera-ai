.PHONY: build up down dataset dataset-vn train eval benchmark error-report test run infer logs auto-train

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

dataset:
	docker compose exec app python app/dataset/dataset_generator.py

dataset-vn:
	docker compose exec app python app/dataset/vietnamese_dataset_generator.py

train:
	docker compose exec app python app/nlp/train_ner.py

eval:
	docker compose exec app python app/nlp/evaluate_ner.py

benchmark:
	docker compose exec app python app/evaluation/benchmark.py

error-report:
	docker compose exec app python app/evaluation/error_analysis.py

test:
	docker compose exec app pytest -q

run:
	docker compose exec -d app uvicorn main:app --host 0.0.0.0 --port 8000

infer:
	curl -F "file=@data/images/sample_transfer.jpg" http://localhost:8000/extract

logs:
	docker compose logs -f app

auto-train:
	docker compose exec app python app/agent/autotrain_agent.py --target-f1 0.92 --max-rounds 6 --base-size 20000 --growth 2000

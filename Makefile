export PYTHONPATH := src

.PHONY: setup setup-full mlflow test pipeline data train eval drift register serve seed clean

setup:
	pip install -r requirements-local.txt

setup-full:
	pip install -r requirements.txt

mlflow:
	mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlruns/mlflow.db

test:
	pytest tests -q

data:
	dvc repro split

pipeline:
	dvc repro

train:
	python -m nastaliqops.models.train_lora

eval:
	python -m nastaliqops.models.evaluate

drift:
	python -m nastaliqops.monitoring.drift

register:
	python -m nastaliqops.models.register

serve:
	docker compose up --build

seed:
	python scripts/seed_stream.py 60

clean:
	rm -rf data/processed data/splits artifacts

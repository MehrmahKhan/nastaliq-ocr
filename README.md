# NastaliqOps

Urdu (Nastaliq) OCR with an LLM correction agent and a human in the loop,
wrapped in a reproducible MLOps pipeline.

The goal of this project is not state of the art accuracy. The goal is a
pipeline you can run twice and get the same thing, where every dataset,
every run, every prompt and every model version is tracked, and where a bad
model cannot reach production by accident.

## What it actually does

You upload a photo of a printed Urdu line. A small fine tuned vision
encoder decoder reads it and reports how sure it is. An agent then decides
what to do with that answer:

| Confidence | What happens |
|---|---|
| high | the text is returned as is, no LLM is called |
| medium | an LLM corrector repairs the line, then a verify step checks the repair |
| low | the line goes straight to the human review queue |

Every correction a reviewer types is stored as a new labelled row, and
those rows are what the next training run learns from. That is the loop.

## Why it is built this way

A plain fine tune and a FastAPI endpoint would be a 2022 project. The parts
that make this an LLMOps and AgentOps project are:

* The LLM is a **component with a version**, not a chat window. Prompts
  live in `prompts/` as versioned YAML and every call is traced.
* The routing is **cost aware**. Calling the LLM on every line is easy and
  wasteful, so the router only pays for it when the small model is unsure,
  and the report measures what that saves.
* The agent loop has **governance**. It retries once, then escalates to a
  human. It cannot loop forever and it cannot silently invent text.
* The promotion gate compares a challenger against a frozen golden set and
  refuses to promote a model that is not actually better.

## Stack

| Layer | Tool |
|---|---|
| Data versioning | DVC, remote on DagsHub (S3 compatible) or AWS S3 |
| Data validation | Pandera schema, gates the pipeline on failure rate |
| Pipeline orchestration | DVC stages (`dvc repro`) |
| Modelling | PyTorch, Hugging Face Transformers, TrOCR style encoder decoder |
| Parameter efficient tuning | PEFT LoRA adapters |
| Experiment tracking | MLflow, hosted on DagsHub |
| Model registry | MLflow registry, aliases Production and Challenger, version tags |
| Agent framework | LangGraph |
| LLM serving | any OpenAI compatible endpoint (Groq, Gemini, local Ollama) |
| Prompt versioning and tracing | Langfuse |
| Evaluation | CER, WER, exact match, plus LLM as a judge |
| Drift monitoring | Evidently, over image features and served confidence |
| API | FastAPI |
| Frontend | single page review and upload UI, RTL aware |
| Containers | Docker, Docker Compose |
| CI/CD | GitHub Actions |
| Service monitoring | Prometheus, Grafana |
| Cloud | AWS EC2, S3, ECR |
| Testing | pytest, run in CI |

## Repository layout

```
params.yaml                 every tunable value, logged to mlflow on each run
dvc.yaml                    the pipeline, stage by stage
src/nastaliqops/
  config.py                 loads params and env
  data/download.py          stage 1, pull the dataset subset
  data/columns.py           works out which columns the dataset actually has
  data/validate.py          stage 2, quality gate
  data/schema.py            the pandera schema the gate enforces
  data/preprocess.py        stage 3, the shared image pipeline
  data/split.py             stage 4, train / val / golden / stream
  monitoring/features.py    scalar image features drift is measured on
  monitoring/drift.py       stage 7, evidently drift report
  models/train_lora.py      stage 5, LoRA fine tune with mlflow logging
  models/evaluate.py        stage 6, golden set evaluation of every variant
  models/register.py        registry and promotion gate
  models/predictor.py       one loader used by both eval and serving
  llm/client.py             provider agnostic LLM wrapper with tracing
  llm/corrector.py          the post correction step
  llm/judge.py              LLM as a judge
  llm/graph.py              the LangGraph agent loop
  review_store.py           human in the loop database
  utils/metrics.py          CER and WER written out by hand
  utils/urdu.py             Urdu normalisation
api/main.py                 FastAPI service
frontend/index.html         the UI
prompts/                    versioned prompts
tests/                      unit tests that run in CI
```

## Dataset

UTRSet-Real from the UTRNet paper (ICDAR 2023), about 11,000 annotated
printed Urdu line images, plus UTRSet-Synth if more volume is needed. It is
licensed CC BY-NC 4.0, so research and coursework use is fine as long as it
is cited.

> Rahman, Ghosh and Arora. UTRNet: High Resolution Urdu Text Recognition in
> Printed Documents. ICDAR 2023.

Only a subset is pulled, set by `data.subset_size` in `params.yaml`, because
the training has to fit in a free Colab session.

The split is deliberate. `golden` is frozen and nothing trains on it, so
promotion decisions mean something. `stream` is held back to act as live
uploads, so drift monitoring and the review queue have real traffic to work
with before anyone else is using the app.

## Running it

### Where each piece runs

The heavy parts are deliberately kept off the development laptop, which
also happens to be the right architecture: training belongs on a GPU and
the API belongs on the cloud instance the course requires anyway.

| Machine | What runs there | Install |
|---|---|---|
| laptop | data stages, DVC, MLflow UI, registry gate | `requirements-local.txt` |
| Colab (free T4) | training and evaluation | `requirements.txt` |
| AWS EC2 | the API and the frontend | the Docker image |

### 1. Setup

```bash
git clone https://github.com/MehrmahKhan/nastaliq-ocr.git
cd nastaliq-ocr
python -m venv .venv && source .venv/bin/activate

pip install -r requirements-local.txt   # laptop: no torch, under a gigabyte
# pip install -r requirements.txt        # colab or docker: the full set

cp .env.example .env     # then fill in the keys
export PYTHONPATH=src    # windows powershell: $env:PYTHONPATH="src"
```

On a machine short of disk space, lower `data.subset_size` in
`params.yaml` as well. 1500 lines is enough to exercise every stage.

### 2. Start MLflow

```bash
docker compose up -d mlflow
```

Or without Docker, which is lighter and does the same job:

```bash
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlruns/mlflow.db
```

Either way the UI is at http://localhost:5000.

### 3. Point DVC at a remote

The easy route is DagsHub. One free account gives you both a DVC remote
and a hosted MLflow server, so there is no bucket to create and no MLflow
process to run locally. Create a repo there called `nastaliq-ops`, put the
username and token in `.env`, then:

```bash
bash scripts/setup_dagshub.sh
```

If you would rather use AWS directly:

```bash
dvc init
dvc remote add -d storage s3://<your-bucket>/dvcstore
dvc remote modify storage region ap-south-1
```

### 4. Run the pipeline

```bash
dvc repro            # ingest, validate, preprocess, split, train, evaluate, drift
dvc push             # data and artifacts go to the remote
git add . && git commit -m "pipeline run"
```

On a laptop without a GPU, run only the stages that do not need torch:

```bash
dvc repro split      # stops after the data is prepared
dvc repro drift      # the evidently report, also torch free
```

Training on CPU is painfully slow. Run `dvc repro` up to the `split` stage
locally, push, then open the Colab notebook, pull the data with
`dvc pull`, run the train stage on the T4, and push the adapter back.

### 5. Register and gate the model

```bash
python -m nastaliqops.models.register
```

Exit code 0 means the challenger was promoted to the Production alias. Exit
code 2 means it was registered but rejected, and the old model stays live.

### 6. Serve

```bash
docker compose up --build
```

* app and review UI: http://localhost:8000
* MLflow: http://localhost:5000
* Prometheus: http://localhost:9090
* Grafana: http://localhost:3001

## Phase 1 scope

Everything above is phase 1: versioned data, a reproducible pipeline,
tracked experiments, a registry with tagged adapter versions, the agent and
the serving layer running in Docker.

Phase 2 adds the parts that need a live deployment: Evidently drift
monitoring on the incoming image and confidence distributions, a retraining
job triggered by drift or by a batch of accumulated corrections, canary
rollout on EC2, and automatic rollback through the registry alias.

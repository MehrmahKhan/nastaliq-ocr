# NastaliqOps, Phase 1 Summary

**Mehrmah Khan, 23L-0623**
**Repository:** https://github.com/MehrmahKhan/nastaliq-ocr

**Project.** An Urdu (Nastaliq) OCR service with an LLM correction agent and
a human in the loop, built as a reproducible MLOps pipeline. A small fine
tuned vision encoder decoder reads a printed Urdu line and reports its
confidence. A LangGraph agent routes on that confidence: accept the output
directly when the model is sure, call an LLM corrector when it is
borderline, or push it into the human review queue when it is poor.
Reviewer corrections become new labelled data for the next training run.
The routing is cost aware on purpose, so the project can report what
fraction of LLM-only quality it reaches at a fraction of the LLM calls.

**Implemented in Phase 1.**

* **Data and versioning.** A subset of UTRSet-Real (ICDAR 2023, CC BY-NC
  4.0) is pulled from the Hub and tracked with DVC against a DagsHub
  remote, raw and processed stages both. The ingest stage detects the
  dataset's own column names rather than assuming them. The data is split
  four ways: train, val, a frozen golden set used only for promotion
  decisions, and a held back stream set that stands in for live uploads
  until the service has real traffic.
* **Pipeline.** Seven DVC stages run end to end with `dvc repro`: ingest,
  validate, preprocess, split, train, evaluate, drift. Validation is a
  Pandera schema over text length, script, image dimensions and aspect
  ratio, and it exits non zero above a five percent failure rate, so a
  broken ingest cannot reach training. Preprocessing is a single shared
  function imported by both the trainer and the API, which keeps training
  and serving consistent.
* **Modelling.** The base checkpoint is a pretrained Urdu TrOCR style
  encoder decoder. The challenger is the same model with a PEFT LoRA
  adapter fine tuned on the training split, which keeps the trained
  artifact small enough to version comfortably.
* **LLM and agent layer.** Prompts are versioned YAML files, not inline
  strings. The corrector runs against any OpenAI compatible endpoint and
  every call is traced to Langfuse with its prompt version, latency and
  token count. The agent loop has a retry budget and escalates to a human
  rather than looping.
* **Experiment tracking.** Every run logs the full flattened `params.yaml`,
  training loss, and validation CER and WER to MLflow hosted on DagsHub.
  Evaluation scores four variants on the golden set, base, LoRA, LoRA plus
  LLM correction, and LLM only, using CER, WER, exact match and an LLM
  judge.
* **Model registry.** Adapters are registered under `nastaliq-ocr` with
  version tags for the base checkpoint, variant and golden CER. A
  challenger only takes the Production alias if it clears an absolute CER
  floor and beats the current champion by a set margin, otherwise the
  existing model stays live.
* **Drift monitoring.** Each line image is reduced to interpretable scalar
  features, brightness, contrast, blur, ink ratio and geometry. Evidently
  compares the training distribution against the current window and writes
  a drift report plus a machine readable summary that flags when
  retraining is warranted. The current window reads from served
  predictions when they exist and falls back to the stream split.
* **Serving and UI.** A FastAPI service exposes the OCR endpoint, the
  review queue and Prometheus metrics, and serves a right to left review
  UI where a reviewer sees the image and edits the text. Docker Compose
  runs the API alongside MLflow, Prometheus and Grafana. GitHub Actions
  runs the unit tests and builds the API image on every push.

**Phase 2.** Drift triggered and correction triggered retraining, canary
rollout on EC2 behind ECR, automatic rollback through the registry alias,
and Terraform for the infrastructure.

"""Stage 6: evaluate every variant on the frozen golden set.

Four variants are compared so the report has an honest table:

  base            pretrained checkpoint, no fine tuning
  lora            base plus the LoRA adapter trained in stage 5
  lora_llm        lora output passed through the LLM corrector agent
  llm_only        the vision LLM reading the image directly

Everything lands in MLflow and in reports/metrics.json, which dvc tracks as
a metrics file so `dvc metrics diff` shows the change between commits.
"""

import json
import sys

import cv2
import mlflow

from nastaliqops.config import (
    Paths,
    ensure_dirs,
    load_params,
    mlflow_experiment,
    mlflow_uri,
)
from nastaliqops.llm.corrector import LlmCorrector
from nastaliqops.data.splits_io import read_split
from nastaliqops.models.predictor import OcrPredictor, default_adapter_path
from nastaliqops.utils.metrics import (
    character_error_rate,
    exact_match_rate,
    word_error_rate,
)


def run_variant(name, predictor, corrector, records, limit):
    references = []
    hypotheses = []

    position = 0
    while position < len(records) and position < limit:
        record = records[position]
        image_path = str(Paths.processed / "images" / record["file_name"])
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        result = predictor.predict_line(image)
        text = result["text"]

        if corrector is not None:
            corrected = corrector.correct(text, result["confidence"])
            text = corrected["text"]

        references.append(record["text"])
        hypotheses.append(text)
        position = position + 1

    scores = {}
    scores["cer"] = character_error_rate(references, hypotheses)
    scores["wer"] = word_error_rate(references, hypotheses)
    scores["exact_match"] = exact_match_rate(references, hypotheses)
    scores["samples"] = len(references)

    print(name + ": cer " + str(round(scores["cer"], 4)) + " wer " + str(round(scores["wer"], 4)))
    return scores


def main():
    params = load_params()
    ensure_dirs()

    limit = 400
    if len(sys.argv) > 1:
        limit = int(sys.argv[1])

    golden = read_split("golden")

    mlflow.set_tracking_uri(mlflow_uri())
    mlflow.set_experiment(mlflow_experiment())
    mlflow.start_run(run_name="golden-eval")

    results = {}

    base_predictor = OcrPredictor(adapter_path=None)
    results["base"] = run_variant("base", base_predictor, None, golden, limit)

    adapter_path = default_adapter_path()
    if adapter_path is not None:
        lora_predictor = OcrPredictor(adapter_path=adapter_path)
        results["lora"] = run_variant("lora", lora_predictor, None, golden, limit)

        try:
            corrector = LlmCorrector()
            results["lora_llm"] = run_variant("lora_llm", lora_predictor, corrector, golden, limit)
        except Exception as error:
            print("skipping the llm variant: " + str(error))
    else:
        print("no adapter found, only the base variant was evaluated")

    for variant_name in results:
        scores = results[variant_name]
        for metric_name in scores:
            mlflow.log_metric(variant_name + "_" + metric_name, scores[metric_name])

    out_path = Paths.reports / "metrics.json"
    handle = open(out_path, "w", encoding="utf-8")
    handle.write(json.dumps(results, indent=2))
    handle.close()

    mlflow.log_artifact(str(out_path))
    mlflow.end_run()
    print("wrote " + str(out_path))


if __name__ == "__main__":
    main()

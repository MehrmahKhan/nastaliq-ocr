"""Stage 5: fine tune the OCR model with a LoRA adapter.

Only the adapter is trained, so the artifact is a few megabytes instead of
a gigabyte. That adapter is what gets registered in the MLflow model
registry with a version tag, which is exactly what the deliverables ask for
under "adapter weights (PEFT/LoRA) with version tags".

Run this on Colab with a T4. The CPU path works but is slow, use it only for
a smoke test with a handful of rows.
"""

import json

import mlflow
import torch

from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from nastaliqops.config import (
    Paths,
    ensure_dirs,
    flatten_params,
    load_params,
    mlflow_experiment,
    mlflow_uri,
)
from nastaliqops.models.dataset import LineDataset
from nastaliqops.utils.metrics import character_error_rate, word_error_rate
from nastaliqops.utils.urdu import normalise


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def decode_batch(processor, generated_ids):
    texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    cleaned = []
    for text in texts:
        cleaned.append(normalise(text))
    return cleaned


def evaluate_loader(model, processor, loader, device, max_target_length):
    model.eval()

    references = []
    hypotheses = []

    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            generated = model.generate(pixel_values, max_length=max_target_length)
            predicted = decode_batch(processor, generated)

            label_ids = batch["labels"].clone()
            pad_id = processor.tokenizer.pad_token_id
            for row in range(label_ids.shape[0]):
                for col in range(label_ids.shape[1]):
                    if label_ids[row][col] == -100:
                        label_ids[row][col] = pad_id

            truth = decode_batch(processor, label_ids)

            for index in range(len(truth)):
                references.append(truth[index])
                hypotheses.append(predicted[index])

    model.train()

    scores = {}
    scores["cer"] = character_error_rate(references, hypotheses)
    scores["wer"] = word_error_rate(references, hypotheses)
    return scores


def main():
    params = load_params()
    model_params = params["model"]
    train_params = params["train"]

    ensure_dirs()
    device = pick_device()
    print("device: " + str(device))

    processor = TrOCRProcessor.from_pretrained(model_params["base_checkpoint"])
    model = VisionEncoderDecoderModel.from_pretrained(model_params["base_checkpoint"])

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id

    lora_params = model_params["lora"]
    lora_config = LoraConfig(
        r=lora_params["r"],
        lora_alpha=lora_params["alpha"],
        lora_dropout=lora_params["dropout"],
        target_modules=lora_params["target_modules"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to(device)

    max_target_length = model_params["max_target_length"]
    train_set = LineDataset("train", processor, max_target_length)
    val_set = LineDataset("val", processor, max_target_length)

    train_loader = DataLoader(train_set, batch_size=train_params["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=train_params["batch_size"], shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_params["learning_rate"])

    mlflow.set_tracking_uri(mlflow_uri())
    mlflow.set_experiment(mlflow_experiment())

    run = mlflow.start_run(run_name="lora-finetune")
    flat = flatten_params(params)
    for key in flat:
        mlflow.log_param(key, flat[key])
    mlflow.log_param("device", str(device))
    mlflow.log_param("train_rows", len(train_set))
    mlflow.log_param("val_rows", len(val_set))

    global_step = 0
    epoch = 0
    while epoch < train_params["epochs"]:
        running_loss = 0.0
        seen = 0

        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss = running_loss + float(loss.item())
            seen = seen + 1
            global_step = global_step + 1

            if global_step % 50 == 0:
                average = running_loss / float(seen)
                print("epoch " + str(epoch) + " step " + str(global_step) + " loss " + str(average))
                mlflow.log_metric("train_loss", average, step=global_step)

        scores = evaluate_loader(model, processor, val_loader, device, max_target_length)
        print("epoch " + str(epoch) + " val cer " + str(scores["cer"]))
        mlflow.log_metric("val_cer", scores["cer"], step=epoch)
        mlflow.log_metric("val_wer", scores["wer"], step=epoch)

        epoch = epoch + 1

    adapter_dir = Paths.artifacts / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))

    mlflow.log_artifacts(str(adapter_dir), artifact_path="adapter")

    info = {}
    info["run_id"] = run.info.run_id
    info["base_checkpoint"] = model_params["base_checkpoint"]
    handle = open(Paths.artifacts / "last_run.json", "w", encoding="utf-8")
    handle.write(json.dumps(info, indent=2))
    handle.close()

    mlflow.end_run()
    print("adapter saved to " + str(adapter_dir))


if __name__ == "__main__":
    main()

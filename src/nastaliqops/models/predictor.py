"""Loads a model once and exposes predict_line.

Two modes. Load the base checkpoint only, which is the baseline, or load the
base plus the LoRA adapter, which is the challenger. The API and the
evaluation script both go through this class so the numbers in the report
and the numbers in production come from the same code path.
"""

import math

import cv2
import numpy as np
import torch

from peft import PeftModel
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from nastaliqops.config import Paths, load_params
from nastaliqops.data.preprocess import preprocess_image
from nastaliqops.utils.urdu import normalise


class OcrPredictor:
    def __init__(self, adapter_path=None):
        params = load_params()
        self.params = params
        self.max_target_length = params["model"]["max_target_length"]
        self.image_height = params["data"]["image_height"]
        self.max_width = params["data"]["max_width"]

        base = params["model"]["base_checkpoint"]
        self.processor = TrOCRProcessor.from_pretrained(base)
        model = VisionEncoderDecoderModel.from_pretrained(base)

        if adapter_path is not None:
            model = PeftModel.from_pretrained(model, adapter_path)
            model = model.merge_and_unload()
            self.variant = "lora"
        else:
            self.variant = "base"

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        model.to(self.device)
        model.eval()
        self.model = model

    def _prepare(self, image_bgr):
        processed = preprocess_image(image_bgr, self.image_height, self.max_width)
        rgb = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
        encoded = self.processor(images=rgb, return_tensors="pt")
        return encoded.pixel_values.to(self.device)

    def predict_line(self, image_bgr):
        """Return the text plus a confidence score in the range 0 to 1.

        Confidence is the mean token probability of the generated sequence.
        It is not calibrated, but it is monotonic enough to drive routing,
        and the thresholds in params.yaml are tuned against the golden set.
        """
        pixel_values = self._prepare(image_bgr)

        with torch.no_grad():
            output = self.model.generate(
                pixel_values,
                max_length=self.max_target_length,
                output_scores=True,
                return_dict_in_generate=True,
            )

        text = self.processor.batch_decode(output.sequences, skip_special_tokens=True)[0]
        text = normalise(text)

        total_log_prob = 0.0
        counted = 0
        sequence = output.sequences[0]
        # scores holds one entry per generated step. the token that step
        # produced normally sits one position further along in sequences,
        # because of the decoder start token. that is not guaranteed: some
        # model and transformers combinations return the two the same
        # length, and then the last step indexes one past the end. stop
        # rather than assume.
        for step_index in range(len(output.scores)):
            position = step_index + 1
            if position >= len(sequence):
                break
            logits = output.scores[step_index][0]
            probabilities = torch.softmax(logits, dim=-1)
            token_id = sequence[position]
            token_probability = float(probabilities[token_id])
            if token_probability <= 0.0:
                token_probability = 1e-9
            total_log_prob = total_log_prob + math.log(token_probability)
            counted = counted + 1

        if counted == 0:
            confidence = 0.0
        else:
            confidence = math.exp(total_log_prob / float(counted))

        result = {}
        result["text"] = text
        result["confidence"] = confidence
        result["variant"] = self.variant
        return result

    def predict_file(self, path):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        return self.predict_line(image)

    def predict_bytes(self, raw_bytes):
        buffer = np.frombuffer(raw_bytes, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        return self.predict_line(image)


def default_adapter_path():
    path = Paths.artifacts / "adapter"
    if path.exists():
        return str(path)
    return None

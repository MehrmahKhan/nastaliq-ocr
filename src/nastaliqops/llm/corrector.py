"""The LLM post correction step.

This is the piece that turns the project from plain supervised OCR into an
LLM assisted pipeline. The small fine tuned model does the reading, and the
language model repairs what the reader got wrong, but only when the reader
is unsure. Calling it on every line would be slow and expensive, which is
the whole point of the confidence router.
"""

from nastaliqops.config import load_params
from nastaliqops.llm.client import LlmClient, load_prompt
from nastaliqops.utils.urdu import normalise


REVIEW_TOKEN = "NEEDS_HUMAN_REVIEW"


class LlmCorrector:
    def __init__(self):
        params = load_params()
        self.params = params
        self.prompt = load_prompt(params["llm"]["corrector_prompt"])
        self.client = LlmClient()

    def correct(self, ocr_text, confidence):
        variables = {}
        variables["ocr_text"] = ocr_text
        variables["confidence"] = round(float(confidence), 3)

        reply = self.client.complete_json(self.prompt, variables, "urdu-ocr-correction")

        text = reply.get("text", ocr_text)
        needs_review = False

        if text is None or text.strip() == "":
            text = ocr_text
        if REVIEW_TOKEN in text:
            needs_review = True
            text = ocr_text

        result = {}
        result["text"] = normalise(text)
        result["original"] = ocr_text
        result["changed"] = bool(reply.get("changed", result["text"] != ocr_text))
        result["reason"] = reply.get("reason", "")
        result["needs_review"] = needs_review
        result["latency"] = reply.get("_latency", 0.0)
        result["tokens"] = reply.get("_tokens", 0)
        return result

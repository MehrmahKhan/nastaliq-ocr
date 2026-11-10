"""LLM as a judge, used in evaluation only, never on the serving path."""

from nastaliqops.config import load_params
from nastaliqops.llm.client import LlmClient, load_prompt


class LlmJudge:
    def __init__(self):
        params = load_params()
        self.prompt = load_prompt(params["llm"]["judge_prompt"])
        self.client = LlmClient()

    def score(self, reference, hypothesis):
        variables = {}
        variables["reference"] = reference
        variables["hypothesis"] = hypothesis

        reply = self.client.complete_json(self.prompt, variables, "urdu-ocr-judge")

        result = {}
        result["score"] = float(reply.get("score", 0))
        result["critical_error"] = bool(reply.get("critical_error", False))
        result["note"] = reply.get("note", "")
        return result

    def score_many(self, references, hypotheses):
        scores = []
        critical = 0

        for index in range(len(references)):
            single = self.score(references[index], hypotheses[index])
            scores.append(single["score"])
            if single["critical_error"]:
                critical = critical + 1

        total = 0.0
        for value in scores:
            total = total + value

        summary = {}
        if len(scores) == 0:
            summary["mean_score"] = 0.0
        else:
            summary["mean_score"] = total / float(len(scores))
        summary["critical_error_count"] = critical
        summary["judged"] = len(scores)
        return summary

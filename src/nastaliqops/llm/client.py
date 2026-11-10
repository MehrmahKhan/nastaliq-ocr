"""One thin wrapper over whichever LLM provider is configured.

Everything is OpenAI compatible, so Groq, a local Ollama server and most
hosted endpoints all work by changing two environment variables. Calls are
traced to Langfuse when the keys are present, which is where the prompt
version, latency, token count and cost per request end up.
"""

import json
import time

import yaml

from openai import OpenAI

from nastaliqops.config import Paths, env


def load_prompt(relative_path):
    path = Paths.root / relative_path
    handle = open(path, "r", encoding="utf-8")
    prompt = yaml.safe_load(handle)
    handle.close()
    return prompt


def _langfuse():
    public_key = env("LANGFUSE_PUBLIC_KEY")
    secret_key = env("LANGFUSE_SECRET_KEY")
    if public_key == "" or secret_key == "":
        return None
    try:
        from langfuse import Langfuse
        return Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=env("LANGFUSE_HOST", "http://localhost:3000"),
        )
    except Exception:
        return None


class LlmClient:
    def __init__(self):
        api_key = env("LLM_API_KEY")
        base_url = env("LLM_BASE_URL", "https://api.groq.com/openai/v1")

        if api_key == "":
            raise RuntimeError("LLM_API_KEY is not set, see .env.example")

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = env("LLM_MODEL", "llama-3.3-70b-versatile")
        self.tracer = _langfuse()

    def complete_json(self, prompt, variables, trace_name):
        """Render the prompt, call the model, parse the JSON reply."""
        system_text = prompt["system"]
        user_text = prompt["user"]
        for key in variables:
            placeholder = "{" + key + "}"
            user_text = user_text.replace(placeholder, str(variables[key]))

        defaults = prompt.get("model_defaults", {})
        started = time.time()

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=defaults.get("temperature", 0.0),
            max_tokens=defaults.get("max_tokens", 300),
            messages=[
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ],
        )

        elapsed = time.time() - started
        raw = response.choices[0].message.content.strip()

        parsed = None
        try:
            parsed = json.loads(raw)
        except Exception:
            start = raw.find("{")
            end = raw.rfind("}")
            if start >= 0 and end > start:
                try:
                    parsed = json.loads(raw[start:end + 1])
                except Exception:
                    parsed = None

        if parsed is None:
            parsed = {"text": raw, "changed": False, "reason": "unparseable reply"}

        if self.tracer is not None:
            try:
                self.tracer.trace(
                    name=trace_name,
                    input=variables,
                    output=parsed,
                    metadata={
                        "prompt_name": prompt["name"],
                        "prompt_version": prompt["version"],
                        "model": self.model,
                        "latency_seconds": elapsed,
                    },
                )
            except Exception:
                pass

        parsed["_latency"] = elapsed
        if response.usage is not None:
            parsed["_tokens"] = response.usage.total_tokens
        else:
            parsed["_tokens"] = 0

        return parsed

"""LLM generator: local (HuggingFace) and API (OpenAI-compatible) backends."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Generator:
    """Wraps a HuggingFace causal LM for grounded RAG generation.

    Loads model + tokenizer once; reused across requests.
    """

    def __init__(
        self,
        model_id: str,
        max_new_tokens: int = 600,
        do_sample: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="cpu",
        )
        self.model.eval()

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def generate(self, system: str, user: str) -> str:
        """Generate an answer given system + user prompts."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=1.0,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class APIGenerator:
    """Generator backed by an OpenAI-compatible API (Groq, OpenAI, Together, etc.).

    Same interface as `Generator` - has a `.generate(system, user) -> str` method.
    Use this when you want fast, high-quality generation without local GPU.
    """

    def __init__(
        self,
        model_id: str,
        api_key: str,
        base_url: str = "https://api.groq.com/openai/v1",
        max_new_tokens: int = 600,
        temperature: float = 0.0,
    ) -> None:
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def generate(self, system: str, user: str) -> str:
        """Call the API and return the assistant's reply."""
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        return (response.choices[0].message.content or "").strip()

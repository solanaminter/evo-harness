"""Local HuggingFace transformers backend: a real (small) model as the fixed brain.

The model is loaded ONCE per process and never mutated — the brain stays
fixed while the harness evolves around it. transformers/torch are imported
lazily so the rest of the package works without them installed.
"""

from __future__ import annotations

from typing import List, Optional

from .agent import Backend


class TransformersBackend(Backend):
    """text-generation pipeline backend (e.g. Qwen2.5-1.5B-Instruct on T4/CPU)."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
                 max_new_tokens: int = 160, device: Optional[str] = None):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        import torch  # noqa: F401  (lazy so core stays dependency-free)
        from transformers import pipeline
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.pipe = pipeline(
            "text-generation",
            model=model_name,
            torch_dtype=dtype,
            device_map="auto" if device is None else device,
            trust_remote_code=False,
        )
        # Greedy decoding: the brain is deterministic, like the dummy was.
        self.gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False,
                               return_full_text=False)

    def complete(self, prompt: str) -> str:
        out = self.pipe(prompt, **self.gen_kwargs)
        text = out[0]["generated_text"]
        return text.strip() if isinstance(text, str) else str(text).strip()

    def complete_many(self, prompts: List[str]) -> List[str]:
        """Batched generation (faster on GPU)."""
        outs = self.pipe(prompts, batch_size=len(prompts), **self.gen_kwargs)
        return [o[0]["generated_text"].strip() for o in outs]

"""Brain-directed harness mutation: the fixed small model reads failure traces
and picks which harness operator to apply. This tests H2 — is the primitive
brain smart enough to know *what* to evolve, or does random mutation +
selection do as well?
"""

from __future__ import annotations

import random
import re
from typing import Callable, List, Optional

OPERATORS = (
    "mutate_docstring", "add_tool", "remove_tool", "disable_tool",
    "enable_tool", "merge_tools", "split_tool",
    "add_validation_wrapper", "enable_retry",
)

_PROPOSER_TEMPLATE = """You are evolving an AI agent's tool harness (its tools are its limbs; you are its brain).
Current tools:
{tools}

Recent failures (task -> what the agent did):
{failures}

Which ONE change below would most likely fix these failures? Reply with EXACTLY ONE of these words and nothing else:
mutate_docstring | add_tool | remove_tool | disable_tool | enable_tool | merge_tools | split_tool | add_validation_wrapper | enable_retry

Guidance:
- mutate_docstring: rewrite a tool's description to be clearer for the agent
- disable_tool / remove_tool: a tool is useless or actively misleading
- add_tool: a needed capability is missing entirely
- add_validation_wrapper / enable_retry: failures come from bad arguments or transient errors
- merge_tools / split_tool: change tool granularity

Your one-word answer:"""


def _format_failures(failures: List[dict], max_n: int = 3) -> str:
    lines = []
    for f in failures[:max_n]:
        trace = "; ".join(
            f"CALL {t['tool']} {t['args']} -> {str(t['result'])[:60]}"
            for t in f.get("trace", [])[:2]
        ) or "no tool calls made"
        lines.append(f"- Task: {f['prompt'][:120]} | Answer: {str(f.get('answer'))[:40]} | {trace}")
    return "\n".join(lines) if lines else "- (no failures recorded)"


def make_directed_proposer(backend, max_tokens: int = 48) -> Callable:
    """Build a proposer closure over a fixed LLM backend.

    Returns proposer(rng, harness, failures) -> Optional[operator name].
    Returns None when the brain's answer is unparseable (caller falls back
    to random mutation).
    """
    # Temporarily cap generation length for the proposer call.
    orig_max = getattr(backend, "max_new_tokens", None)

    def proposer(rng: random.Random, harness, failures: List[dict]) -> Optional[str]:
        tools = "\n".join(f"- {t.name}: {t.description}"
                          for t in harness.enabled_tools())
        prompt = _PROPOSER_TEMPLATE.format(
            tools=tools, failures=_format_failures(failures))
        try:
            if orig_max is not None:
                backend.max_new_tokens = max_tokens
            out = backend.complete(prompt)
        except Exception:
            return None
        finally:
            if orig_max is not None:
                backend.max_new_tokens = orig_max
        for op in OPERATORS:
            if re.search(rf"\b{re.escape(op)}\b", out):
                return op
        return None

    return proposer

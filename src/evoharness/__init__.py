"""evoharness — evolve an LLM agent's tool harness under selection pressure.

The "brain" (the language model) is held FIXED; only the harness — the set
of tools, their docstrings, and their scaffolding — is mutated and selected.
"""

from .config import Config
from .harness import Harness, ToolDef, TOOL_REGISTRY
from .benchmark import TASKS, FAMILIES, make_tools
from .agent import Backend, RegexBackend, LLMBackend, run_agent
from .evolve import evolve, evaluate_harness

__all__ = [
    "Config",
    "Harness",
    "ToolDef",
    "TOOL_REGISTRY",
    "TASKS",
    "FAMILIES",
    "make_tools",
    "Backend",
    "RegexBackend",
    "LLMBackend",
    "run_agent",
    "evolve",
    "evaluate_harness",
]

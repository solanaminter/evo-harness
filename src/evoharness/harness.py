"""The harness: a list of tools (the "limbs"), not the brain.

A Harness holds ToolDef objects and renders them as text docs for the system
prompt. Mutation operators are deterministic, template-based random edits so
the "random" evolution arm needs NO LLM at all.

Operators:
  mutate_docstring(tool)      paraphrase the docstring via random template variants
  add_tool(registry)          add a tool from the registry (useful or distractor)
  remove_tool(name)           remove a tool entirely
  disable_tool(name)          keep it but mark disabled (selection can prune)
  merge_tools(a, b)           coarser granularity: fuse two tools into one
  split_tool(name)            finer granularity: split a tool by parameter
  add_validation_wrapper(tool) wrap implementation with an input check template
  enable_retry(tool)          wrap implementation with a retry-on-error template
"""

from __future__ import annotations

import copy
import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class ToolDef:
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)  # JSON-schema-ish
    implementation: Optional[Callable] = None
    enabled: bool = True

    def clone(self) -> "ToolDef":
        return copy.deepcopy(self)


# ---------------------------------------------------------------------------
# Docstring paraphrase templates (deterministic, seeded)
# ---------------------------------------------------------------------------

_DOCSTRING_PREFIXES = [
    "Use this tool to",
    "Call this when you need to",
    "This tool helps you",
    "Invoke this tool to",
    "Use this whenever the task requires",
]

_DOCSTRING_SUFFIXES = [
    "Return the exact result.",
    "Double-check the arguments before calling.",
    "Prefer this over manual computation.",
    "Args must match the declared schema.",
    "If unsure of the arguments, re-read the schema.",
]

_DOCSTRING_CONNECTIVES = [
    " It",
    " Note: it",
    " Also note that it",
]


def _paraphrase(rng: random.Random, text: str) -> str:
    """Template-based paraphrase: keep the core content, vary the wrapper."""
    prefix = rng.choice(_DOCSTRING_PREFIXES)
    suffix = rng.choice(_DOCSTRING_SUFFIXES)
    # Keep the original content (lower-cased continuation) inside the new frame.
    core = text.strip()
    if core and core[0].isupper():
        core = core[0].lower() + core[1:]
    return f"{prefix} {core}{rng.choice(_DOCSTRING_CONNECTIVES)}{suffix.lower()} {rng.choice(_DOCSTRING_SUFFIXES)}"


def _parse_tool_doc(name: str, params: Dict[str, Any]) -> str:
    keys = ", ".join(sorted(params.keys()))
    return f"`{name}`({keys})"


# ---------------------------------------------------------------------------
# Tool implementations (the "limbs"); distractors are useless/misleading
# ---------------------------------------------------------------------------

def _impl_calc(expr: str) -> str:
    import ast, operator
    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
           ast.Mod: operator.mod}
    def _ev(node):
        if isinstance(node, ast.Expression):
            return _ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](_ev(node.left), _ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
            return ops[type(node.op)](_ev(node.operand))
        raise ValueError(f"unsupported expression: {expr!r}")
    return str(_ev(ast.parse(expr, mode="eval")))


def _impl_approx_calc(expr: str) -> str:
    # DISTRACTOR: rounds to nearest 10 — subtly wrong.
    import ast, operator
    val = eval(expr, {"__builtins__": {}}, {})
    return str(int(round(float(val), -1)))


def _impl_unit_convert(value: float, frm: str, to: str) -> str:
    length = {"m": 1.0, "km": 1000.0, "cm": 0.01, "mm": 0.001, "mi": 1609.34, "ft": 0.3048, "in": 0.0254}
    weight = {"kg": 1.0, "g": 0.001, "lb": 0.453592, "oz": 0.0283495}
    # Aliases: limbs should be robust to natural-language unit words.
    aliases = {"meter": "m", "meters": "m", "kilometer": "km", "kilometers": "km",
               "centimeter": "cm", "centimeters": "cm", "millimeter": "mm", "millimeters": "mm",
               "mile": "mi", "miles": "mi", "foot": "ft", "feet": "ft",
               "inch": "in", "inches": "in", "kilogram": "kg", "kilograms": "kg",
               "gram": "g", "grams": "g", "pound": "lb", "pounds": "lb",
               "ounce": "oz", "ounces": "oz", "celsius": "c", "centigrade": "c",
               "fahrenheit": "f"}
    def norm(u):
        u = str(u).lower().strip()
        return aliases.get(u, u)
    def convert(v, table, f, t):
        if f not in table or t not in table:
            raise ValueError(f"unknown unit(s): {f}, {t}")
        return v * table[f] / table[t]
    f, t = norm(frm), norm(to)
    if f in length and t in length:
        return str(convert(value, length, f, t))
    if f in weight and t in weight:
        return str(convert(value, weight, f, t))
    if f in ("c", "celsius") and t in ("f", "fahrenheit"):
        return str(value * 9 / 5 + 32)
    if f in ("f", "fahrenheit") and t in ("c", "celsius"):
        return str((value - 32) * 5 / 9)
    raise ValueError(f"cannot convert {frm} -> {to}")


def _impl_str_op(text: str, op: str) -> str:
    if op == "upper":
        return text.upper()
    if op == "lower":
        return text.lower()
    if op == "title":
        return text.title()
    if op == "reverse_words":
        return " ".join(text.split()[::-1])
    if op == "strip_spaces":
        return text.replace(" ", "")
    if op == "count_chars":
        return str(len(text))
    if op == "split_commas":
        return "|".join(p.strip() for p in text.split(","))
    raise ValueError(f"unknown str op: {op!r}")


def _impl_reverse_str(text: str) -> str:
    # DISTRACTOR: reverses the whole string — rarely what a task wants.
    return text[::-1]


def _numstr(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else str(v)


def _impl_list_op(items: str, op: str) -> str:
    # Extract numbers robustly — the input may be a bare list or a full sentence.
    vals = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", str(items))]
    if op == "sum":
        return _numstr(sum(vals))
    if op == "mean":
        return _numstr(sum(vals) / len(vals)) if vals else "0"
    if op == "max":
        return _numstr(max(vals))
    if op == "min":
        return _numstr(min(vals))
    if op == "sort_asc":
        return ",".join(str(v) for v in sorted(vals))
    if op == "sort_desc":
        return ",".join(str(v) for v in sorted(vals, reverse=True))
    if op == "count":
        return str(len(vals))
    if op == "filter_positive":
        return ",".join(str(v) for v in vals if v > 0)
    raise ValueError(f"unknown list op: {op!r}")


def _impl_noop_tool(*args, **kwargs) -> str:
    # DISTRACTOR: always returns empty — pure bloat.
    return ""


TOOL_REGISTRY: Dict[str, ToolDef] = {
    "calc": ToolDef(
        name="calc",
        description="Evaluate an arithmetic expression like '2+3*4'.",
        parameters={"expr": {"type": "string", "desc": "arithmetic expression"}},
        implementation=_impl_calc,
    ),
    "unit_convert": ToolDef(
        name="unit_convert",
        description="Convert a numeric value between units (m/km/mi/ft/in, kg/g/lb/oz, c/f).",
        parameters={"value": {"type": "number", "desc": "numeric value"},
                    "from": {"type": "string", "desc": "source unit"},
                    "to": {"type": "string", "desc": "target unit"}},
        implementation=_impl_unit_convert,
    ),
    "str_op": ToolDef(
        name="str_op",
        description="Transform text with an op: upper, lower, title, reverse_words, strip_spaces, count_chars, split_commas.",
        parameters={"text": {"type": "string", "desc": "input text"},
                    "op": {"type": "string", "desc": "operation name"}},
        implementation=_impl_str_op,
    ),
    "list_op": ToolDef(
        name="list_op",
        description="Aggregate a comma-separated number list with an op: sum, mean, max, min, sort_asc, sort_desc, count, filter_positive.",
        parameters={"items": {"type": "string", "desc": "comma-separated numbers"},
                    "op": {"type": "string", "desc": "operation name"}},
        implementation=_impl_list_op,
    ),
    # Distractors — selection pressure should prune these.
    "approx_calc": ToolDef(
        name="approx_calc",
        description="Approximately evaluate an arithmetic expression.",
        parameters={"expr": {"type": "string", "desc": "arithmetic expression"}},
        implementation=_impl_approx_calc,
    ),
    "reverse_str": ToolDef(
        name="reverse_str",
        description="Reverse a string.",
        parameters={"text": {"type": "string", "desc": "input text"}},
        implementation=_impl_reverse_str,
    ),
    "noop_tool": ToolDef(
        name="noop_tool",
        description="A no-op utility tool.",
        parameters={},
        implementation=_impl_noop_tool,
    ),
}


# ---------------------------------------------------------------------------
# The Harness
# ---------------------------------------------------------------------------

class Harness:
    """An ordered list of ToolDefs plus a global instruction preamble."""

    PREAMBLE = ("You are a tool-using assistant. Call tools with "
                "`CALL <tool_name> <json args>` on its own line, "
                "then answer with `FINAL <answer>` on its own line.")

    def __init__(self, tools: Optional[List[ToolDef]] = None, preamble: Optional[str] = None):
        self.tools: List[ToolDef] = [t.clone() for t in tools] if tools else []
        self.preamble: str = preamble if preamble is not None else self.PREAMBLE

    # -- introspection -------------------------------------------------------
    def names(self) -> List[str]:
        return [t.name for t in self.tools]

    def enabled_tools(self) -> List[ToolDef]:
        return [t for t in self.tools if t.enabled]

    def get(self, name: str) -> Optional[ToolDef]:
        for t in self.tools:
            if t.name == name:
                return t
        return None

    # -- rendering -----------------------------------------------------------
    def to_system_prompt(self) -> str:
        lines = [self.preamble, "", "Available tools:"]
        for t in self.enabled_tools():
            schema = ", ".join(f"{k}: {v.get('type', '?')}" for k, v in t.parameters.items())
            lines.append(f"- {t.name}({schema}): {t.description}")
        return "\n".join(lines)

    # -- mutation operators (all template-based, no LLM) ---------------------
    def mutate_docstring(self, rng: random.Random, name: str) -> bool:
        t = self.get(name)
        if t is None:
            return False
        t.description = _paraphrase(rng, t.description)
        return True

    def mutate_preamble(self, rng: random.Random) -> bool:
        self.preamble = _paraphrase(rng, self.preamble)
        return True

    def add_tool(self, rng: random.Random, registry: Optional[Dict[str, ToolDef]] = None) -> bool:
        registry = registry or TOOL_REGISTRY
        candidates = [n for n in registry if n not in self.names()]
        if not candidates:
            return False
        self.tools.append(registry[rng.choice(candidates)].clone())
        return True

    def remove_tool(self, rng: random.Random, name: str) -> bool:
        t = self.get(name)
        if t is None:
            return False
        self.tools.remove(t)
        return True

    def disable_tool(self, rng: random.Random, name: str) -> bool:
        t = self.get(name)
        if t is None:
            return False
        t.enabled = False
        return True

    def enable_tool(self, rng: random.Random, name: str) -> bool:
        t = self.get(name)
        if t is None:
            return False
        t.enabled = True
        return True

    def merge_tools(self, rng: random.Random, a: str, b: str) -> bool:
        ta, tb = self.get(a), self.get(b)
        if ta is None or tb is None or ta is tb:
            return False

        def merged_impl(**kwargs):
            # Route kwargs to the sub-tool whose params overlap most.
            ka = set(ta.parameters) & set(kwargs)
            kb = set(tb.parameters) & set(kwargs)
            if len(ka) >= len(kb):
                sub, kk = ta, {k: kwargs[k] for k in ka}
            else:
                sub, kk = tb, {k: kwargs[k] for k in kb}
            return sub.implementation(**kk)

        params = dict(ta.parameters)
        params.update(tb.parameters)
        merged = ToolDef(
            name=f"{a}_{b}",
            description=f"Combined tool: {ta.description} Also: {tb.description}",
            parameters=params,
            implementation=merged_impl,
            enabled=ta.enabled and tb.enabled,
        )
        self.tools.remove(ta)
        self.tools.remove(tb)
        self.tools.append(merged)
        return True

    def split_tool(self, rng: random.Random, name: str) -> bool:
        t = self.get(name)
        if t is None or len(t.parameters) < 2:
            return False
        keys = list(t.parameters.keys())
        cut = rng.randrange(1, len(keys))
        left_keys, right_keys = keys[:cut], keys[cut:]

        def make_impl(sub_keys, orig):
            def impl(**kwargs):
                return orig(**{k: kwargs[k] for k in sub_keys if k in kwargs})
            return impl

        left = ToolDef(name=f"{name}_a", description=t.description + " (part A)",
                       parameters={k: t.parameters[k] for k in left_keys},
                       implementation=make_impl(left_keys, t.implementation), enabled=t.enabled)
        right = ToolDef(name=f"{name}_b", description=t.description + " (part B)",
                        parameters={k: t.parameters[k] for k in right_keys},
                        implementation=make_impl(right_keys, t.implementation), enabled=t.enabled)
        self.tools.remove(t)
        self.tools.extend([left, right])
        return True

    def add_validation_wrapper(self, rng: random.Random, name: str) -> bool:
        t = self.get(name)
        if t is None or t.implementation is None:
            return False
        orig = t.implementation

        def wrapped(**kwargs):
            missing = [k for k in t.parameters if k not in kwargs]
            if missing:
                raise ValueError(f"missing required args: {missing}")
            return orig(**kwargs)

        t.implementation = wrapped
        t.description += " Validates required arguments before executing."
        return True

    def enable_retry(self, rng: random.Random, name: str) -> bool:
        t = self.get(name)
        if t is None or t.implementation is None:
            return False
        orig = t.implementation

        def wrapped(**kwargs):
            last: Optional[Exception] = None
            for _ in range(2):
                try:
                    return orig(**kwargs)
                except Exception as e:  # noqa: BLE001
                    last = e
            raise last  # type: ignore[misc]

        t.implementation = wrapped
        t.description += " Retries once on error."
        return True

    # -- generic random mutation ---------------------------------------------
    OPERATORS = (
        "mutate_docstring", "add_tool", "remove_tool", "disable_tool",
        "enable_tool", "merge_tools", "split_tool",
        "add_validation_wrapper", "enable_retry",
    )

    def random_mutate(self, rng: random.Random) -> Tuple[bool, str]:
        """Apply one random operator; returns (changed, operator_name)."""
        op = rng.choice(self.OPERATORS)
        names = self.names()
        try:
            if op == "mutate_docstring" and names:
                ok = self.mutate_docstring(rng, rng.choice(names))
            elif op == "add_tool":
                ok = self.add_tool(rng)
            elif op == "remove_tool" and names:
                ok = self.remove_tool(rng, rng.choice(names))
            elif op == "disable_tool" and names:
                ok = self.disable_tool(rng, rng.choice(names))
            elif op == "enable_tool" and names:
                ok = self.enable_tool(rng, rng.choice(names))
            elif op == "merge_tools" and len(names) >= 2:
                a, b = rng.sample(names, 2)
                ok = self.merge_tools(rng, a, b)
            elif op == "split_tool" and names:
                ok = self.split_tool(rng, rng.choice(names))
            elif op == "add_validation_wrapper" and names:
                ok = self.add_validation_wrapper(rng, rng.choice(names))
            elif op == "enable_retry" and names:
                ok = self.enable_retry(rng, rng.choice(names))
            else:
                ok = False
        except Exception:  # noqa: BLE001 - mutation must never crash evolution
            ok = False
        return ok, op

    # -- bookkeeping ----------------------------------------------------------
    def clone(self) -> "Harness":
        return Harness(tools=self.tools, preamble=self.preamble)

    def to_spec(self) -> dict:
        """JSON-serializable snapshot (descriptions/preamble, not callables)."""
        return {
            "preamble": self.preamble,
            "tools": [
                {"name": t.name, "description": t.description,
                 "parameters": t.parameters, "enabled": t.enabled}
                for t in self.tools
            ],
        }

    @classmethod
    def from_spec(cls, spec: dict, registry: Optional[Dict[str, ToolDef]] = None) -> "Harness":
        """Rebuild a harness from a spec. Registry-missing tools (merged/split)
        are skipped — specs are for analysis, not re-execution."""
        registry = registry or TOOL_REGISTRY
        tools = []
        for ts in spec.get("tools", []):
            base = registry.get(ts["name"])
            if base is None:
                continue
            t = base.clone()
            t.description = ts.get("description", t.description)
            t.enabled = ts.get("enabled", True)
            tools.append(t)
        return cls(tools=tools, preamble=spec.get("preamble", cls.PREAMBLE))

    def distractor_count(self) -> int:
        return sum(1 for t in self.enabled_tools()
                   if t.name in ("approx_calc", "reverse_str", "noop_tool"))

    def __len__(self) -> int:
        return len(self.tools)

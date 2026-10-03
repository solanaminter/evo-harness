"""Tool-calling agent loop with a pluggable LLM backend (the "brain").

Protocol:
  - The model emits lines like:  CALL calc {"expr": "2+2"}
  - The model emits its answer like:  FINAL <answer>
  - Up to max_turns tool-call rounds; parser is regex-based and robust.

Backends:
  - RegexBackend: rule-based dummy, no model — for smoke tests and baselines.
  - LLMBackend: HTTP JSON POST to a transformers-server / HF inference-style
    endpoint. Network only used when explicitly selected.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Tuple

from .harness import Harness, ToolDef

CALL_RE = re.compile(r"^CALL\s+(\w+)\s+(\{.*\})\s*$", re.IGNORECASE)
FINAL_RE = re.compile(r"^FINAL\s+(.*)$", re.IGNORECASE)


class Backend:
    """Abstract brain: text in, text out."""

    def complete(self, prompt: str) -> str:
        raise NotImplementedError


class RegexBackend(Backend):
    """Deterministic rule-based dummy brain for smoke tests.

    Understands the tool docs in the prompt and emits CALL/FINAL lines for
    the benchmark's task families. Intentionally imperfect: it falls back to
    the first available tool whose name it does not recognize correctly, which
    is what lets the harness (tool docs) matter — better docstrings / fewer
    distractors change its behavior.
    """

    def _tools_from_prompt(self, prompt: str) -> List[str]:
        names = []
        for line in prompt.splitlines():
            m = re.match(r"-\s+(\w+)\(", line.strip())
            if m:
                names.append(m.group(1))
        return names

    def complete(self, prompt: str) -> str:
        tools = self._tools_from_prompt(prompt)
        # If a tool result already arrived, answer with it (last one wins).
        results = [l for l in prompt.splitlines() if l.startswith("Result of ")]
        if results:
            return f"FINAL {results[-1].split(': ', 1)[1].strip()}"
        # Find the most recent user task line.
        task = ""
        for line in reversed(prompt.splitlines()):
            if line.startswith("Task:"):
                task = line[len("Task:"):].strip()
                break
        return self._solve(task, tools)

    def _pick(self, tools: List[str], preferred: List[str]) -> Optional[str]:
        for p in preferred:
            if p in tools:
                return p
        # Misled fallback: pick first enabled tool (distractors can win here).
        return tools[0] if tools else None

    def _solve(self, task: str, tools: List[str]) -> str:
        tl = task.lower()

        # --- string family ---
        if "uppercase" in tl or ("upper" in tl and "case" in tl):
            t = self._pick(tools, ["str_op", "reverse_str", "noop_tool"])
            txt = self._quoted(task)
            return self._emit(t, {"text": txt, "op": "upper"})
        if "lowercase" in tl:
            t = self._pick(tools, ["str_op", "reverse_str"])
            return self._emit(t, {"text": self._quoted(task), "op": "lower"})
        if "title-case" in tl or "title case" in tl:
            t = self._pick(tools, ["str_op", "reverse_str"])
            return self._emit(t, {"text": self._quoted(task), "op": "title"})
        if "reverse the word order" in tl:
            t = self._pick(tools, ["str_op", "reverse_str"])
            return self._emit(t, {"text": self._quoted(task), "op": "reverse_words"})
        if "remove all spaces" in tl:
            t = self._pick(tools, ["str_op", "reverse_str"])
            return self._emit(t, {"text": self._quoted(task), "op": "strip_spaces"})
        if "count the characters" in tl:
            t = self._pick(tools, ["str_op", "reverse_str"])
            return self._emit(t, {"text": self._quoted(task), "op": "count_chars"})
        if "split on commas" in tl:
            t = self._pick(tools, ["str_op", "reverse_str"])
            return self._emit(t, {"text": self._quoted(task), "op": "split_commas"})

        # --- unit family ---
        m = re.search(r"convert\s+([\d.]+)\s+(\w+)\s+to\s+(\w+)", tl)
        if m:
            t = self._pick(tools, ["unit_convert", "calc", "approx_calc"])
            return self._emit(t, {"value": float(m.group(1)), "from": m.group(2), "to": m.group(3)})

        # --- list family ---
        if any(k in tl for k in ["sum these", "mean of", "max of", "min of",
                                 "sort ascending", "sort descending",
                                 "count the numbers", "keep only positives"]):
            op_map = {"sum these": "sum", "mean of": "mean", "max of": "max",
                      "min of": "min", "sort ascending": "sort_asc",
                      "sort descending": "sort_desc", "count the numbers": "count",
                      "keep only positives": "filter_positive"}
            op = next(v for k, v in op_map.items() if k in tl)
            t = self._pick(tools, ["list_op", "str_op", "noop_tool"])
            nums = self._quoted(task)
            return self._emit(t, {"items": nums, "op": op})

        # --- arithmetic family ---
        t = self._pick(tools, ["calc", "approx_calc", "noop_tool"])
        expr = self._arith_expr(task)
        return self._emit(t, {"expr": expr})

    @staticmethod
    def _quoted(task: str) -> str:
        m = re.search(r"'([^']*)'", task)
        return m.group(1) if m else task

    @staticmethod
    def _arith_expr(task: str) -> str:
        # Grab the expression starting at the first digit (with an optional
        # leading open paren); stop at the first non-expression character.
        m = re.search(r"\(?\d[\d+\-*/().^% ]*", task)
        if m:
            expr = m.group(0).strip().rstrip(".").strip()
            return expr.replace("^", "**")
        return task

    @staticmethod
    def _emit(tool: Optional[str], args: dict) -> str:
        if tool is None:
            return "FINAL unknown"
        return f"CALL {tool} {json.dumps(args)}\nFINAL <pending>"


class LLMBackend(Backend):
    """HTTP backend for a real model server (network only when selected).

    Expects POST {endpoint} with {"prompt": ..., "max_tokens": ...} returning
    {"text": "..."}. Adapt to your server's schema as needed.
    """

    def __init__(self, endpoint: str, max_tokens: int = 512, timeout: int = 120):
        self.endpoint = endpoint
        self.max_tokens = max_tokens
        self.timeout = timeout

    def complete(self, prompt: str) -> str:
        import urllib.request
        payload = json.dumps({"prompt": prompt, "max_tokens": self.max_tokens}).encode()
        req = urllib.request.Request(self.endpoint, data=payload,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode())
        return data.get("text", "")


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def parse_output(text: str) -> Tuple[List[Tuple[str, dict]], Optional[str]]:
    """Extract (tool, args) calls and the FINAL answer from model text."""
    calls: List[Tuple[str, dict]] = []
    final: Optional[str] = None
    for line in text.splitlines():
        line = line.strip()
        m = CALL_RE.match(line)
        if m:
            try:
                calls.append((m.group(1), json.loads(m.group(2))))
            except json.JSONDecodeError:
                continue
            continue
        m = FINAL_RE.match(line)
        if m and m.group(1).strip() not in ("<pending>", ""):
            final = m.group(1).strip()
    return calls, final


def execute_call(harness: Harness, tool_name: str, args: dict) -> str:
    tool = harness.get(tool_name)
    if tool is None or not tool.enabled:
        return f"ERROR: unknown or disabled tool '{tool_name}'"
    if tool.implementation is None:
        return f"ERROR: tool '{tool_name}' has no implementation"
    # Protocol-level shim: JSON key "from" is a Python keyword; tools declare "frm".
    args = {("frm" if k == "from" else k): v for k, v in args.items()}
    try:
        return str(tool.implementation(**args))
    except Exception as e:  # noqa: BLE001 - agent must see the error, not crash
        return f"ERROR: {type(e).__name__}: {e}"


PROTOCOL_EXAMPLE = """
Example of a correct response (follow this exact format):
Task: What is 12*8?
CALL calc {"expr": "12*8"}
Result of calc: 96
FINAL 96

Now solve the task below. Output CALL lines and then FINAL <answer>."""


def build_system_prompt(harness: Harness, protocol_example: bool = False) -> str:
    system = harness.to_system_prompt()
    if protocol_example:
        system += "\n" + PROTOCOL_EXAMPLE
    return system


def run_agent(harness: Harness, backend: Backend, task_prompt: str,
              max_turns: int = 6, protocol_example: bool = False) -> Dict:
    """Run the tool loop. Returns {answer, success-candidate, trace, turns}."""
    system = build_system_prompt(harness, protocol_example)
    transcript = f"{system}\n\nTask: {task_prompt}\n"
    trace: List[Dict] = []
    answer: Optional[str] = None

    for turn in range(max_turns):
        out = backend.complete(transcript)
        calls, final = parse_output(out)
        for tool_name, args in calls:
            result = execute_call(harness, tool_name, args)
            trace.append({"turn": turn, "tool": tool_name, "args": args, "result": result})
            transcript += f"Result of {tool_name}: {result}\n"
        if final is not None:
            answer = final
            break
        if not calls:
            # No calls and no FINAL — nudge once, then give up.
            transcript += "Please call a tool or answer with FINAL <answer>.\n"
    return {"answer": answer, "trace": trace, "turns": len(trace)}


def solve_task(harness: Harness, backend: Backend, task, max_turns: int = 6) -> bool:
    res = run_agent(harness, backend, task.prompt, max_turns=max_turns)
    return res["answer"] is not None and task.check(res["answer"])

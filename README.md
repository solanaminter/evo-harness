# evo-harness

**Research question: did the harness come first, or the brain?**

In biology, the sensorimotor periphery came first: primitive creatures had
sensors and limbs long before anything like a brain existed, and nervous
systems evolved *around* a body that could already sense and act. Embodied
cognition argues the same point from the other side — much of what looks like
intelligence is actually the body doing the work.

This repo asks the AI analogue: **with the language model (the "brain") held
completely fixed, can evolutionary pressure on the agent's tooling and
scaffolding (the "harness" — tools, their docstrings, validation, retries,
granularity) produce a smarter agent?** And which harness mutations matter
most?

## Pre-registered hypotheses

- **H1 — harness beats prompt.** With the brain fixed, evolving the harness
  (tools + docs + scaffolding) beats evolving the global instruction preamble
  alone. Test: `random` arm vs `prompt` arm vs `baseline` in `exp1_core`.
- **H2 — directed beats random.** Brain-directed harness mutation (an LLM
  proposes edits from failure traces) beats random template mutation +
  selection. Test: `directed` arm vs `random` arm (directed proposer is
  currently a stub that falls back to random — see `src/evoharness/evolve.py`).
- **H3 — docs beat structure.** Tool description/docstring mutations contribute
  more to fitness gains than structural add/remove operations. Test: ablate
  operator classes and compare per-generation fitness deltas.
- **H4 — selection prunes.** Selection pressure removes unused/misleading
  tools. Test: seed populations with distractor tools (`approx_calc`,
  `reverse_str`, `noop_tool`) and measure their survival rate across
  generations.

## Repo layout

```
evo-harness/
├── README.md
├── requirements.txt            # pyyaml, numpy, matplotlib (transformers optional)
├── SMOKE_TEST.md              # smoke-test record
├── src/evoharness/
│   ├── __init__.py
│   ├── config.py               # Config dataclass, YAML load/save
│   ├── harness.py              # Harness + ToolDef + 9 template mutation operators
│   ├── benchmark.py            # 40 synthetic tool-use tasks, 4 families × 10
│   ├── agent.py                # tool-calling loop, RegexBackend, LLMBackend (HTTP)
│   └── evolve.py               # tournament selection, per-arm mutation, CSV logging
└── experiments/
    ├── exp1_core/              # H1: baseline vs prompt vs random (regex dummy brain)
    │   ├── config.yaml         # 10 gens, pop 8, train 24 / test 16, seeds 1..3
    │   └── run.py
    └── exp2_llm/               # H1..H4 with a REAL fixed brain (Qwen2.5-1.5B, Modal T4)
        ├── config_llm.yaml     # 6 gens, pop 6, train 12 / test 28, seeds 1..2,
        │                       # arms: baseline/prompt/random/directed
        ├── modal_app.py        # pilot() protocol check + run_exp2() full run
        ├── directed.py         # brain-directed proposer (LLM picks operator)
        └── analyze.py          # figures + RESULTS.md from downloaded runs
```

## How to run

Prerequisites: Python 3.10+, `pip install -r requirements.txt`. No network
calls in the default path — the `regex` backend is a deterministic rule-based
dummy brain, so the core loop runs on stdlib + numpy/matplotlib/pyyaml only.

```bash
cd ~/workspace/evo-harness

# Core experiment (exp1_core): ~1 min with the dummy backend
PYTHONPATH=src python3 experiments/exp1_core/run.py

# Results land in experiments/exp1_core/results/seed{1,2,3}/{baseline,prompt,random}/fitness_log.csv
# columns: generation, arm, seed, best, mean, best_tools, best_preamble_len
```

To use a real model as the fixed brain, point `llm_endpoint` at a local
inference server and set `backend: llm-http` in the config (see
`LLMBackend` in `src/evoharness/agent.py` for the expected JSON schema).
The model output protocol is:

```
CALL <tool_name> {"arg": "value"}
FINAL <answer>
```

## Design notes

- **Deterministic where possible.** All RNG is seeded (`random.Random(seed)`);
  benchmark gold answers are computed at import time and self-checked.
- **Random mutation needs no LLM.** All nine harness operators are
  template-based, so the `random` arm runs with zero model calls.
- **Directed mutation is wired in exp2.** `evolve.py` defines the proposer
  interface `proposer(rng, harness, failures) -> operator-name`; exp2 injects
  a real one (`experiments/exp2_llm/directed.py`) where the fixed small model
  reads failure traces and picks the operator. Without an injected proposer
  it falls back to random mutation.
- **Caveat on the dummy backend.** `RegexBackend` reads tool *names* from the
  system prompt but largely ignores docstrings/preamble, so the `prompt` arm
  will show ~zero effect with it. H1/H3 need a real LLM backend for a fair
  test; the dummy is for pipeline validation and baselines.

## Status

- [x] Repo skeleton, package, benchmark (40 tasks), agent loop, evolution loop
- [x] Smoke test passes (see `SMOKE_TEST.md`)
- [x] `exp1_core` full results + matplotlib figures (regex dummy brain — pipeline validation)
- [x] Real LLM backend wired (`src/evoharness/local_backend.py`, H2 directed proposer)
- [x] Literature review: `docs/LITERATURE.md` (~3,860 words, 39 verified citations)
- [~] `exp2_llm` running on Modal T4 (Qwen2.5-1.5B-Instruct fixed brain); pilot 4/8
- [ ] `exp2_llm` figures + RESULTS.md
- [ ] **Not pushed to GitHub — no token on hand (pending Solana's action).**
  Repo stays local until he supplies a GitHub token for publishing.

## exp2_llm — the real-brain experiment (in progress)

The fixed "primitive brain" is **Qwen2.5-1.5B-Instruct** (greedy decoding),
run on a single Modal T4 via `experiments/exp2_llm/modal_app.py`. Cost-sane by
design: one container, model cached on a Modal volume, ~5k short generations
total, billed against Modal's free monthly credits (no paid API keys).

- **Pilot** (`pilot()`): 4/8 tasks solved with distractors present. Failure
  modes are harness-fixable: the brain sometimes skips tool calls, ignores
  tool results, or mis-formats FINAL — exactly what docstring/preamble
  evolution should fix.
- **Full run** (`run_exp2()`): 2 seeds × 4 arms (baseline / prompt-only /
  random harness / brain-directed harness), 6 generations, pop 6, train 12,
  held-out test 28. Per-run artifacts: `fitness_log.csv`, `ops_log.csv`
  (operator usage for H3), `summary.json` (train + test fitness),
  `best_harness.json` (before/after docstring analysis).
- **Fetch results**: `modal volume get evo-harness-runs exp2_llm
  ./experiments/exp2_llm/results_remote`, then
  `python3 experiments/exp2_llm/analyze.py` for figures.

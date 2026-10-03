#!/usr/bin/env python3
"""exp2_llm on Modal: fixed Qwen2.5-1.5B-Instruct brain on T4, harness evolves.

Usage (from repo root):
    ~/workspace/.venvs/hack/bin/modal run experiments/exp2_llm/modal_app.py::pilot
    ~/workspace/.venvs/hack/bin/modal run experiments/exp2_llm/modal_app.py::run_exp2

Results persist on the `evo-harness-runs` volume under exp2_llm/; fetch with:
    modal volume get evo-harness-runs exp2_llm ./experiments/exp2_llm/results_remote

Cost control: single T4, one container, model cached on volume after first
download. No paid API keys; billed against Modal's free monthly credits.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import modal

SRC = Path.home() / "workspace" / "evo-harness" / "src"
EXP_DIR = Path.home() / "workspace" / "evo-harness" / "experiments" / "exp2_llm"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "accelerate", "pyyaml", "numpy")
    .add_local_dir(str(SRC), remote_path="/root/evosrc")
    .add_local_dir(str(EXP_DIR), remote_path="/root/exp")
)

app = modal.App("evo-harness-exp2", image=image)
model_vol = modal.Volume.from_name("evo-harness-models", create_if_missing=True)
runs_vol = modal.Volume.from_name("evo-harness-runs", create_if_missing=True)

REMOTE_SRC = "/root/evosrc"
REMOTE_EXP = "/root/exp"
REMOTE_RESULTS = "/runs/exp2_llm"


def _bootstrap():
    """Inside-container imports: path + HF cache on the model volume."""
    sys.path.insert(0, REMOTE_SRC)
    os.environ["HF_HOME"] = "/models/hf"
    os.environ["HF_HUB_CACHE"] = "/models/hf/hub"


@app.function(
    gpu="T4",
    volumes={"/models": model_vol, "/runs": runs_vol},
    timeout=1200,
)
def pilot(model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", n_tasks: int = 8):
    """Protocol check: can the tiny brain follow CALL/FINAL? Prints traces."""
    _bootstrap()
    import random
    from evoharness.agent import run_agent
    from evoharness.benchmark import train_test_split
    from evoharness.evolve import seed_harness
    from evoharness.local_backend import TransformersBackend

    backend = TransformersBackend(model_name, max_new_tokens=160)
    rng = random.Random(1)
    harness = seed_harness(rng, useful_only=False)  # distractors included
    train, _ = train_test_split(seed=1, train_size=12)
    out = []
    for task in train[:n_tasks]:
        res = run_agent(harness, backend, task.prompt, max_turns=3,
                        protocol_example=True)
        ok = res["answer"] is not None and task.check(res["answer"])
        print(f"\n--- {task.id} [{task.family}] ok={ok}\n"
              f"TASK: {task.prompt}\nANSWER: {res['answer']}", flush=True)
        for t in res["trace"][:4]:
            print(f"  CALL {t['tool']} {t['args']} -> {str(t['result'])[:80]}",
                  flush=True)
        out.append({"id": task.id, "ok": ok, "answer": res["answer"],
                    "n_calls": len(res["trace"])})
    acc = sum(o["ok"] for o in out) / len(out)
    print(f"\nPILOT accuracy: {acc:.3f} ({sum(o['ok'] for o in out)}/{len(out)})",
          flush=True)
    return {"accuracy": acc, "tasks": out}


@app.function(
    gpu="T4",
    volumes={"/models": model_vol, "/runs": runs_vol},
    timeout=21600,
)
def run_exp2(model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
    """Full exp2: all (seed, arm) evolutions + test evals, on one container."""
    _bootstrap()
    from evoharness.config import Config
    from evoharness.evolve import run_all

    cfg = Config.from_yaml(Path(REMOTE_EXP) / "config_llm.yaml")
    cfg.model_name = model_name
    cfg.results_dir = Path(REMOTE_RESULTS) / "results"

    # The directed arm's brain: same fixed small model, reading failure traces.
    sys.path.insert(0, REMOTE_EXP)
    from directed import make_directed_proposer  # noqa: E402
    from evoharness.local_backend import TransformersBackend  # noqa: E402

    backend = TransformersBackend(model_name, max_new_tokens=cfg.max_new_tokens)
    proposers = {"directed": make_directed_proposer(backend)}

    # evolve() builds its own backend from cfg; share one model instance by
    # monkey-patching _make_backend for this run (model load is the slow part).
    import evoharness.evolve as ev
    ev._make_backend = lambda cfg: backend  # noqa: SLF001

    logs = run_all(cfg, proposers=proposers)
    print(f"\nDONE: {len(logs)} runs. Results on volume at {REMOTE_RESULTS}",
          flush=True)
    return [str(p) for p in logs]


@app.function(
    gpu="T4",
    volumes={"/models": model_vol, "/runs": runs_vol},
    timeout=21600,
)
def run_exp2_resume(model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
    """Resume exp2 after the 2026-10-02 run died on client disconnect.

    Completed before the kill: seed1/{baseline,prompt,random,directed} and
    seed2/baseline. This runs exactly the three missing combos —
    (2, prompt), (2, random), (2, directed) — deterministically (same seeds,
    same RNG), writing to the same volume paths. Run with --detach.
    """
    _bootstrap()
    from evoharness.config import Config
    from evoharness.evolve import evolve

    cfg = Config.from_yaml(Path(REMOTE_EXP) / "config_llm.yaml")
    cfg.model_name = model_name
    cfg.results_dir = Path(REMOTE_RESULTS) / "results"

    sys.path.insert(0, REMOTE_EXP)
    from directed import make_directed_proposer  # noqa: E402
    from evoharness.local_backend import TransformersBackend  # noqa: E402

    backend = TransformersBackend(model_name, max_new_tokens=cfg.max_new_tokens)
    proposers = {"directed": make_directed_proposer(backend)}

    import evoharness.evolve as ev
    ev._make_backend = lambda cfg: backend  # noqa: SLF001

    missing = [(2, "prompt"), (2, "random"), (2, "directed")]
    logs = []
    for seed, arm in missing:
        print(f"\n=== RESUME seed {seed} arm {arm} ===", flush=True)
        logs.append(str(evolve(cfg, seed, arm, proposers)))
    print(f"\nRESUME DONE: {len(logs)} runs. Results on volume at {REMOTE_RESULTS}",
          flush=True)
    return logs

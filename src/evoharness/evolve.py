"""Evolution loop: the brain is FIXED; only the harness evolves.

Per generation:
  1. Evaluate each harness on the TRAIN split (fitness = success rate).
  2. Tournament selection.
  3. Per-arm mutation strategy produces offspring:
     - "baseline": clone, no mutation (control: does selection alone help?)
     - "prompt":   mutate only the global instruction preamble
     - "random":   random harness operator (template-based, no LLM needed)
     - "directed": brain-directed harness mutation — an LLM proposes an edit
                    from failure traces. Falls back to random unless a
                    `directed_proposer` callable is injected via `proposers`.
  4. Log best/mean fitness per generation to CSV, applied operators to
     ops_log.csv, and a final summary.json (train + held-out test fitness).

Directed proposer interface:
    proposer(rng, harness: Harness, failures: List[dict]) -> Optional[str]
    where failures = [{"task_id":..., "prompt":..., "trace": [...]}]
    and the return value is the name of a mutation operator to apply
    (or None to fall back to random).
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .agent import Backend, run_agent, solve_task
from .benchmark import Task, train_test_split
from .config import Config
from .harness import Harness, TOOL_REGISTRY

DirectedProposer = Callable[[random.Random, Harness, List[dict]], Optional[str]]


def evaluate_harness(harness: Harness, backend: Backend, tasks: List[Task],
                     max_turns: int = 6,
                     protocol_example: bool = False) -> Tuple[float, List[dict]]:
    """Fitness = fraction of tasks solved. Also returns failure traces."""
    ok, failures = 0, []
    for task in tasks:
        res = run_agent(harness, backend, task.prompt, max_turns=max_turns,
                        protocol_example=protocol_example)
        good = res["answer"] is not None and task.check(res["answer"])
        if good:
            ok += 1
        else:
            failures.append({"task_id": task.id, "prompt": task.prompt,
                             "answer": res["answer"], "trace": res["trace"]})
    return ok / max(len(tasks), 1), failures


def seed_harness(rng: random.Random, useful_only: bool = True) -> Harness:
    """Generation-0 harness: all 4 useful tools, optionally + distractors."""
    names = ["calc", "unit_convert", "str_op", "list_op"]
    if not useful_only:
        names += ["approx_calc", "reverse_str", "noop_tool"]
    return Harness(tools=[TOOL_REGISTRY[n].clone() for n in names])


def _make_backend(cfg: Config) -> Backend:
    if cfg.backend == "llm-http":
        from .agent import LLMBackend
        return LLMBackend(cfg.llm_endpoint)
    if cfg.backend == "llm-local":
        from .local_backend import TransformersBackend
        return TransformersBackend(cfg.model_name,
                                   max_new_tokens=cfg.max_new_tokens)
    from .agent import RegexBackend
    return RegexBackend()


def _apply_named_op(child: Harness, rng: random.Random, op: str) -> str:
    """Apply a named operator with a random target. Returns the op used
    ('random_fallback' if the named op could not be applied)."""
    names = child.names()
    target = rng.choice(names) if names else None
    ok = False
    if op == "mutate_docstring" and target:
        ok = child.mutate_docstring(rng, target)
    elif op == "disable_tool" and target:
        ok = child.disable_tool(rng, target)
    elif op == "remove_tool" and target:
        ok = child.remove_tool(rng, target)
    elif op == "enable_tool" and target:
        ok = child.enable_tool(rng, target)
    elif op == "add_tool":
        ok = child.add_tool(rng)
    elif op == "add_validation_wrapper" and target:
        ok = child.add_validation_wrapper(rng, target)
    elif op == "enable_retry" and target:
        ok = child.enable_retry(rng, target)
    elif op == "merge_tools" and len(names) >= 2:
        a, b = rng.sample(names, 2)
        ok = child.merge_tools(rng, a, b)
    elif op == "split_tool" and target:
        ok = child.split_tool(rng, target)
    if ok:
        return op
    ok2, op2 = child.random_mutate(rng)
    return op2 if ok2 else "random_fallback"


def _mutate_arm(arm: str, rng: random.Random, harness: Harness,
                failures: List[dict],
                proposers: Optional[Dict[str, DirectedProposer]] = None
                ) -> Tuple[Harness, str]:
    """Mutate per arm strategy. Returns (child, operator_name)."""
    child = harness.clone()
    if arm == "baseline":
        return child, "none"  # no mutation: pure selection control
    if arm == "prompt":
        child.mutate_preamble(rng)
        return child, "mutate_preamble"
    if arm == "directed":
        proposer = (proposers or {}).get("directed")
        op = proposer(rng, child, failures) if proposer else None
        if op is None:
            ok, op2 = child.random_mutate(rng)
            return child, op2 if ok else "random_fallback"
        return child, _apply_named_op(child, rng, op)
    # "random" (and any unknown arm): random template-based operator.
    ok, op = child.random_mutate(rng)
    return child, op if ok else "random_fallback"


def evolve(cfg: Config, seed: int, arm: str,
           proposers: Optional[Dict[str, DirectedProposer]] = None) -> Path:
    """Run one evolutionary run; write CSV logs + summary.json; return log path."""
    rng = random.Random(seed)
    backend = _make_backend(cfg)
    train, test = train_test_split(seed=seed, train_size=cfg.train_size)

    # Initial population: distractors included so selection has pruning work.
    population = [seed_harness(rng, useful_only=(i % 2 == 0))
                  for i in range(cfg.population_size)]

    results_dir = Path(cfg.results_dir) / f"seed{seed}" / arm
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / cfg.log_csv
    ops_path = results_dir / "ops_log.csv"

    last_best: Optional[Harness] = None
    last_best_fit = 0.0
    with open(log_path, "w", newline="") as f, \
         open(ops_path, "w", newline="") as of:
        writer = csv.writer(f)
        writer.writerow(["generation", "arm", "seed", "best", "mean",
                         "best_tools", "best_preamble_len"])
        op_writer = csv.writer(of)
        op_writer.writerow(["generation", "arm", "seed", "operator", "count"])
        for gen in range(cfg.generations):
            # Evaluate.
            fitness, failures = [], []
            for h in population:
                fit, fails = evaluate_harness(
                    h, backend, train, max_turns=cfg.max_turns,
                    protocol_example=cfg.protocol_example)
                fitness.append(fit)
                failures.append(fails)
            best_idx = max(range(len(population)), key=lambda i: fitness[i])
            best, mean = fitness[best_idx], sum(fitness) / len(fitness)
            last_best = population[best_idx].clone()
            last_best_fit = best
            writer.writerow([gen, arm, seed, f"{best:.4f}", f"{mean:.4f}",
                             ";".join(population[best_idx].names()),
                             len(population[best_idx].preamble)])
            f.flush()
            print(f"[seed {seed} | {arm}] gen {gen}: best={best:.3f} mean={mean:.3f} "
                  f"tools={population[best_idx].names()}", flush=True)

            # Select + reproduce (elitism of 1 + tournament).
            elite = population[best_idx].clone()
            new_pop = [elite]
            op_counts: Counter = Counter()
            while len(new_pop) < cfg.population_size:
                a, b = rng.sample(range(len(population)), cfg.tournament_k)
                winner = a if fitness[a] >= fitness[b] else b
                if rng.random() < cfg.mutation_rate:
                    child, op = _mutate_arm(arm, rng, population[winner],
                                            failures[winner], proposers)
                    op_counts[op] += 1
                else:
                    child, op = population[winner].clone(), "clone"
                    op_counts[op] += 1
                new_pop.append(child)
            for op, cnt in sorted(op_counts.items()):
                op_writer.writerow([gen, arm, seed, op, cnt])
            of.flush()
            population = new_pop

    # Final artifacts: best-harness snapshot + held-out test evaluation.
    assert last_best is not None
    (results_dir / "best_harness.json").write_text(
        json.dumps(last_best.to_spec(), indent=2))
    test_fit, _ = evaluate_harness(last_best, backend, test,
                                   max_turns=cfg.max_turns,
                                   protocol_example=cfg.protocol_example)
    summary = {
        "arm": arm, "seed": seed, "model": cfg.model_name,
        "generations": cfg.generations, "population": cfg.population_size,
        "train_size": len(train), "test_size": len(test),
        "best_train_fitness": round(last_best_fit, 4),
        "best_test_fitness": round(test_fit, 4),
        "best_tools": last_best.names(),
        "n_enabled": len(last_best.enabled_tools()),
        "n_distractors": last_best.distractor_count(),
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[seed {seed} | {arm}] DONE train={last_best_fit:.3f} "
          f"test={test_fit:.3f} tools={last_best.names()}", flush=True)
    return log_path


def run_all(cfg: Config,
            proposers: Optional[Dict[str, DirectedProposer]] = None) -> List[Path]:
    """Run every (seed, arm) combination from the config."""
    logs = []
    for seed in cfg.seeds:
        for arm in cfg.arms:
            logs.append(evolve(cfg, seed, arm, proposers))
    return logs

#!/usr/bin/env python3
"""exp2_llm analysis: build all figures + a results table from downloaded runs.

Usage (from repo root, after `modal volume get evo-harness-runs exp2_llm
./experiments/exp2_llm/results_remote`):
    python3 experiments/exp2_llm/analyze.py

Inputs : experiments/exp2_llm/results_remote/results/seed{1,2}/{arm}/
             fitness_log.csv, ops_log.csv, summary.json, best_harness.json
Outputs: experiments/exp2_llm/figures/*.png + RESULTS.md
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results_remote" / "results"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

ARMS = ["baseline", "prompt", "random", "directed"]
COLORS = {"baseline": "#7f7f7f", "prompt": "#1f77b4",
          "random": "#ff7f0e", "directed": "#2ca02c"}
DISTRACTORS = {"approx_calc", "reverse_str", "noop_tool"}


def load_logs():
    per_arm = defaultdict(lambda: defaultdict(list))  # arm -> gen -> [best...]
    per_arm_mean = defaultdict(lambda: defaultdict(list))
    ops = defaultdict(lambda: defaultdict(int))       # (arm, op) -> total
    summaries = []
    for seed_dir in sorted(RESULTS.glob("seed*")):
        for arm in ARMS:
            d = seed_dir / arm
            fl = d / "fitness_log.csv"
            if not fl.exists():
                continue
            with open(fl) as f:
                for row in csv.DictReader(f):
                    g = int(row["generation"])
                    per_arm[arm][g].append(float(row["best"]))
                    per_arm_mean[arm][g].append(float(row["mean"]))
            with open(d / "ops_log.csv") as f:
                for row in csv.DictReader(f):
                    ops[(arm, row["operator"])] += int(row["count"])
            summaries.append(json.loads((d / "summary.json").read_text()))
    return per_arm, per_arm_mean, ops, summaries


def fig_fitness_curves(per_arm):
    fig, ax = plt.subplots(figsize=(8, 5))
    for arm in ARMS:
        gens = sorted(per_arm[arm])
        if not gens:
            continue
        arr = np.array([per_arm[arm][g] for g in gens])  # gen x seed
        mu, sd = arr.mean(1), arr.std(1)
        ax.plot(gens, mu, label=arm, color=COLORS[arm], lw=2)
        ax.fill_between(gens, mu - sd, mu + sd, color=COLORS[arm], alpha=0.2)
    ax.set_xlabel("generation")
    ax.set_ylabel("train fitness (success rate)")
    ax.set_title("Harness evolution with a fixed 1.5B brain\n(mean ± std across seeds)")
    ax.legend()
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(FIG / "fitness_curves.png", dpi=150)


def fig_test_generalization(summaries):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, key, title in [(axes[0], "best_train_fitness", "train (evolved on)"),
                           (axes[1], "best_test_fitness", "held-out test")]:
        vals = defaultdict(list)
        for s in summaries:
            vals[s["arm"]].append(s[key])
        arms = [a for a in ARMS if a in vals]
        mu = [np.mean(vals[a]) for a in arms]
        sd = [np.std(vals[a]) for a in arms]
        ax.bar(arms, mu, yerr=sd, color=[COLORS[a] for a in arms],
               capsize=4, alpha=0.85)
        ax.set_title(title)
        ax.set_ylim(0, 1.02)
        for i, (m, s_) in enumerate(zip(mu, sd)):
            ax.text(i, m + s_ + 0.02, f"{m:.2f}", ha="center", fontsize=9)
    axes[0].set_ylabel("final elite fitness")
    fig.suptitle("Generalization: does the evolved harness transfer?")
    fig.tight_layout()
    fig.savefig(FIG / "test_generalization.png", dpi=150)


def fig_operators(ops):
    # Group operators into classes for H3: docstring vs structural vs scaffolding.
    classes = {
        "mutate_docstring": "docstring", "mutate_preamble": "preamble",
        "add_tool": "structural", "remove_tool": "structural",
        "disable_tool": "structural", "enable_tool": "structural",
        "merge_tools": "structural", "split_tool": "structural",
        "add_validation_wrapper": "scaffolding", "enable_retry": "scaffolding",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    # Left: raw operator counts for the two mutation arms.
    op_names = sorted({op for (arm, op) in ops if arm in ("random", "directed")
                       and op not in ("none", "clone", "random_fallback")})
    x = np.arange(len(op_names))
    w = 0.35
    for i, arm in enumerate(["random", "directed"]):
        counts = [ops[(arm, op)] for op in op_names]
        axes[0].bar(x + i * w, counts, w, label=arm, color=COLORS[arm], alpha=0.85)
    axes[0].set_xticks(x + w / 2)
    axes[0].set_xticklabels(op_names, rotation=30, ha="right", fontsize=8)
    axes[0].set_ylabel("times applied")
    axes[0].set_title("Operator usage: random vs brain-directed")
    axes[0].legend()
    # Right: class totals (H3).
    for i, arm in enumerate(["random", "directed"]):
        cls_counts = defaultdict(int)
        for (a, op), c in ops.items():
            if a == arm and op in classes:
                cls_counts[classes[op]] += c
        labels = ["docstring", "structural", "scaffolding"]
        axes[1].bar(np.arange(len(labels)) + i * w, [cls_counts[l] for l in labels],
                    w, label=arm, color=COLORS[arm], alpha=0.85)
    axes[1].set_xticks(np.arange(3) + w / 2)
    axes[1].set_xticklabels(["docstring", "structural", "scaffolding"])
    axes[1].set_title("H3: mutation class totals")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIG / "operator_ablation.png", dpi=150)


def fig_size_vs_fitness(per_arm):
    fig, ax = plt.subplots(figsize=(7, 5))
    for seed_dir in sorted(RESULTS.glob("seed*")):
        for arm in ARMS:
            fl = seed_dir / arm / "fitness_log.csv"
            if not fl.exists():
                continue
            with open(fl) as f:
                for row in csv.DictReader(f):
                    n_tools = len(row["best_tools"].split(";")) if row["best_tools"] else 0
                    ax.scatter(n_tools, float(row["best"]), color=COLORS[arm],
                               alpha=0.5, s=28)
    for arm in ARMS:
        ax.scatter([], [], color=COLORS[arm], label=arm, s=40)
    ax.set_xlabel("# tools in best harness (H4: pruning)")
    ax.set_ylabel("train fitness")
    ax.set_title("Harness size vs fitness across all generations")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "size_vs_fitness.png", dpi=150)


def fig_evolved_tool():
    """Before/after docstring of a tool from the directed arm's best harness."""
    spec = None
    for seed_dir in sorted(RESULTS.glob("seed*")):
        p = seed_dir / "directed" / "best_harness.json"
        if p.exists():
            spec = json.loads(p.read_text())
            break
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axis("off")
    if spec is None:
        ax.text(0.5, 0.5, "no directed run found", ha="center")
    else:
        # Pick the tool whose description changed most vs the registry default.
        from pathlib import Path as _P
        import sys
        sys.path.insert(0, str(_P(__file__).resolve().parent.parent.parent / "src"))
        from evoharness.harness import TOOL_REGISTRY
        best, best_len = None, -1
        for t in spec["tools"]:
            base = TOOL_REGISTRY.get(t["name"])
            if base and len(t["description"]) > best_len:
                best, best_len = t, len(t["description"])
        lines = ["Evolved tool docstring (directed arm, final elite):",
                 "",
                 f"tool: {best['name']}",
                 "",
                 "BEFORE (gen 0):",
                 TOOL_REGISTRY[best["name"]].description,
                 "",
                 "AFTER (evolved):",
                 best["description"]]
        ax.text(0.02, 0.95, "\n".join(lines), va="top", ha="left",
                fontsize=9, family="monospace", wrap=True)
    ax.set_title("An evolved limb: docstring before → after")
    fig.tight_layout()
    fig.savefig(FIG / "evolved_tool_example.png", dpi=150)


def write_results_md(summaries):
    lines = ["# exp2_llm results", "",
             "Fixed brain: Qwen2.5-1.5B-Instruct (greedy). "
             "Fitness = success rate on synthetic tool-use tasks.", "",
             "| arm | seed | train | test | tools | distractors |",
             "|---|---|---|---|---|---|"]
    for s in sorted(summaries, key=lambda s: (s["arm"], s["seed"])):
        lines.append(f"| {s['arm']} | {s['seed']} | {s['best_train_fitness']:.3f} | "
                     f"{s['best_test_fitness']:.3f} | {';'.join(s['best_tools'])} | "
                     f"{s['n_distractors']} |")
    # Arm means.
    lines += ["", "## Arm means (test fitness)", ""]
    vals = defaultdict(list)
    for s in summaries:
        vals[s["arm"]].append(s["best_test_fitness"])
    for arm in ARMS:
        if arm in vals:
            v = vals[arm]
            lines.append(f"- **{arm}**: {np.mean(v):.3f} ± {np.std(v):.3f} (n={len(v)})")
    (HERE / "RESULTS.md").write_text("\n".join(lines) + "\n")


def main():
    per_arm, _, ops, summaries = load_logs()
    assert summaries, f"no runs found under {RESULTS}"
    fig_fitness_curves(per_arm)
    fig_test_generalization(summaries)
    fig_operators(ops)
    fig_size_vs_fitness(per_arm)
    fig_evolved_tool()
    write_results_md(summaries)
    print(f"figures -> {FIG}")
    print((HERE / "RESULTS.md").read_text())


if __name__ == "__main__":
    main()

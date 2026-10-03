"""Experiment configuration (dataclass, YAML-serializable)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import yaml


@dataclass
class Config:
    # Evolution
    population_size: int = 8
    generations: int = 10
    seeds: List[int] = field(default_factory=lambda: [1, 2, 3])
    arms: List[str] = field(default_factory=lambda: ["baseline", "prompt", "random"])
    tournament_k: int = 2
    mutation_rate: float = 0.5  # fraction of offspring mutated per generation

    # Benchmark split
    train_size: int = 24
    test_size: int = 16
    total_tasks: int = 40

    # Model (fixed brain — never mutated)
    model_name: str = "regex-dummy"
    backend: str = "regex"  # "regex" | "llm-http" | "llm-local"
    llm_endpoint: str = "http://localhost:8000/generate"
    max_turns: int = 6
    max_new_tokens: int = 160  # per model call (llm-local)
    protocol_example: bool = False  # worked CALL/FINAL example in system prompt

    # Paths
    results_dir: Path = Path("results")
    log_csv: str = "fitness_log.csv"

    def to_yaml(self, path: Path) -> None:
        d = dict(self.__dict__)
        d["results_dir"] = str(d["results_dir"])
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(d, sort_keys=False))

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        d = yaml.safe_load(Path(path).read_text())
        d["results_dir"] = Path(d.get("results_dir", "results"))
        return cls(**d)


def default_config() -> Config:
    return Config()

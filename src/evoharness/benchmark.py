"""Synthetic tool-use benchmark: 40 tasks, 4 families of 10.

Families:
  1. arithmetic_word_problems — multi-step calc requiring calc tools
  2. string_transform           — case/split/join ops
  3. unit_conversion            — length/weight/temp
  4. list_ops                   — sort/filter/sum on small lists

Each task: id, family, prompt, gold_answer, checker(answer) -> bool.
Gold answers are deterministic and computable at import time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

FAMILIES = ["arithmetic", "string", "unit", "list"]

Checker = Callable[[str], bool]


def _exact(gold: str) -> Checker:
    gold_n = gold.strip()
    def check(answer: str) -> bool:
        return answer.strip() == gold_n
    return check


def _float_close(gold: float, tol: float = 1e-6) -> Checker:
    def check(answer: str) -> bool:
        try:
            return abs(float(answer.strip().rstrip(".")) - gold) <= tol
        except ValueError:
            return False
    return check


@dataclass
class Task:
    id: str
    family: str
    prompt: str
    gold_answer: str
    checker: Checker

    def check(self, answer: str) -> bool:
        return self.checker(answer)


def _arithmetic_tasks() -> List[Task]:
    # (prompt, expr) — expr evaluated deterministically for the gold answer.
    items = [
        ("What is 12 * 8 + 15?", "12*8+15"),
        ("A box holds 24 apples. You add 3 boxes worth, then give away 17. How many?", "24+3*24-17"),
        ("Compute (45 + 55) * 2 - 30.", "(45+55)*2-30"),
        ("A train goes 60 km/h for 3 hours, then 80 km/h for 2 hours. Total km?", "60*3+80*2"),
        ("What is 7^3 divided by 7?", "7**3/7"),
        ("You have $500, spend $129, earn $250, spend $88. Balance?", "500-129+250-88"),
        ("Compute 1000 / 8 + 15 * 4.", "1000/8+15*4"),
        ("A recipe needs 2.5 cups flour per batch; you make 4 batches. Cups needed?", "2.5*4"),
        ("What is 15% of 240?", "0.15*240"),
        ("Compute (123 + 456) * (789 - 700) / 9.", "(123+456)*(789-700)/9"),
    ]
    tasks = []
    for i, (prompt, expr) in enumerate(items):
        gold = str(eval(expr, {"__builtins__": {}}))
        tasks.append(Task(id=f"arith_{i:02d}", family="arithmetic", prompt=prompt,
                          gold_answer=gold, checker=_exact(gold)))
    return tasks


def _string_tasks() -> List[Task]:
    items = [
        ("Convert to UPPERCASE: 'hello world'", "hello world", "upper", "HELLO WORLD"),
        ("Convert to lowercase: 'Good Morning'", "Good Morning", "lower", "good morning"),
        ("Title-case: 'the quick brown fox'", "the quick brown fox", "title", "The Quick Brown Fox"),
        ("Reverse the word order of: 'one two three'", "one two three", "reverse_words", "three two one"),
        ("Remove all spaces from: 'a b c d'", "a b c d", "strip_spaces", "abcd"),
        ("Count the characters in: 'abcdefg'", "abcdefg", "count_chars", "7"),
        ("Split on commas and join with pipes: 'red, green, blue'", "red, green, blue", "split_commas", "red|green|blue"),
        ("Convert to UPPERCASE: 'agent 007'", "agent 007", "upper", "AGENT 007"),
        ("Reverse the word order of: 'alpha beta'", "alpha beta", "reverse_words", "beta alpha"),
        ("Count the characters in: 'tool use!'", "tool use!", "count_chars", "9"),
    ]
    tasks = []
    for i, (prompt, text, op, gold) in enumerate(items):
        tasks.append(Task(id=f"str_{i:02d}", family="string", prompt=prompt,
                          gold_answer=gold, checker=_exact(gold)))
    return tasks


def _unit_tasks() -> List[Task]:
    items = [
        ("Convert 5 km to meters.", 5.0, "km", "m", 5000.0),
        ("Convert 2 miles to kilometers.", 2.0, "mi", "km", 2 * 1609.34 / 1000.0),
        ("Convert 10 kg to pounds.", 10.0, "kg", "lb", 10.0 / 0.453592),
        ("Convert 32 fahrenheit to celsius.", 32.0, "f", "c", 0.0),
        ("Convert 100 celsius to fahrenheit.", 100.0, "c", "f", 212.0),
        ("Convert 12 feet to inches.", 12.0, "ft", "in", 144.0),
        ("Convert 500 grams to kilograms.", 500.0, "g", "kg", 0.5),
        ("Convert 1 inch to centimeters.", 1.0, "in", "cm", 2.54),
        ("Convert 3 ounces to grams.", 3.0, "oz", "g", 3 * 0.0283495 / 0.001),
        ("Convert 2500 mm to meters.", 2500.0, "mm", "m", 2.5),
    ]
    tasks = []
    for i, (prompt, value, frm, to, gold) in enumerate(items):
        tasks.append(Task(id=f"unit_{i:02d}", family="unit", prompt=prompt,
                          gold_answer=str(gold), checker=_float_close(gold)))
    return tasks


def _list_tasks() -> List[Task]:
    from .harness import _numstr
    # (prompt, items_str, op, gold_str)
    items = [
        ("Sum these numbers: 1, 2, 3, 4, 5", "1,2,3,4,5", "sum", _numstr(15)),
        ("Mean of: 10, 20, 30", "10,20,30", "mean", _numstr(20.0)),
        ("Max of: 3, 9, 1, 7", "3,9,1,7", "max", _numstr(9.0)),
        ("Min of: 3, 9, 1, 7", "3,9,1,7", "min", _numstr(1.0)),
        ("Sort ascending: 5, 2, 8, 1", "5,2,8,1", "sort_asc", "1.0,2.0,5.0,8.0"),
        ("Sort descending: 5, 2, 8, 1", "5,2,8,1", "sort_desc", "8.0,5.0,2.0,1.0"),
        ("Count the numbers: 4, 4, 4", "4,4,4", "count", "3"),
        ("Keep only positives from: -1, 2, -3, 4", "-1,2,-3,4", "filter_positive", "2.0,4.0"),
        ("Sum these numbers: 0.5, 1.5, 2.0", "0.5,1.5,2.0", "sum", _numstr(4.0)),
        ("Mean of: 2, 4", "2,4", "mean", _numstr(3.0)),
    ]
    tasks = []
    for i, (prompt, items_str, op, gold) in enumerate(items):
        tasks.append(Task(id=f"list_{i:02d}", family="list", prompt=prompt,
                          gold_answer=gold, checker=_exact(gold)))
    return tasks


def build_tasks() -> List[Task]:
    return _arithmetic_tasks() + _string_tasks() + _unit_tasks() + _list_tasks()


TASKS: List[Task] = build_tasks()

assert len(TASKS) == 40, f"expected 40 tasks, got {len(TASKS)}"
assert all(len([t for t in TASKS if t.family == f]) == 10 for f in FAMILIES)

# Sanity: every task's gold answer passes its own checker.
for _t in TASKS:
    assert _t.check(_t.gold_answer), f"self-check failed for {_t.id}"


def make_tools():
    """Convenience: fresh tool implementation dict from the registry."""
    from .harness import TOOL_REGISTRY
    return {name: td.clone() for name, td in TOOL_REGISTRY.items()}


def train_test_split(seed: int = 0, train_size: int = 24) -> tuple:
    """Deterministic train/test split, stratified by family."""
    import random
    rng = random.Random(seed)
    train, test = [], []
    for fam in FAMILIES:
        fam_tasks = [t for t in TASKS if t.family == fam]
        rng.shuffle(fam_tasks)
        n_train = int(round(train_size / len(FAMILIES)))
        train.extend(fam_tasks[:n_train])
        test.extend(fam_tasks[n_train:])
    return train, test

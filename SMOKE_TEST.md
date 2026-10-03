# SMOKE_TEST.md

Run: 2026-10-02 (PDT). Command: `PYTHONPATH=src python3` heredoc against the
`evoharness` package, plus `evolve()` on a tiny config (pop 4, 2 generations,
arms baseline/random, seed 1, train 24 tasks, `regex` backend).

## Result: PASS

### 1. Agent end-to-end (RegexBackend, fixed brain) — 10/10 tasks pass
Sampled across all 4 benchmark families with the full useful toolset:

| task | answer | gold | result |
|---|---|---|---|
| arith_00 | 111 | 111 | PASS |
| arith_02 | 170 | 170 | PASS |
| arith_06 | 185.0 | 185.0 | PASS |
| str_00 | HELLO WORLD | HELLO WORLD | PASS |
| str_05 | 7 | 7 | PASS |
| unit_00 | 5000.0 | 5000.0 | PASS |
| unit_03 | 0.0 | 0.0 | PASS |
| list_00 | 15 | 15 | PASS |
| list_04 | 1.0,2.0,5.0,8.0 | 1.0,2.0,5.0,8.0 | PASS |
| list_07 | 2.0,4.0 | 2.0,4.0 | PASS |

### 2. All mutation operators — OK, no crashes
Exercised with correct arguments and verified the harness still renders
(`to_system_prompt()`): `mutate_docstring`, `add_tool`, `remove_tool`,
`disable_tool`, `enable_tool`, `merge_tools`, `split_tool`,
`add_validation_wrapper`, `enable_retry`, `mutate_preamble`, plus the generic
`random_mutate` path (landed on `disable_tool`).

### 3. Evolve loop — CSV output correct
- 2 generations × 2 arms completed; CSV has the expected header
  (`generation,arm,seed,best,mean,best_tools,best_preamble_len`) and one row
  per generation.
- Baseline fitness 0.833 on train split (dummy backend ceiling — word-problem
  arithmetic the rule-based brain can't parse, e.g. "What is 15% of 240?").

## Bugs found and fixed during the smoke test
1. `execute_call` crashed on JSON key `"from"` (Python keyword) — added a
   protocol shim mapping `"from"` → `"frm"` in `agent.py`.
2. `RegexBackend._arith_expr` required a trailing `?`, so "Compute … ." prompts
   produced garbage expressions — regex rewritten to grab from the first digit.
3. `list_op` returned `"15.0"` vs gold `"15"` — added `_numstr` formatting and
   computed benchmark golds with the same helper.
4. `unit_convert` rejected natural-language units ("meters") — added a unit
   alias table to the limb.
5. `RegexBackend` re-emitted `CALL` forever after receiving a result — now
   emits `FINAL <last result>` once a `Result of …` line is in the transcript.

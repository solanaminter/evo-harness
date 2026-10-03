# exp2_llm results

Fixed brain: Qwen2.5-1.5B-Instruct (greedy). Fitness = success rate on synthetic tool-use tasks.

| arm | seed | train | test | tools | distractors |
|---|---|---|---|---|---|
| baseline | 1 | 0.417 | 0.286 | calc;unit_convert;str_op;list_op;approx_calc;reverse_str;noop_tool | 3 |
| baseline | 2 | 0.583 | 0.429 | calc;unit_convert;str_op;list_op | 0 |
| directed | 1 | 0.500 | 0.357 | calc;unit_convert;str_op;list_op;approx_calc;reverse_str;noop_tool | 3 |
| directed | 2 | 0.583 | 0.429 | calc;unit_convert;str_op;list_op | 0 |
| prompt | 1 | 0.500 | 0.357 | calc;unit_convert;str_op;list_op | 0 |
| prompt | 2 | 0.750 | 0.393 | calc;unit_convert;str_op;list_op | 0 |
| random | 1 | 0.500 | 0.357 | calc;unit_convert;str_op;list_op;approx_calc;reverse_str;noop_tool | 3 |
| random | 2 | 0.667 | 0.321 | calc;unit_convert;str_op;list_op | 0 |

## Arm means (test fitness)

- **baseline**: 0.357 ± 0.071 (n=2)
- **prompt**: 0.375 ± 0.018 (n=2)
- **random**: 0.339 ± 0.018 (n=2)
- **directed**: 0.393 ± 0.036 (n=2)

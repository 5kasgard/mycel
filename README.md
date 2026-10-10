# Mycel

A fungus-inspired computing substrate of "dumb" nuclei. This repository holds the digital-native version (Mycel-D) and its test harness.

| File | What it is |
|---|---|
| `dcore.c`, `dmycel.py` | Mycel-D: nuclei as bitmasks, tubes as 8-bit counters, yes/no contests; C core on 64-bit words; energy model from the Horowitz 45 nm cost table |
| `baselines_d.py` | Digital baselines: numpy MLP (float32, int8, pruned int8), EWC |
| `protect.py` | Bit-flip faults and storage protection (parity, SECDED) for both sides |
| `tasks4.py`, `data_orig.py` | Original five tasks and six unseen tasks |
| `exp20.py` | Round four test for one task and seed range |
| `exp21_forget.py`, `forget_d.py` | Round four forgetting test |
| `analyze_r4.py` | Scores round four by the pre-registered rules |
| `workflow-run.yml` | GitHub Actions runner; must live at `.github/workflows/run.yml` (see below) |

## Running experiments on GitHub Actions

The workflow file has to be added by the repository owner (the assistant's GitHub access cannot write workflow files): create `.github/workflows/run.yml` with the contents of `workflow-run.yml`.

Then edit `jobs.json` (a list of `{"id", "cmd"}`) and push. Each entry runs as its own parallel job; all `results_*.json` files are committed to `results/` when every job ends.

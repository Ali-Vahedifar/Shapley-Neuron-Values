# CIFAR-100 results

The reported CIFAR-100 numbers: what was selected, what it scored, and the
artifacts each number came from.  Everything here is the **clean evaluation
half** (D_E, seeds 42/43/44) of the GTEP campaign described in
[../../docs/PROTOCOL.md](../../docs/PROTOCOL.md).

| File | What it is |
|---|---|
| `metrics_summary.md` / `.csv` | ACC, AvgAcc, BWT, FWT, PS, AF per method and scenario, mean ± sd over the three seeds |
| `metrics_per_seed.csv` | the same, one row per run |
| `costs_summary.csv` | GPU-hours, minutes per task, peak GPU memory, energy, parameter count, checkpoint size, and whether every seed held the GPU exclusively |
| `accuracy_matrices.json` | the 10×10 task-by-task accuracy matrix of every run |
| `search_space.md` / `.json` | the search space, and every one of the R = 30 trials per method with its D_HT score |
| `blocks/<method>_<scenario>.json` | the campaign block: all tuning trials, the winner, the pointers to its clean runs |
| `runs/<run>/result.json` | one clean run: config, metrics, matrix, per-task histories, inference records, cost summary, hardware, class order |
| `runs/<run>/command.json` | the exact command line that produced it |
| `runs/<run>/train.log.gz` | its training log |

The selected hyperparameters themselves are one level up, in
[../../hyperparameters/cifar100_best.md](../../hyperparameters/cifar100_best.md),
with `best_config('snv', 'class_il')` to read them from code.

## The best version

**SNV-A** (`SNV/snv_adaptive.py`, run through `snv_adaptive_run.py`) is the SNV
reported here, and the best of the two: the dense SNV that preceded it collapses
to chance in Class-IL (its search block, `blocks/snv_class_il.json`, tops out at
HARMONIC 0.044 and never earned a clean run).  SNV-A adds task-local training,
routed Class-IL inference and a per-task adaptive mask size.

| scenario | ACC (D_E) | rank among the methods here |
|---|---:|---|
| Class-IL | 0.3618 ± 0.0206 | best except the joint-training upper bound |
| Task-IL | 0.8175 ± 0.0157 | third, behind joint training and WSN (which is given the task identity) |

Its selected configuration, both scenarios:

```
lr = 0.0012981647010649356, truncation = 0.05,
task_local, frozen_norm_eval, bn_recal (batch), adaptive (coverage @ 0.9)
routing = rot_energy_z, rot_aux = 1.0    # Class-IL
routing = maxprob,       rot_aux = 0.0   # Task-IL
```

Reproduce it with `bash scripts/run_cifar100_snv_best.sh`.

## Names

The campaign directory names are kept as they were written, so a `command.json`
still matches its run.  Two of them differ from the names this package uses:
`snv_adaptive` is **snv** (SNV-A), and `mcl3` is **mcl**.  The tables above use
the package names.

## What is not here

* **Checkpoints.**  Each clean run's `ckpt_run0.pt` is 45-130 MB, 6.2 GB for the
  66 runs, so they stay in the campaign directory rather than in this package:
  `/root/Snv_continual_learning/gtep_paper_first_20260906/runs/<run>/ckpt_run0.pt`.
  `scripts/run_cifar100_selected.sh` regenerates a run, checkpoint included.
* **The 1,832 tuning runs** (133 GB).  Their configurations and scores are all
  in `blocks/` and `search_space.json`; only their raw run directories are left
  behind.
* **The raw cost ledger** (`costs.json`, ~5 MB per run).  Its summary is in every
  `result.json` under `cost_summary`, and aggregated in `costs_summary.csv`.
* **Per-neuron arrays** inside `result.json` histories (SNV masks and the like)
  were pruned on import; each is replaced by `{"_pruned_array": <length>}`.
  `results/import_cifar100.py` does the pruning and documents it.

## Rebuilding

```bash
bash scripts/export_cifar100_results.sh                      # re-tabulate what is here
CAMPAIGN=/path/to/campaign bash scripts/export_cifar100_results.sh   # re-import, then tabulate
```

# GTEP evaluation protocol

This document describes the protocol on CIFAR-100, the benchmark the reported
numbers come from.  The same code runs CIFAR-20, TinyImageNet-200 and
ImageNet-1k: pass `--dataset`, and the half geometry follows from the class
count (`DATASET_CLASSES` / `geometry()` in `audited_gtep.py`).

| dataset | classes | per half | tasks | classes per task |
|---|---:|---:|---:|---:|
| `cifar20` | 20 | 10 | 5 | 2 |
| `cifar100` | 100 | 50 | 10 | 5 |
| `tinyimagenet` | 200 | 100 | 10 | 10 |
| `imagenet1k` | 1000 | 500 | 10 | 50 |

`GTEP_TASKS` overrides the task count. Only CIFAR-100 has been run end to end
under this package; the other three are wired up but untuned.


This is implemented in `audited_gtep.py`, and `campaign/run_campaign.py` runs it
end to end.

## Data

* CIFAR-100's train and test sets are pooled. Each class is split 70 / 10 / 20
  into train / validation / test (`datasets.py`, `TRAIN_FRAC/VAL_FRAC/TEST_FRAC`).
* The 100 classes are split into two **disjoint halves** of 50 classes with a
  fixed `split_seed = 1234`:
  * **D_HT** (half 1) is used only to select hyperparameters. Runs on it are
    scored on the *validation* split.
  * **D_E** (half 2) is used only for the reported numbers, scored on the *test*
    split.
* Each half is a sequence of **10 tasks × 5 classes**. Class and task order
  follow the run seed (42, 43, 44).
* The two scenarios:
  * **Class-IL** predicts over every class seen so far, with no task identity.
  * **Task-IL** is given the task identity and uses that task's head.

## Training policy (`GTEP_PROTOCOL=legacy`, the setting the paper's numbers use)

| Setting | Value |
|---|---|
| Optimiser | Adam, weight decay 0, no LR schedule |
| Batch size | 64 |
| Epochs | at most 200 per task, early stopping with patience 15 on validation loss |
| Backbone | CIFAR ResNet-18, except PEC, NISPA and SpaceNet, which use their own architectures (docs/BASELINES.md) |
| Replay buffer | 1,000 examples for UniCLUN, the only memory-based method |

With `GTEP_PROTOCOL=paper`, the module instead uses SGD with method-specific
schedules and a wider search space. The paper's CIFAR-100 numbers were **not**
produced this way.

## Hyperparameter search

1. For each method, **R = 30** configurations are drawn from its search space
   (`SPACE` in `audited_gtep.py`) with sample seed 7.
2. Each configuration is trained on D_HT with seeds 42, 43 and 44.
3. **Selection score:** `HARMONIC = 2·ACC·AvgAcc / (ACC + AvgAcc)` on the D_HT
   validation split, averaged over the three seeds. The configuration with the
   highest mean wins. D_E never influences selection.
4. The winner is retrained on D_E with seeds 42, 43 and 44. These three runs
   are the reported numbers.

## One run (`audited_gtep.one`)

1. Build the benchmark, the method and the resolved training policy.
2. **Random-init row.** Evaluate the untrained model on every task. This gives
   `b_t` for FWT.
3. For each task t:
   * train it, including internal validation and early stopping;
   * fill row t of the accuracy matrix `A`: seen tasks `k ≤ t`, and future
     tasks `k > t` in the full output space (this provides the superdiagonal
     `A[t, t+1]` that FWT and plasticity need);
   * save a checkpoint and write `progress.json`.
4. Compute metrics (`metrics.py`, see METRICS.md), plus `AvgAcc` and `HARMONIC`.
5. Run the inference benchmark at batch sizes 1 and 64 for every task (see COSTS.md).
6. Write `result.json`. It contains the matrix, metrics, config, policy,
   histories, inference records, cost summary, hardware and class order.

## Cost-measurement rules

* Tuning runs may share a GPU. Their timings are never reported.
* Each D_E winner run takes the GPU **exclusively** (a per-GPU file lock plus
  an NVML check that no other process is on the device). The ledger records
  whether exclusivity held for every phase. `build_report.py` refuses to
  report a run where it did not.
* Inference FLOPs are counted in a separate pass outside the latency timer.

## SNV (SNV-A)

`snv_adaptive_run.py` only replaces the SNV constructor with `SNVAdaptive`.
Splits, epochs, metrics, checkpointing and the cost ledger are the same as for
every other method. These switches are fixed and recorded in each config:
`task_local, frozen_norm_eval, bn_recal, adaptive, adaptive_rule=coverage,
adaptive_coverage=0.9`, plus the routing rule, which differs by scenario:
`routing=maxprob` under Task-IL and `routing=rot_energy_z, rot_aux=1.0` under
Class-IL (see `hyperparameters/cifar100_best.md`). The Shapley estimator settings
are `reverse_tmc`, 32 permutations, loss payoff, masked training, and
consolidation for half of the per-task epoch budget. Only `lr` and the
truncation threshold are searched.

## Provenance

`run_campaign.py` records a SHA-256 of every file that can change a result
(`SOURCE_FILES` in `audited_gtep.py`) in `protocol.json`. It refuses to
continue if any of those files changes while a campaign directory is in use.

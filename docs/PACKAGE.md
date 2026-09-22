# Shapley Neuron Value (SNV)

SNV values each neuron by its Shapley contribution to the task's loss, freezes
the neurons that carry the value, and gives the next task what is left.  The
version here is **SNV-A** (`SNV/snv_adaptive.py`) -- task-local training, routed
Class-IL inference and a per-task adaptive mask size.  It is the SNV the results
were produced with; the dense SNV that preceded it is still in
`SNV/snv_core.py`, because SNV-A is built on top of it.

This package holds SNV, every continual-learning baseline it is compared
against, the GTEP evaluation protocol, the CL metrics, the cost measurements,
the data loaders for four benchmarks and the CIFAR-100 hyperparameters that were
selected under the protocol.  No checkpoints, no run directories -- source code
and the selected configurations.

## Layout

```
SNV/
  snv_core.py            Shapley neuron valuation, masks, the dense SNV learner
  snv_adaptive.py        SNV-A: task-local phases, routed CIL, adaptive |S_t|
snv_adaptive_run.py      runs SNV-A through the unchanged GTEP worker
snv_core.py              import shim for SNV/snv_core.py

baselines/
  __init__.py            the method registry (build_method, ALL_METHODS, ...)
  regularization.py      re-exports EWC, SI, LwF, SGD
  sparse.py              re-exports WSN, SpaceNet, NISPA, PEC
  joint_audited.py       JointPrefix, the audited joint-training upper bound
  SGD/ Joint/            bounds
  EWC/ SI/ LwF/          regularisation
  WSN/ PEC/ SpaceNet/ NISPA/   sparse / architecture
  UniCLUN/               continual learning + machine unlearning
  MCL/                   Matryoshka Continual Learning

audited_gtep.py          GTEP: halves, search spaces, one run, the run queue
campaign/
  run_campaign.py        the whole campaign: R x 3-seed search, then 3 clean runs
  build_report.py        metric / cost / hyperparameter tables
  snv_adaptive_compare.py
datasets.py              CIFAR-100, CIFAR-20, TinyImageNet-200, ImageNet-1k, PMNIST
models.py                ResNet-18 / ResNet-50 backbones, multi-head models
metrics.py               ACC, BWT, FWT, PS (+ P, S, AF)
audit_cost.py            cost ledger: GPU-hours, peak memory, params, GFLOPs, energy
cost.py, inrun.py        lightweight trackers used by train.py
train.py                 standalone single-method trainer
cl_base.py, training_policy.py, utils.py, method_loader.py, wsn_helpers.py

hyperparameters/
  cifar100_best.json     the selected config + clean-half metrics per method
  cifar100_best.md       the same as a table
  __init__.py            best_config(method, scenario); python -m hyperparameters
  extract_cifar100_best.py   regenerates both files from a campaign directory

results/cifar100/        the reported CIFAR-100 results (11 MB, no checkpoints)
  metrics_summary.md/.csv, metrics_per_seed.csv, costs_summary.csv
  accuracy_matrices.json, search_space.md/.json
  blocks/                every tuning trial, its score and the winner
  runs/<run>/            result.json, command.json, train.log.gz per clean run
  import_cifar100.py, make_cifar100_tables.py
scripts/
  smoke_test.sh              every method, 2 tasks, 1 epoch, 64 samples ($DATASET picks one)
  run_cifar100_campaign.sh   the full campaign that produced results/cifar100
  run_cifar100_selected.sh   rerun the selected configs on the clean half
  run_cifar100_snv_best.sh   the best version (SNV-A) alone
  export_cifar100_results.sh rebuild the hyperparameter and result tables

docs/                    PROTOCOL.md, METRICS.md, COSTS.md, BASELINES.md
tests/                   SNV, SNV-A, baselines, metrics, datasets, hyperparameters
```

## Methods

| Group | Methods |
|---|---|
| Ours | **SNV** (SNV-A) |
| Bounds | SGD (lower), Joint training (upper) |
| Regularisation | EWC, SI, LwF |
| Sparse / architecture | WSN (Task-IL only), PEC (Class-IL only), SpaceNet, NISPA |
| CL + unlearning | UniCLUN |
| Matryoshka | MCL (+ the `mcl_uniform` and `mcl_g` ablation arms) |

WSN needs the task identity at test time, so it is evaluated only in Task-IL;
PEC is defined only for Class-IL.  Implementation notes per baseline are in
[docs/BASELINES.md](BASELINES.md), upstream repositories in
[THIRD_PARTY.md](../THIRD_PARTY.md).

## Datasets

`datasets.py` builds all four benchmarks the same way: the native train and test
partitions are pooled per class and re-split 70 / 10 / 20, so the proportions
hold for every dataset (TinyImageNet ships 91 / 9, which slicing alone cannot
fix).  Labels are remapped per scenario -- cumulative global indices for
Class-IL, task-local 0..C-1 for Task-IL.

| `--dataset` | classes | source | per GTEP half | tasks x classes |
|---|---:|---|---:|---|
| `cifar100` | 100 | downloads | 50 | 10 x 5 |
| `cifar20` | 20 | the CIFAR-100 superclasses (coarse labels) | 10 | 5 x 2 |
| `tinyimagenet` | 200 | downloads | 100 | 10 x 10 |
| `imagenet1k` | 1000 | supply it yourself (ImageFolder at `$DATA/imagenet`) | 500 | 10 x 50 |

CIFAR-20 is CIFAR-100's images under its 20 superclass labels; torchvision does
not expose them, so `datasets.CIFAR20` reads the coarse labels from the same
pickle.  Set `GTEP_DATA_ROOT` (or `--data_root`) to the dataset directory;
CIFAR and TinyImageNet download on first use, ImageNet-1k cannot and must be
arranged as class-per-directory.  `GTEP_TASKS` overrides the task count.

## Install

```bash
pip install -r requirements.txt     # Python 3.10, PyTorch 2.6 / CUDA 12.4
```

## CIFAR-100 results

The reported numbers, the artifacts behind them and the search that produced
them are in [results/cifar100/](../results/cifar100/) (11 MB: tables, campaign
blocks, one `result.json` + `command.json` + gzipped log per clean run; the
45-130 MB checkpoints stay in the campaign directory, see that folder's README).

Class-IL ACC on the clean half, mean ± sd over seeds 42/43/44
([full table](../results/cifar100/metrics_summary.md), Task-IL included):

| method | ACC | | method | ACC |
|---|---:|---|---|---:|
| joint (upper bound) | 0.6543 ± 0.0240 | | sgd (lower bound) | 0.0872 ± 0.0070 |
| **snv (SNV-A)** | **0.3618 ± 0.0206** | | si | 0.0856 ± 0.0049 |
| pec | 0.3284 ± 0.0125 | | ewc | 0.0839 ± 0.0066 |
| uniclun | 0.3275 ± 0.0404 | | nispa | 0.0824 ± 0.0054 |
| mcl | 0.2722 ± 0.0255 | | spacenet | 0.0677 ± 0.0043 |
| lwf | 0.1006 ± 0.0179 | | | |

SNV-A is the best version of SNV and the best method here apart from the joint
upper bound; the dense SNV it replaced never left chance in Class-IL. In Task-IL
it reaches 0.8175 ± 0.0157, behind joint training and WSN.

## Selected CIFAR-100 hyperparameters

Tuned on D_HT (half 1, 3 seeds, 30 configs per method) and reported on the three
clean D_E runs, under the protocol in [docs/PROTOCOL.md](PROTOCOL.md).  The
full table, with per-method Class-IL and Task-IL winners and their clean-half
ACC / BWT / FWT / PS, is [hyperparameters/cifar100_best.md](../hyperparameters/cifar100_best.md);
the space itself and all 30 trials per method are in
[results/cifar100/search_space.md](../results/cifar100/search_space.md).

SNV-A's winners:

| scenario | config | clean D_E ACC |
|---|---|---:|
| Class-IL | `lr=0.0012981647, truncation=0.05, task_local, frozen_norm_eval, bn_recal=batch, adaptive=coverage@0.9, routing=rot_energy_z, rot_aux=1.0` | 0.3618 ± 0.0206 |
| Task-IL | `lr=0.0012981647, truncation=0.05, task_local, frozen_norm_eval, bn_recal=batch, adaptive=coverage@0.9, routing=maxprob, rot_aux=0.0` | 0.8175 ± 0.0157 |

In code:

```python
from hyperparameters import best_config, entry
best_config('snv', 'class_il')        # the dict to pass as --config
entry('ewc', 'task_il')['clean_eval_D_E']
```

or print a ready-to-run command:

```bash
python -m hyperparameters snv class_il
```

## Running

A single run (one method, scenario, half, seed):

```bash
GTEP_PROTOCOL=legacy python audited_gtep.py --one --method ewc --scenario task_il \
    --dataset cifar100 --half 2 --seed 42 --epochs 200 --patience 15 \
    --out runs/ewc_til_s42 --config '{"lr":0.0007,"ewc_lambda":2029.6,"ewc_gamma":0.912}'
```

SNV-A takes the same flags through its own entry point (no `--method`):

```bash
python snv_adaptive_run.py --scenario class_il --dataset cifar100 --half 2 --seed 42 \
    --epochs 200 --patience 15 --out runs/snv_cil_s42 \
    --config "$(python -c "import json,hyperparameters as h;print(json.dumps(h.best_config('snv','class_il')))")"
```

The whole campaign and the paper tables:

```bash
python campaign/run_campaign.py --out runs/cifar100 --gpus 0 1 2 3
python campaign/build_report.py --campaign runs/cifar100 --out reports/cifar100
```

`--methods snv mcl` or `--scenarios class_il` runs a subset.

## Quick check

```bash
bash scripts/smoke_test.sh                  # every method, 2 tasks, 1 epoch, 64 samples
DATASET=cifar20 bash scripts/smoke_test.sh  # the same on another benchmark
python -m pytest -q tests
```

`tests/` covers the SNV estimator and masks, SNV-A, the audited baselines, the
metrics and cost ledger, the four benchmarks' geometry and splits, and the
exported hyperparameters.

The smoke script skips SNV-A on CPU -- its Shapley valuation takes over 30
minutes on ResNet-18 without a GPU.  `tests/test_snv.py::TestSNVAdaptive` covers
SNV-A on CPU in seconds; `GTEP_DEVICE=cuda:0 bash scripts/smoke_test.sh`
includes it.

## Provenance

The CIFAR-100 hyperparameters in `hyperparameters/` were exported from the
frozen GTEP campaign at `gtep_paper_first_20260906`; `extract_cifar100_best.py`
records the campaign path and the export date in the JSON.  Only CIFAR-100 has
been run end to end here -- CIFAR-20, TinyImageNet and ImageNet-1k are wired
through the same loaders, protocol and registry, but untuned.

Each run used a single GPU; the runs behind `hyperparameters/cifar100_best.*`
were measured on an NVIDIA RTX 6000 Ada Generation with PyTorch 2.6 / CUDA
12.4, as each `result.json` records under `hardware`.

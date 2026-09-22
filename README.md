# Shapley Neuron Valuation for Continual Learning
### *Which Neurons Matter Most?*

**Accepted at ICML 2026** 🎉

[![Paper](https://img.shields.io/badge/Paper-ICML%202026-blue)](https://arxiv.org/abs/XXXX.XXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-yellow.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-ee4c2c.svg)](https://pytorch.org/)

---

## The Problem

Neural networks are notorious for **catastrophic forgetting** — teach a model something new and it overwrites what it already knew. Replay buffers help, but they're expensive, don't scale, and sometimes aren't even allowed (think GDPR). Architecture expansion works too, but the model keeps growing.

We asked a different question: *what if the model already has everything it needs, and we just need to figure out which parts matter?*

## Our Approach

**Shapley Neuron Valuation (SNV)** borrows a 70-year-old idea from cooperative game theory — the [Shapley value](https://en.wikipedia.org/wiki/Shapley_value) — and applies it to neurons. Instead of treating all neurons equally or making binary keep/discard decisions, SNV computes a *fair* importance score for every neuron by measuring its marginal contribution across all possible subsets of the network. The important neurons get frozen; the rest stay plastic for the next task.

No replay buffer. No extra parameters. No task labels at test time (in the Class-IL setting). Just a smarter way of deciding what to protect.

---

## Repository layout

```
SNV/                        the method
  snv_core.py               Shapley neuron valuation, the estimator, the masks
  snv_adaptive.py           SNV-A: task-local phases, routed Class-IL inference,
                            per-task adaptive mask size -- the reported SNV
snv_adaptive_run.py         runs SNV-A through the unchanged GTEP worker
snv_core.py                 import shim for SNV/snv_core.py

baselines/                  every method SNV is compared against
  __init__.py               the registry: build_method, ALL_METHODS, ...
  regularization.py         re-exports EWC, SI, LwF, SGD
  sparse.py                 re-exports WSN, SpaceNet, NISPA, PEC
  joint_audited.py          JointPrefix, the audited joint-training upper bound
  SGD/ Joint/               bounds (lower, upper)
  EWC/ SI/ LwF/             regularisation
  WSN/ PEC/ SpaceNet/ NISPA/    sparse / architecture
  UniCLUN/                  continual learning + machine unlearning
  MCL/                      Matryoshka Continual Learning

audited_gtep.py             the GTEP protocol: disjoint halves, search spaces,
                            one run, the cost ledger hooks, the run queue
campaign/
  run_campaign.py           the full campaign: R x 3-seed search, then the
                            winner three times on the clean half
  build_report.py           metric, cost and hyperparameter tables
                            (CSV, XLSX, Markdown, LaTeX, HTML)
  snv_adaptive_compare.py   SNV-A pilots against the audited trials

datasets.py                 CIFAR-100, CIFAR-20, TinyImageNet-200, ImageNet-1k
models.py                   ResNet-18 / ResNet-50 backbones, multi-head models
metrics.py                  ACC, BWT, FWT, PS (+ P, S, AF)
audit_cost.py               cost ledger: GPU-hours, peak memory, parameters,
                            GFLOPs, latency, energy
cost.py, inrun.py           lightweight trackers used by train.py
train.py                    standalone single-method trainer
cl_base.py                  the ContinualMethod interface every method implements
training_policy.py, utils.py, method_loader.py, wsn_helpers.py

hyperparameters/            the selected CIFAR-100 configurations
  cifar100_best.json/.md    each method's winner and the metrics it scored
  __init__.py               best_config(method, scenario)
  extract_cifar100_best.py  regenerates both files from a campaign directory

results/cifar100/           the reported results and the runs behind them
  metrics_summary.md/.csv   ACC, BWT, FWT, PS per method, mean +/- sd
  metrics_per_seed.csv      one row per run
  costs_summary.csv         GPU-hours, memory, parameters, energy
  accuracy_matrices.json    the task-by-task matrix of every run
  search_space.md/.json     the space and every trial's score
  blocks/                   the campaign blocks: all trials and the winner
  runs/                     result.json, command.json and log per clean run
results/import_cifar100.py, results/make_cifar100_tables.py

scripts/
  smoke_test.sh             every method, 2 tasks, 1 epoch, 64 samples
  run_cifar100_campaign.sh  the campaign that produced results/cifar100
  run_cifar100_selected.sh  rerun the selected configurations
  run_cifar100_snv_best.sh  SNV-A with its selected configuration
  export_cifar100_results.sh    rebuild the hyperparameter and result tables

tests/                      the estimator, SNV-A, the baselines, the metrics and
                            cost ledger, the four benchmarks, the hyperparameters
docs/                       PROTOCOL.md, METRICS.md, COSTS.md, BASELINES.md,
                            PACKAGE.md (the full technical README)
THIRD_PARTY.md              the upstream repositories the baselines follow
requirements.txt            Python 3.10, PyTorch 2.6 / CUDA 12.4
```

Full technical documentation: [docs/PACKAGE.md](docs/PACKAGE.md).

## Running

```bash
pip install -r requirements.txt

# every method on a tiny slice, to check the pipeline
bash scripts/smoke_test.sh
python -m pytest -q tests

# SNV-A with the selected CIFAR-100 configuration, clean half, 3 seeds
bash scripts/run_cifar100_snv_best.sh

# the whole campaign (every method, both scenarios, tuning then evaluation)
bash scripts/run_cifar100_campaign.sh
```

The selected hyperparameters are readable from code:

```python
from hyperparameters import best_config, entry
best_config('snv', 'class_il')        # SNV-A's Class-IL winner
entry('ewc', 'task_il')['clean_eval_D_E']
```

## Results

CIFAR-100, Class-IL, accuracy on the held-out evaluation half, mean ± sd over
seeds 42/43/44. Hyperparameters were selected on a disjoint half, so no
configuration ever saw the data it is reported on
([protocol](docs/PROTOCOL.md), [full tables](results/cifar100/metrics_summary.md)).

| Method | ACC | | Method | ACC |
|---|---:|---|---|---:|
| Joint training (upper bound) | 0.6543 ± 0.0240 | | SGD (lower bound) | 0.0872 ± 0.0070 |
| **SNV** | **0.3618 ± 0.0206** | | SI | 0.0856 ± 0.0049 |
| PEC | 0.3284 ± 0.0125 | | EWC | 0.0839 ± 0.0066 |
| UniCLUN | 0.3275 ± 0.0404 | | NISPA | 0.0824 ± 0.0054 |
| MCL | 0.2722 ± 0.0255 | | SpaceNet | 0.0677 ± 0.0043 |
| LwF | 0.1006 ± 0.0179 | | | |

SNV is the strongest method here apart from the joint-training upper bound. In
Task-IL it reaches 0.8175 ± 0.0157, behind joint training and WSN, which is
given the task identity at test time.

The selected configurations, the search space and every trial's score are in
[hyperparameters/cifar100_best.md](hyperparameters/cifar100_best.md) and
[results/cifar100/search_space.md](results/cifar100/search_space.md); the runs
themselves, with their logs and cost measurements, are in
[results/cifar100/](results/cifar100/).

---

## Citation

```bibtex
@inproceedings{snv2026icml,
  title     = {Shapley Neuron Values for Continual Learning: Which Neurons Matter Most?},
  author    = {[Author names]},
  booktitle = {Proceedings of the International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```


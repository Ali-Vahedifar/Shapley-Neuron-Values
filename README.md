# Shapley Neuron Valuation for Continual Learning
### *Which Neurons Matter Most?*

Reference implementation of **SNV**, from *"Shapley Neuron Values for Continual
Learning: Which Neurons Matter Most?"* (ICML 2026) — Mohammad Ali Vahedifar,
Abhisek Ray, Qi Zhang; DIGIT and Department of Electrical and Computer
Engineering, Aarhus University.

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-yellow.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## The idea

Neural networks are heavily over-parameterised. Rather than storing past data or
growing the architecture, SNV asks which neurons a task actually depends on and
freezes those, leaving the rest plastic for what comes next.

"Which neurons matter" is answered with the **Shapley value** from cooperative
game theory. Neurons are the players, model accuracy is the payoff, and a
neuron's value is its average marginal contribution over subsets of the network.
The Shapley value is the unique valuation satisfying Efficiency, Null
contribution, Symmetry and Linearity — so the ranking is principled rather than
heuristic, and it is **real-valued**, which is what lets it discriminate under a
tight capacity budget where a binary keep/discard score cannot.

No replay buffer. No architecture growth. No task identity at test time in the
class-incremental setting.

---

## How it works

1. **Train** the current task, with the gradients of frozen neurons masked out.
2. **Compute mean activations** `μ_i` on the task's validation split.
3. **Estimate Shapley values** by truncated Monte-Carlo sampling with a
   multi-armed-bandit stopping rule. Masking a neuron replaces its output with
   `μ_i` — not zero — so the signal statistics reaching later layers are
   preserved.
4. **Select** the top `k = ⌊c·N⌋` neurons as `S_t`.
5. **Freeze** them: `B_t = B_{t-1} ∪ S_t`.
6. **Repeat.**

A neuron is a convolutional filter (or a hidden unit of a non-classifier linear
layer), so `N = Σ_l C_l` — 4,800 for ResNet-18. Freezing a neuron covers its
weights, its bias, and the affine parameters *and running statistics* of the
BatchNorm that scales it: a filter whose BN keeps adapting does not compute a
fixed function, and forgetting is not actually prevented.

Per Figure 1, a neuron may enter the top-`r`% for several tasks, so the top-`k`
is taken over all `N` neurons — already-frozen neurons stay eligible, and
`reuse_stats()` reports how much of each `S_t` was already held.

---

## Installation

```bash
pip install -r requirements.txt
```

Python 3.8+, PyTorch ≥ 1.12, torchvision, numpy, scipy, matplotlib, tqdm.

CIFAR-100, TinyImageNet and MNIST download on first use. **ImageNet-1k must be
supplied manually** at `<data_root>/imagenet/{train,val}` as class-per-directory
`ImageFolder` trees; it cannot be fetched automatically.

---

## Quick start

```python
import torch
from snv_core import SNVContinualLearner
from models import create_model
from datasets import ContinualLearningBenchmark

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

benchmark = ContinualLearningBenchmark('cifar100', num_tasks=10, scenario='class_il')
model = create_model('cifar100', benchmark.classes_per_task, num_tasks=10)

learner = SNVContinualLearner(model, device, sparsity_ratio=0.1, scenario='class_il')

for task_id in range(10):
    train_loader, val_loader, test_loader = benchmark.get_task_data(task_id)
    learner.train_task(task_id, train_loader, val_loader)
```

Command line:

```bash
python train.py --method snv --dataset imagenet1k --num_tasks 50 \
                --scenario class_il --sparsity 0.02 --num_runs 10 --verbose
```

| Argument | Default | Description |
|---|---|---|
| `--method` | `snv` | any method in the registry (see below) |
| `--dataset` | `cifar100` | `pmnist`, `cifar100`, `tinyimagenet`, `imagenet1k` |
| `--num_tasks` | `10` | 10, 20 or 50 (50 on ImageNet-1k) |
| `--scenario` | `class_il` | `class_il` or `task_il` |
| `--sparsity` | `0.1` | capacity budget *c* |
| `--truncation` | `0.1` | truncation threshold *τ* |
| `--confidence` | `0.95` | bandit confidence *α* |
| `--max_permutations` | `200` | safety cap on EstimateSNV |
| `--shapley_eval_batches` | `8` | validation batches per `V(S)` |
| `--epochs` / `--batch_size` | per dataset | 200/64, 100/128 for ImageNet-1k, 20/10 for PMNIST |
| `--num_runs` | `10` | independent runs for mean ± std |

---

## Methods

| Family | Methods |
|---|---|
| Buffer-free | `snv`, `nfl`, `nfl+`, `dcnet`, `nispa`, `pec`, `spacenet`, `wsn`, `lwf`, `si`, `ewc` |
| Memory-based | `icarl`, `derpp`, `dytox` |
| Bounds | `sgd` (lower), `joint` (upper) |

`wsn` needs the task identity to select its subnetwork at test time, so it has
no class-incremental result — `train.py` refuses `--method wsn --scenario
class_il` rather than reporting a meaningless number. Buffer sizes default to
2,000 exemplars (CIFAR-100 / TinyImageNet, CIL), 20,000 (ImageNet-1k, CIL) and
200 (TIL).

Baselines are re-implementations on the shared backbone and training loop, so a
comparison isolates the algorithm rather than the surrounding infrastructure.
Where an original relies on machinery this repo does not carry, the class
docstring says so — see in particular `DyTox`, `NISPA` and `SpaceNet`.

---

## Reported results

ImageNet-1k, ResNet-18, averaged over 10 runs (paper Table 1):

| | CIL 10 | CIL 20 | CIL 50 | TIL 10 | TIL 20 | TIL 50 |
|---|---|---|---|---|---|---|
| **SNV** | **41.30** | **34.20** | **25.60** | **57.82** | **50.45** | **40.18** |
| NFL+ | 38.42 | 31.50 | 22.40 | 51.36 | 45.80 | 37.20 |
| DCNet | 37.80 | 30.10 | 20.80 | 50.15 | 44.30 | 35.80 |
| WSN | — | — | — | 48.73 | 43.20 | 35.40 |
| DyTox *(20k buffer)* | 40.15 | 33.20 | 24.50 | 59.40 | 52.10 | 42.80 |

SNV's BWT is 0.0 under TIL and −0.05 under CIL; it exceeds every buffer-free
baseline and, under CIL, DyTox as well — despite storing nothing.

Capacity-budget sweep, TIL, 10 tasks (paper Table 3):

| *c* | CIFAR-100 SNV | CIFAR-100 WSN | TinyImageNet SNV | TinyImageNet WSN |
|---|---|---|---|---|
| 0.03 | 71.74 | 59.65 | 69.89 | 60.72 |
| 0.05 | 74.52 | 60.19 | 73.24 | 63.22 |
| 0.10 | 76.19 | 61.22 | 74.73 | 61.96 |
| 0.30 | 77.89 | 63.15 | 74.38 | 62.92 |
| 0.50 | 79.76 | 64.00 | 74.82 | 61.06 |

The gap over WSN widens as the budget tightens, which is the argument for a
real-valued valuation: at `c = 0.03` the relative ordering among candidates is
what decides the subnetwork, and a binary score has no ordering to give.

---

## Reproducing

```bash
bash run_experiments.sh            # everything
bash run_experiments.sh hparams    # GTEP phase 1 only
bash run_experiments.sh imagenet   # Table 1
bash run_experiments.sh sparsity   # Table 3
bash run_experiments.sh pruning    # Fig. 4
```

**Hyperparameters** follow the two-phase protocol: `hparam_search.py` grid-searches
using *only* the first task's validation split, and the winner is then frozen for
every later task. Run it before `train.py`.

**Pruning analysis** (Fig. 4): `pruning.py` prunes globally by magnitude at
increasing rates and re-measures accuracy. A *sharp* cliff is the good outcome —
it means the parameters carry information; tolerance to deep pruning exposes
redundancy. The reported critical percentage is the first rate at which accuracy
falls below half its unpruned value.

**Cost.** EstimateSNV dominates the runtime and it is worth being concrete about
why. A permutation evaluates `V(S)` once per *active* neuron, and every neuron is
active at the start, so the first permutation costs `N = 4,800` subset
evaluations on ResNet-18. Each evaluation is a forward pass over
`--shapley_eval_batches` validation batches, which is the main lever on
wall-clock.

The bandit then retires neurons whose interval no longer straddles the top-`k`
boundary, but that only bites once the per-marginal variance has come down — with
the variance typical of raw accuracy deltas, nearly every neuron stays active for
the first several permutations. Measured on this implementation with `k = ⌊cN⌋`:

| per-marginal variance | `c = 0.03` | `c = 0.1` | `c = 0.5` |
|---|---|---|---|
| 10⁻² | 99.5 % active | 99.9 % | 100 % |
| 10⁻³ | 30.6 % | 55.5 % | 85.4 % |
| 10⁻⁴ | 5.8 % | 15.6 % | 35.4 % |
| 10⁻⁵ | 1.9 % | 4.9 % | 11.9 % |

So the saving is real but arrives late, and it arrives soonest at tight budgets —
convenient, since tight `c` is where SNV's advantage lies. Budget accordingly
rather than expecting a small constant factor over a freezing baseline.
`--max_permutations` caps the loop, and a run stopped by the cap is reported with
`converged=False` rather than presented as confidence-certified.

---

## Evaluation metrics

With `A[i, j]` the accuracy on task *j* after training task *i*:

| Metric | Definition |
|---|---|
| ACC | `(1/T) Σ_t A[T, t]` |
| BWT | `(1/(T-1)) Σ_{t<T} (A[T,t] − A[t,t])` |
| PS | `2PS/(P+S)`, with `P = (1/(T-1)) Σ_{t≥2} (A[t,t] − A[t-1,t])/(1 − A[t-1,t])` and `S = 1 − (1/(T-1)) Σ_{t<T} (A[t,t] − A[T,t])` |
| FWT | `(1/(T-1)) Σ_{t≥2} (A[t-1,t] − b_t)`, `b` = random-init accuracy (RAC) |
| AF | `(1/(T-1)) Σ_{t<T} (max_{k≥t} A[k,t] − A[T,t])` |
| CAP | `|∪_t S_t| / N` |

PS needs the first superdiagonal `A[t-1, t]` — the accuracy on task *t* before it
is trained. `train.py` records it after each task, creating task *t+1*'s head
first so the measurement is defined. Note `S = 1 + BWT`, so zero forgetting gives
`S = 1` and `PS = 2P/(P+1)`, which is below 1 whenever `P < 1`: zero BWT and
`PS < 1` are consistent, and PS stays bounded by plasticity.

---

## Experimental setup

| | PMNIST | CIFAR-100 | TinyImageNet | ImageNet-1k |
|---|---|---|---|---|
| Architecture | 4-layer MLP (200) | ResNet-18 | ResNet-18 | ResNet-18 |
| Input | 784 | 32×32×3 | 64×64×3 | 224×224×3 |
| Tasks | 10 | 10 / 20 | 10 / 20 | 10 / 20 / 50 |
| Batch size | 10 | 64 | 64 | 128 |
| Max epochs | 20 | 200 | 200 | 100 |
| Optimiser | Adam | Adam | Adam | Adam |
| Init | He | He | He | He |
| Split | 70 / 10 / 20 | 70 / 10 / 20 | 70 / 10 / 20 | 70 / 10 / 20 |
| Runs | 10 | 10 | 10 | 10 |

Early stopping on validation loss throughout. Splits are taken per task over the
pooled native train and test partitions — the native partitions are not in a
70/10/20 ratio, so slicing the train set alone cannot produce them.
Hardware: a single NVIDIA A6000.

---

## Project structure

```
snv_continual_learning/
├── snv_core.py                  # SNV: masking, Shapley estimation, bandit, freezing
├── cl_base.py                   # shared method base + reservoir buffer
├── baselines.py                 # method registry
├── baselines_regularization.py  # SGD, EWC, SI, LwF
├── baselines_memory.py          # iCaRL, DER++, DyTox
├── baselines_sparse.py          # WSN, NFL, NFL+, SpaceNet, NISPA, DCNet, PEC
├── models.py                    # MLP / ResNet-18 backbones, multi-head wrapper
├── datasets.py                  # benchmarks, 70/10/20 splits, ImageNet-1k
├── metrics.py                   # ACC, BWT, PS, FWT, AF, CAP
├── train.py                     # experiment driver
├── pruning.py                   # Fig. 4
├── hparam_search.py             # GTEP phase 1
├── utils.py                     # plots and tables
├── test_snv.py                  # test suite
└── run_experiments.sh
```

---

## Tests

```bash
python test_snv.py
```

42 tests covering the neuron definition and conv↔norm pairing, that the freezing
mask blocks exactly the frozen weights (BatchNorm statistics included) and that a
learned task survives a later one, the Shapley axioms — Efficiency, Null
contribution and Symmetry — checked against the exact closed form from Eq. (5) on
a model small enough to enumerate, that the Monte-Carlo estimator converges to
it, the bandit's behaviour around the top-`k` boundary, `|S_t| = ⌊c·N⌋` with reuse
permitted, every metric against worked examples, the 70/10/20 split, head
routing, and that every registered method trains a task.

---

## Citation

```bibtex
@inproceedings{vahedifar2026snv,
  title     = {Shapley Neuron Values for Continual Learning: Which Neurons Matter Most?},
  author    = {Vahedifar, Mohammad Ali and Ray, Abhisek and Zhang, Qi},
  booktitle = {Proceedings of the International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

## License

MIT.

## Acknowledgments

Builds on cooperative game theory (Shapley, 1953) and the Lottery Ticket
Hypothesis (Frankle & Carbin, 2019).

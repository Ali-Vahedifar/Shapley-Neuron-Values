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


<p align="center">
  <img src="figures/snv_conceptual.png" alt="SNV conceptual overview — neurons are valued using Shapley values, important ones are frozen, and remaining capacity stays plastic for future tasks." width="85%"/>
</p>
<p align="center"><em>
  <b>Figure 1.</b> SNV overview. After training on each task, Shapley values quantify every neuron's contribution. The most important neurons (dark) are frozen to lock in knowledge; the rest (light) remain available for future tasks. The result is zero forgetting by construction.
</em></p>


---

## Why Shapley Values?

There are simpler proxies for neuron importance — magnitude pruning, Fisher information, gradient norms. We tried them. The difference shows up when capacity gets tight:

| Budget *c* | SNV (ACC%) | NFL+ / Fisher (ACC%) | Gap |
|:---:|:---:|:---:|:---:|
| 0.10 | 76.19 | 70.68 | +5.51 |
| 0.05 | 74.52 | — | — |
| 0.03 | 71.74 | — | — |

*CIFAR-100, 10-task Task-IL, ResNet-18*

At generous budgets, any reasonable importance measure does okay. As the budget shrinks, the gap between Shapley-based selection and Fisher-based selection **widens monotonically** — Shapley finds the neurons that actually matter, not just the ones that happen to have large gradients right now.

---

## Key Results

### Task-IL (10 tasks, task identity provided at test time)

| Dataset | SNV *c*=0.1 | NFL+ | WSN *c*=0.1 | DyTox (5120) |
|---|:---:|:---:|:---:|:---:|
| Permuted MNIST | 97.45 (BWT=0.0) | 93.12 | 91.11 | 99.52 |
| CIFAR-100 | 76.19 (BWT=0.0) | 70.68 | 61.22 | 81.63 |
| TinyImageNet | 74.73 (BWT=0.0) | 58.21 | 61.96 | 75.84 |

SNV closes to within ~1–2% of the *best memory-based method* (DyTox with a 5,120-sample buffer) while achieving **perfect zero forgetting** and storing **zero exemplars**.

### Class-IL (the hard setting — no task identity at test time)

| Dataset | SNV | NFL+ | PEC | DyTox |
|---|:---:|:---:|:---:|:---:|
| CIFAR-100 (10 tasks) | **54.70** | 53.70 | 29.40 | 57.40 |
| CIFAR-100 (20 tasks) | **44.85** | 44.03 | 24.11 | 47.07 |
| TinyImageNet (10 tasks) | **45.70** | 44.70 | 19.40 | 49.40 |
| TinyImageNet (20 tasks) | **37.47** | 36.65 | 15.91 | 40.51 |

**Bold** = best among memory-free methods. SNV leads every memory-free baseline on CIFAR-100 and TinyImageNet by a consistent margin, and approaches the memory-based DyTox despite using no replay buffer at all.

### Parameter Efficiency

Counter-intuitively, *sharp accuracy drops under weight pruning are a good sign* — they mean the method is actually using its parameters. SNV and NFL+ both show steep cliffs, while methods like PEC tolerate 80% pruning with barely a dip, revealing massive redundancy.

---

## How It Works (in 30 seconds)

1. **Train** on the current task, with frozen neurons masked out of the gradient
2. **Compute mean activations** for each neuron on validation data
3. **Estimate Shapley values** via Monte Carlo sampling + truncation + multi-armed bandit acceleration
4. **Select top-*k*** neurons (*k = c · N*) as important for this task
5. **Freeze** those neurons by updating the cumulative mask: *B_t = B_{t-1} ∪ S_t*
6. **Repeat** for the next task

The Shapley estimation is the expensive part — but with truncation (skip evaluations when the model is basically non-functional) and MAB (stop sampling neurons whose importance is already confidently resolved), the overhead is only ~1.24× NFL+'s cost.

---

## Installation

```bash
git clone https://github.com/<your-username>/snv-continual-learning.git
cd snv-continual-learning

# Create environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, PyTorch ≥ 1.12, torchvision, numpy, scipy, matplotlib, tqdm.

---

## Quick Start

```python
import torch
from snv_core import SNVContinualLearner
from models import create_model
from datasets import ContinualLearningBenchmark

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set up a 10-task CIFAR-100 benchmark
benchmark = ContinualLearningBenchmark(
    dataset_name='cifar100',
    num_tasks=10,
    scenario='class_il'
)

# Create a ResNet-18 and wrap it in the SNV learner
model = create_model(dataset='cifar100', num_classes=100)
learner = SNVContinualLearner(
    model=model,
    device=device,
    sparsity_ratio=0.1,   # allocate 10% of neurons per task
    lr=0.001
)

# Train sequentially
for task_id in range(10):
    train_loader, val_loader, test_loader = benchmark.get_task_data(task_id)
    learner.train_task(task_id, train_loader, val_loader)
```

---

## Reproducing Paper Results

Run all experiments from the paper (10 runs each, multiple sparsity ratios, both Class-IL and Task-IL):

```bash
bash run_experiments.sh
```

Or run a specific configuration:

```bash
python train.py \
    --dataset cifar100 \
    --num_tasks 10 \
    --scenario class_il \
    --sparsity 0.1 \
    --epochs 200 \
    --num_runs 10 \
    --gpu 0
```

### Command-line arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `cifar100` | `pmnist`, `cifar100`, or `tinyimagenet` |
| `--num_tasks` | `10` | Number of sequential tasks (10 or 20) |
| `--scenario` | `class_il` | `class_il` (no task ID at test) or `task_il` |
| `--sparsity` | `0.1` | Fraction of neurons allocated per task (*c*) |
| `--epochs` | `200` | Max epochs per task (20 for PMNIST) |
| `--batch_size` | `64` | Batch size (10 for PMNIST) |
| `--num_runs` | `10` | Independent runs for mean ± std |
| `--gpu` | `0` | GPU device ID |

### Tests

```bash
python test_snv.py
```

Verifies model architectures, mask logic, metric computations, and dataset splits against paper specifications.

---

## Project Structure

```
snv-continual-learning/
├── snv_core.py        # Core algorithm: Shapley estimation, masking, MAB
├── models.py          # MLP (PMNIST) and ResNet-18 (CIFAR/TinyImageNet)
├── datasets.py        # Benchmarks with train/val/test splits
├── metrics.py         # ACC, BWT, FWT, PS, AF, Intransigence
├── utils.py           # Visualization (accuracy matrices, Shapley heatmaps, etc.)
├── train.py           # Main training loop with multi-run aggregation
├── test_snv.py        # Unit tests
├── run_experiments.sh # Reproduce all paper experiments
├── requirements.txt
├── figures/           # Paper figures and conceptual diagrams
└── README.md
```

---


## Citation

```bibtex
@inproceedings{snv2026icml,
  title     = {Shapley Neuron Values for Continual Learning: Which Neurons Matter Most?},
  author    = {Mohammad Ali Vahedifar, Abhisek Ray, Qi Zhang},
  booktitle = {Proceedings of the International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

---

## License

Released under the [MIT License](LICENSE).

## Acknowledgments

This work builds on ideas from cooperative game theory (Shapley, 1953) and the Lottery Ticket Hypothesis (Frankle & Carlin, 2019). We thank the reviewers for their constructive feedback during the review process.

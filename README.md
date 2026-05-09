# Shapley Neuron Valuation for Continual Learning

This repository contains the official implementation of **Shapley Neuron Valuation (SNV)**, a memory-free continual learning method that uses cooperative game theory to identify and protect important neurons.

> **Anonymous submission for ICML 2026**

## Abstract

Continual learning enables neural networks to learn tasks sequentially without forgetting previously acquired knowledge. However, catastrophic forgetting remains a fundamental challenge. We address this with **Shapley Neuron Valuation (SNV)**, a principled framework grounded in cooperative game theory that quantifies neuron importance. By selectively freezing important neurons while keeping others plastic, SNV enables **memory-free continual learning without architectural expansion**.

## Key Features

- **Memory-free**: No replay buffer required
- **No architecture expansion**: Fixed model capacity
- **Theoretically grounded**: Based on Shapley values from cooperative game theory
- **Zero backward transfer**: Achieves BWT = 0.0 by construction
- **Efficient**: Multi-armed bandit acceleration for Shapley value estimation

## Installation

```bash
# Clone the repository
git clone https://github.com/anonymous/snv-continual-learning.git
cd snv-continual-learning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
snv_continual_learning/
├── snv_core.py          # Core SNV algorithm implementation
├── models.py            # Model architectures (MLP, ResNet-18)
├── datasets.py          # Dataset loaders and benchmarks
├── metrics.py           # Evaluation metrics (ACC, BWT, FWT, PS, AF)
├── utils.py             # Visualization and analysis utilities
├── train.py             # Main training script
├── run_experiments.sh   # Script to run all experiments
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Usage

### Quick Start

```python
import torch
from snv_core import SNVContinualLearner
from models import create_model
from datasets import ContinualLearningBenchmark

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create benchmark
benchmark = ContinualLearningBenchmark(
    dataset_name='cifar100',
    num_tasks=10,
    scenario='class_il'
)

# Create model
model = create_model(dataset='cifar100', num_classes=100)

# Create SNV learner
learner = SNVContinualLearner(
    model=model,
    device=device,
    sparsity_ratio=0.1,  # c = 0.1
    lr=0.001
)

# Train on tasks
for task_id in range(10):
    train_loader, val_loader, test_loader = benchmark.get_task_data(task_id)
    learner.train_task(task_id, train_loader, val_loader)
```

### Running Experiments

```bash
# Run single experiment
python train.py \
    --dataset cifar100 \
    --num_tasks 10 \
    --scenario class_il \
    --sparsity 0.1 \
    --num_runs 10 \
    --gpu 0

# Run all experiments from the paper
bash run_experiments.sh
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `cifar100` | Dataset name (pmnist, cifar100, tinyimagenet) |
| `--num_tasks` | `10` | Number of tasks (10 or 20) |
| `--scenario` | `class_il` | Learning scenario (class_il or task_il) |
| `--sparsity` | `0.1` | Sparsity ratio c (neurons per task) |
| `--lr` | `0.001` | Learning rate |
| `--batch_size` | `64` | Batch size (10 for PMNIST) |
| `--epochs` | `200` | Maximum epochs (20 for PMNIST) |
| `--num_runs` | `10` | Number of runs for averaging |
| `--gpu` | `0` | GPU device ID |

## Datasets

### Permuted MNIST (PMNIST)
- 10 tasks with different pixel permutations
- 10 classes per task (all digits, different permutation)
- Input: 28×28 → 784 (flattened)

### CIFAR-100
- 100 classes divided into 10 or 20 tasks
- 10 or 5 classes per task respectively
- Input: 32×32×3

### TinyImageNet
- 200 classes divided into 10 or 20 tasks
- 20 or 10 classes per task respectively
- Input: 64×64×3

## Experimental Setup

| Setting | PMNIST | CIFAR-100 | TinyImageNet |
|---------|--------|-----------|--------------|
| Architecture | 4-layer MLP | ResNet-18 | ResNet-18 |
| Hidden dim | 200 | - | - |
| Batch size | 10 | 64 | 64 |
| Epochs | 20 | 200 | 200 |
| Optimizer | Adam | Adam | Adam |
| Learning rate | 0.001 | 0.001 | 0.001 |

## Evaluation Metrics

- **ACC**: Average Accuracy across all tasks
- **BWT**: Backward Transfer (negative = forgetting)
- **FWT**: Forward Transfer
- **PS**: Plasticity-Stability ratio
- **AF**: Average Forgetting
- **CAP**: Capacity used (% of neurons frozen)

## Algorithm

### Shapley Neuron Value Estimation

The Shapley value for neuron $i$ is:

$$\phi_i = \sum_{S \subseteq M \setminus \{i\}} \frac{|S|!(|M|-|S|-1)!}{|M|!}[V(S \cup \{i\}) - V(S)]$$

Key optimizations:
1. **Monte Carlo estimation**: Sample permutations instead of exact computation
2. **Truncation**: Skip computations when subset performance is below threshold
3. **Multi-armed bandit**: Focus sampling on neurons near the top-k boundary

### Continual Learning Framework

1. Train on task $t$ with gradient masking (frozen neurons blocked)
2. Compute mean activations for Shapley value estimation
3. Estimate Shapley values via Monte Carlo + MAB
4. Select top-$k$ neurons ($k = c \cdot N$) as important
5. Update cumulative mask: $B_t = B_{t-1} \cup S_t$
6. Repeat for next task

## Results Summary

### Task-IL (10 tasks, c=0.1)

| Dataset | ACC (%) | BWT |
|---------|---------|-----|
| PMNIST | 97.45 | 0.0 |
| CIFAR-100 | 76.19 | 0.0 |
| TinyImageNet | 74.73 | 0.0 |

### Class-IL (10 tasks)

| Dataset | ACC (%) | BWT | PS |
|---------|---------|-----|-----|
| PMNIST | 93.45 | -0.06 | 0.96 |
| CIFAR-100 | 54.70 | -0.04 | 0.69 |
| TinyImageNet | 45.70 | -0.05 | 0.62 |

## Citation

```bibtex
@inproceedings{anonymous2026shapley,
  title={Shapley Neuron Values for Continual Learning: Which Neurons Matter Most?},
  author={Anonymous},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```

## License

This project is released under the MIT License.

## Acknowledgments

This work builds upon insights from cooperative game theory (Shapley, 1953) and the Lottery Ticket Hypothesis (Frankle & Carlin, 2019).

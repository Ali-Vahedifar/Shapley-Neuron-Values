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
  <img src="Shapley Conceptual.png" alt="SNV conceptual overview — neurons are valued using Shapley values, important ones are frozen, and remaining capacity stays plastic for future tasks." width="85%"/>
</p>
<p align="center"><em>
  <b>Figure 1.</b> SNV overview. After training on each task, Shapley values quantify every neuron's contribution. The most important neurons in each task are frozen to lock in knowledge; the rest remain available for future tasks. The result is zero forgetting by construction.
</em></p>


## How It Works (in 30 seconds)

1. **Train** on the current task, with frozen neurons masked out of the gradient
2. **Compute mean activations** for each neuron on validation data
3. **Estimate Shapley values** via Monte Carlo sampling + truncation + multi-armed bandit acceleration
4. **Select top-*k*** neurons (*k = c · N*) as important for this task
5. **Freeze** those neurons by updating the cumulative mask: *B_t = B_{t-1} ∪ S_t*
6. **Repeat** for the next task

The Shapley estimation is the expensive part — but with truncation (skip evaluations when the model is basically non-functional) and MAB (stop sampling neurons whose importance is already confidently resolved).


```

**Requirements:** Python 3.8+, PyTorch ≥ 1.12, torchvision, numpy, scipy, matplotlib, tqdm.

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


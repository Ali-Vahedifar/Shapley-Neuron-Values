"""
Continual-learning evaluation metrics (paper Fig. 6, Appendix "Evaluation Metrices").

All metrics read from the accuracy matrix A, where A[i, j] is the accuracy on
task j after training on task i.  Unlike a lower-triangular-only matrix, the
first superdiagonal A[t-1, t] is required: the plasticity term normalises each
task's learning gain by the head-room above its zero-shot accuracy.

    ACC_T = (1/T) sum_{t=1..T} A[T, t]

    BWT_T = (1/(T-1)) sum_{t=1..T-1} (A[T, t] - A[t, t])

    PS_T  = 2 P S / (P + S)

            P = (1/(T-1)) sum_{t=2..T}   (A[t, t] - A[t-1, t]) / (1 - A[t-1, t])
            S = 1 - (1/(T-1)) sum_{t=1..T-1} (A[t, t] - A[T, t])

    FWT_T = (1/(T-1)) sum_{t=2..T} (A[t-1, t] - b_t)     b = random-init baseline

Indices above are the paper's 1-based ones; the code is 0-based.
Note S = 1 + BWT, so a method with zero forgetting has S = 1 and PS = 2P/(P+1),
which is strictly below 1 whenever P < 1.  Zero BWT and PS < 1 are therefore
consistent, and PS is bounded by plasticity rather than pinned to it.
"""

from typing import Dict, Optional, Sequence

import numpy as np


class ContinualLearningMetrics:
    """Accumulates A and computes ACC / BWT / PS / FWT / AF / Intransigence."""

    def __init__(self, num_tasks: int):
        self.num_tasks = num_tasks
        self.accuracy_matrix = np.full((num_tasks, num_tasks), np.nan)
        self.random_baseline: Optional[np.ndarray] = None

    # -- recording ---------------------------------------------------------- #
    def update(self, current_task: int, task_accuracies: Sequence[float]) -> None:
        """Record row ``current_task``.

        ``task_accuracies`` may cover tasks 0..current_task (the usual case) or
        0..current_task+1, in which case the trailing entry fills the
        superdiagonal slot A[t, t+1] that the plasticity term needs.
        """
        for j, acc in enumerate(task_accuracies):
            if j < self.num_tasks:
                self.accuracy_matrix[current_task, j] = acc

    def record_zero_shot(self, after_task: int, next_task: int, accuracy: float) -> None:
        """Store A[after_task, next_task] -- task next_task before it is trained."""
        self.accuracy_matrix[after_task, next_task] = accuracy

    def set_random_baseline(self, baseline: Sequence[float]) -> None:
        """b_t: accuracy of the randomly initialised model on each task (RAC)."""
        self.random_baseline = np.asarray(baseline, dtype=float)

    # -- metrics ------------------------------------------------------------ #
    def get_average_accuracy(self, after_task: Optional[int] = None) -> float:
        """ACC_T = (1/T) sum_t A[T, t]."""
        row = self.num_tasks - 1 if after_task is None else after_task
        return float(np.nanmean(self.accuracy_matrix[row, :row + 1]))

    def get_backward_transfer(self) -> float:
        """BWT_T = (1/(T-1)) sum_{t<T} (A[T, t] - A[t, t])."""
        T = self.num_tasks
        if T < 2:
            return 0.0
        final = self.accuracy_matrix[T - 1, :T - 1]
        learned = np.array([self.accuracy_matrix[t, t] for t in range(T - 1)])
        return float(np.nanmean(final - learned))

    def get_plasticity(self) -> float:
        """P = (1/(T-1)) sum_{t=2..T} (A[t,t] - A[t-1,t]) / (1 - A[t-1,t])."""
        T = self.num_tasks
        if T < 2:
            return float('nan')
        terms = []
        for t in range(1, T):
            zero_shot = self.accuracy_matrix[t - 1, t]
            learned = self.accuracy_matrix[t, t]
            if np.isnan(zero_shot) or np.isnan(learned):
                continue
            headroom = 1.0 - zero_shot
            if headroom <= 1e-12:
                terms.append(1.0)       # nothing left to learn; treat as fully plastic
            else:
                terms.append((learned - zero_shot) / headroom)
        return float(np.mean(terms)) if terms else float('nan')

    def get_stability(self) -> float:
        """S = 1 - (1/(T-1)) sum_{t<T} (A[t,t] - A[T,t]).  Equivalently 1 + BWT."""
        T = self.num_tasks
        if T < 2:
            return float('nan')
        drops = [self.accuracy_matrix[t, t] - self.accuracy_matrix[T - 1, t]
                 for t in range(T - 1)]
        return float(1.0 - np.nanmean(drops))

    def get_plasticity_stability_ratio(self) -> float:
        """PS_T = 2 P S / (P + S), the harmonic mean of plasticity and stability.

        P and S are floored at zero first.  A harmonic mean is only meaningful
        for non-negative operands: a method that ends a task *worse* on it than
        it started gives P < 0, and 2PS/(P+S) would then flip sign or blow up
        near P = -S rather than degrading.  Flooring reports such a method as
        zero plasticity, which is what it demonstrated.  The raw values remain
        available through ``get_plasticity`` and ``get_stability``.
        """
        p, s = self.get_plasticity(), self.get_stability()
        if np.isnan(p) or np.isnan(s):
            return float('nan')
        p, s = max(p, 0.0), max(s, 0.0)
        if p + s < 1e-12:
            return 0.0
        return float(2.0 * p * s / (p + s))

    def get_forward_transfer(self) -> float:
        """FWT_T = (1/(T-1)) sum_{t=2..T} (A[t-1, t] - b_t), b = RAC."""
        T = self.num_tasks
        if T < 2:
            return float('nan')
        if self.random_baseline is None:
            return float('nan')
        terms = []
        for t in range(1, T):
            zero_shot = self.accuracy_matrix[t - 1, t]
            if not np.isnan(zero_shot):
                terms.append(zero_shot - self.random_baseline[t])
        return float(np.mean(terms)) if terms else float('nan')

    def get_average_forgetting(self) -> float:
        """AF = (1/(T-1)) sum_{t<T} max_{k>=t} (A[k, t]) - A[T, t]."""
        T = self.num_tasks
        if T < 2:
            return 0.0
        vals = []
        for t in range(T - 1):
            col = self.accuracy_matrix[t:, t]
            if np.all(np.isnan(col)):
                continue
            vals.append(np.nanmax(col) - self.accuracy_matrix[T - 1, t])
        return float(np.mean(vals)) if vals else 0.0

    def get_intransigence(self, joint_accuracy: Optional[Sequence[float]] = None) -> float:
        """I = (1/T) sum_t (A*[t] - A[t, t]), A* from joint training."""
        if joint_accuracy is None:
            return float('nan')
        diag = np.array([self.accuracy_matrix[t, t] for t in range(self.num_tasks)])
        return float(np.nanmean(np.asarray(joint_accuracy, dtype=float) - diag))

    # -- reporting ---------------------------------------------------------- #
    def get_all_metrics(self, joint_accuracy: Optional[Sequence[float]] = None) -> Dict[str, float]:
        return {
            'ACC': self.get_average_accuracy(),
            'BWT': self.get_backward_transfer(),
            'PS': self.get_plasticity_stability_ratio(),
            'P': self.get_plasticity(),
            'S': self.get_stability(),
            'FWT': self.get_forward_transfer(),
            'AF': self.get_average_forgetting(),
            'I': self.get_intransigence(joint_accuracy),
        }

    def get_accuracy_matrix(self) -> np.ndarray:
        return self.accuracy_matrix.copy()

    def print_summary(self, joint_accuracy: Optional[Sequence[float]] = None) -> None:
        m = self.get_all_metrics(joint_accuracy)
        print('\n' + '=' * 52)
        print('Continual Learning Metrics')
        print('=' * 52)
        print(f"  ACC  {m['ACC'] * 100:8.2f} %")
        print(f"  BWT  {m['BWT'] * 100:8.2f} %")
        print(f"  PS   {m['PS']:8.4f}    (P = {m['P']:.4f}, S = {m['S']:.4f})")
        print(f"  FWT  {m['FWT'] * 100:8.2f} %" if not np.isnan(m['FWT'])
              else '  FWT       n/a    (no random-init baseline recorded)')
        print(f"  AF   {m['AF'] * 100:8.2f} %")
        if joint_accuracy is not None:
            print(f"  I    {m['I'] * 100:8.2f} %")
        print('=' * 52)
        print('\nAccuracy matrix A[i, j]  (rows: after training task i)')
        for i in range(self.num_tasks):
            cells = ' '.join('  --  ' if np.isnan(v) else f'{v * 100:6.2f}'
                             for v in self.accuracy_matrix[i])
            print(f'  T{i + 1:<3d} {cells}')


# ------------------------------------------------------------------------- #
# helpers
# ------------------------------------------------------------------------- #
def compute_capacity(mask_manager) -> float:
    """CAP = |union_t S_t| / N * 100."""
    return mask_manager.get_capacity_used()


def compute_per_task_accuracies(model, test_loaders, device, current_task: int,
                                scenario: str = 'class_il',
                                include_next: bool = False) -> np.ndarray:
    """Accuracy on tasks 0..current_task, optionally also on task current_task+1.

    The extra column is what fills A[t, t+1] for the plasticity term.  Under TIL
    the head of the evaluated task is used; under CIL no task identity is passed.
    """
    import torch

    last = current_task + 1 if include_next and current_task + 1 < len(test_loaders) else current_task
    model.eval()
    accuracies = []
    with torch.no_grad():
        for task_id in range(last + 1):
            correct = total = 0
            for x, y in test_loaders[task_id]:
                x, y = x.to(device), y.to(device)
                head = task_id if scenario == 'task_il' else None
                try:
                    logits = model(x, head) if head is not None else model(x)
                except TypeError:
                    logits = model(x)
                correct += logits.argmax(1).eq(y).sum().item()
                total += y.numel()
            accuracies.append(correct / total if total else 0.0)
    return np.array(accuracies)

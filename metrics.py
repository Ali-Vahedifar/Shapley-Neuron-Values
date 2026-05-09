"""
Evaluation Metrics for Continual Learning.

Metrics implemented:
- Average Accuracy (ACC)
- Backward Transfer (BWT)
- Forward Transfer (FWT)
- Plasticity-Stability Ratio (PS)
- Intransigence (I)
- Average Forgetting (AF)

All metrics are computed from the accuracy matrix A where A[i,j] is the
accuracy on task j after training on task i.

Anonymous submission for ICML 2026.
"""

import numpy as np
from typing import Dict, Optional, Tuple


class ContinualLearningMetrics:
    """
    Computes continual learning evaluation metrics from accuracy matrix.
    
    The accuracy matrix A is of shape (T, T) where:
    - A[i, j] = accuracy on task j after training on tasks 0 through i
    - Diagonal A[i, i] = accuracy on task i immediately after training
    - Lower triangular = backward transfer (forgetting)
    - Upper triangular = forward transfer
    """
    
    def __init__(self, num_tasks: int):
        """
        Args:
            num_tasks: Total number of tasks
        """
        self.num_tasks = num_tasks
        self.accuracy_matrix = np.zeros((num_tasks, num_tasks))
        self.random_accuracy = None  # Baseline random accuracy
        
    def update(self, current_task: int, task_accuracies: np.ndarray):
        """
        Update accuracy matrix after training on a task.
        
        Args:
            current_task: Index of task just trained (0-indexed)
            task_accuracies: Array of accuracies on tasks 0 through current_task
        """
        for j, acc in enumerate(task_accuracies):
            self.accuracy_matrix[current_task, j] = acc
    
    def set_random_accuracy(self, random_acc: float):
        """Set random baseline accuracy for forward transfer computation."""
        self.random_accuracy = random_acc
    
    def get_average_accuracy(self, after_task: Optional[int] = None) -> float:
        """
        Compute Average Accuracy (ACC).
        
        ACC = (1/T) * Σ_{j=1}^{T} A[T, j]
        
        Average accuracy on all tasks after training on all tasks.
        
        Args:
            after_task: Task index to compute ACC after (default: last task)
            
        Returns:
            Average accuracy
        """
        if after_task is None:
            after_task = self.num_tasks - 1
        
        return np.mean(self.accuracy_matrix[after_task, :after_task + 1])
    
    def get_backward_transfer(self) -> float:
        """
        Compute Backward Transfer (BWT).
        
        BWT = (1/(T-1)) * Σ_{j=1}^{T-1} (A[T, j] - A[j, j])
        
        Measures how much learning new tasks affects performance on old tasks.
        - Negative BWT indicates forgetting
        - Zero BWT indicates no forgetting (ideal)
        - Positive BWT indicates backward knowledge transfer
        
        Returns:
            Backward transfer score
        """
        if self.num_tasks < 2:
            return 0.0
        
        bwt = 0.0
        for j in range(self.num_tasks - 1):
            final_acc = self.accuracy_matrix[self.num_tasks - 1, j]
            initial_acc = self.accuracy_matrix[j, j]
            bwt += final_acc - initial_acc
        
        return bwt / (self.num_tasks - 1)
    
    def get_forward_transfer(self) -> float:
        """
        Compute Forward Transfer (FWT).
        
        FWT = (1/(T-1)) * Σ_{j=2}^{T} (A[j-1, j] - RAC)
        
        Measures how much previous learning helps on new tasks.
        RAC is the random accuracy baseline.
        
        Returns:
            Forward transfer score
        """
        if self.num_tasks < 2:
            return 0.0
        
        if self.random_accuracy is None:
            # Default to 1/num_classes for each task
            self.random_accuracy = 0.1  # Assuming 10 classes per task
        
        fwt = 0.0
        for j in range(1, self.num_tasks):
            # Accuracy on task j before training on it (after training on j-1)
            pre_train_acc = self.accuracy_matrix[j - 1, j] if j > 0 else self.random_accuracy
            fwt += pre_train_acc - self.random_accuracy
        
        return fwt / (self.num_tasks - 1)
    
    def get_plasticity_stability_ratio(self) -> float:
        """
        Compute Plasticity-Stability Ratio (PS).
        
        PS = (Plasticity) / (Plasticity + Stability Loss)
        
        where:
        - Plasticity = average diagonal accuracy (learning capability)
        - Stability Loss = average forgetting
        
        Higher PS indicates better balance between learning and retention.
        
        Returns:
            Plasticity-Stability ratio
        """
        # Plasticity: average accuracy immediately after learning each task
        plasticity = np.mean(np.diag(self.accuracy_matrix))
        
        # Stability loss: average forgetting
        forgetting = -self.get_backward_transfer()  # Negate since BWT is negative for forgetting
        
        if plasticity + forgetting == 0:
            return 0.0
        
        return plasticity / (plasticity + max(forgetting, 0))
    
    def get_average_forgetting(self) -> float:
        """
        Compute Average Forgetting (AF).
        
        AF = (1/(T-1)) * Σ_{j=1}^{T-1} max_{t∈{j,...,T-1}} (A[t, j] - A[T, j])
        
        Maximum accuracy drop for each task.
        
        Returns:
            Average forgetting score
        """
        if self.num_tasks < 2:
            return 0.0
        
        forgetting = 0.0
        for j in range(self.num_tasks - 1):
            # Find maximum accuracy on task j over all subsequent training
            max_acc = np.max(self.accuracy_matrix[j:, j])
            final_acc = self.accuracy_matrix[self.num_tasks - 1, j]
            forgetting += max_acc - final_acc
        
        return forgetting / (self.num_tasks - 1)
    
    def get_intransigence(self, joint_accuracy: Optional[np.ndarray] = None) -> float:
        """
        Compute Intransigence (I).
        
        I = (1/T) * Σ_{j=1}^{T} (A*[j] - A[j, j])
        
        where A*[j] is the accuracy on task j when trained jointly on all data.
        Measures inability to learn new tasks.
        
        Args:
            joint_accuracy: Array of joint training accuracies (upper bound)
            
        Returns:
            Intransigence score
        """
        if joint_accuracy is None:
            return 0.0  # Cannot compute without joint training baseline
        
        intrans = 0.0
        for j in range(self.num_tasks):
            intrans += joint_accuracy[j] - self.accuracy_matrix[j, j]
        
        return intrans / self.num_tasks
    
    def get_all_metrics(
        self,
        joint_accuracy: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Args:
            joint_accuracy: Optional joint training accuracies for intransigence
            
        Returns:
            Dictionary of all metrics
        """
        return {
            'ACC': self.get_average_accuracy(),
            'BWT': self.get_backward_transfer(),
            'FWT': self.get_forward_transfer(),
            'PS': self.get_plasticity_stability_ratio(),
            'AF': self.get_average_forgetting(),
            'I': self.get_intransigence(joint_accuracy)
        }
    
    def get_accuracy_matrix(self) -> np.ndarray:
        """Return the full accuracy matrix."""
        return self.accuracy_matrix.copy()
    
    def print_summary(self, joint_accuracy: Optional[np.ndarray] = None):
        """Print a formatted summary of all metrics."""
        metrics = self.get_all_metrics(joint_accuracy)
        
        print("\n" + "=" * 50)
        print("Continual Learning Metrics Summary")
        print("=" * 50)
        print(f"Average Accuracy (ACC):        {metrics['ACC']*100:.2f}%")
        print(f"Backward Transfer (BWT):       {metrics['BWT']:.4f}")
        print(f"Forward Transfer (FWT):        {metrics['FWT']:.4f}")
        print(f"Plasticity-Stability (PS):     {metrics['PS']:.4f}")
        print(f"Average Forgetting (AF):       {metrics['AF']:.4f}")
        if joint_accuracy is not None:
            print(f"Intransigence (I):             {metrics['I']:.4f}")
        print("=" * 50)
        
        print("\nAccuracy Matrix (rows=after training, cols=task):")
        print("-" * 50)
        for i in range(self.num_tasks):
            row = " ".join([f"{self.accuracy_matrix[i,j]*100:6.2f}" 
                          for j in range(i + 1)])
            print(f"Task {i}: {row}")


def compute_capacity(
    mask_manager,
    total_neurons: int
) -> float:
    """
    Compute Capacity (CAP) metric.
    
    CAP = (|∪_{t=1}^{T} M_t|) / |θ_dense| × 100%
    
    Percentage of neurons used by all tasks combined.
    
    Args:
        mask_manager: NeuronMaskManager instance
        total_neurons: Total neurons in the dense model
        
    Returns:
        Capacity percentage
    """
    return mask_manager.get_capacity_used()


def compute_per_task_accuracies(
    model,
    test_loaders,
    device,
    current_task: int
) -> np.ndarray:
    """
    Compute accuracies on all tasks seen so far.
    
    Args:
        model: The neural network model
        test_loaders: List of test data loaders for each task
        device: Torch device
        current_task: Current task index (0-indexed)
        
    Returns:
        Array of accuracies for tasks 0 through current_task
    """
    import torch
    
    model.eval()
    accuracies = []
    
    with torch.no_grad():
        for task_id in range(current_task + 1):
            correct = 0
            total = 0
            
            for batch in test_loaders[task_id]:
                x, y = batch[0].to(device), batch[1].to(device)
                outputs = model(x)
                _, predicted = outputs.max(1)
                correct += predicted.eq(y).sum().item()
                total += y.size(0)
            
            acc = correct / total if total > 0 else 0.0
            accuracies.append(acc)
    
    return np.array(accuracies)

"""
Shapley Neuron Valuation (SNV) for Continual Learning
Core implementation of the SNV algorithm.

This implementation follows the methodology described in:
"Shapley Neuron Values for Continual Learning: Which Neurons Matter Most?"

Anonymous submission for ICML 2026.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from collections import defaultdict
import copy
from tqdm import tqdm


class NeuronMaskManager:
    """
    Manages neuron-level masks for continual learning.
    
    In SNV, a 'Neuron' is defined as a convolutional filter (kernel) for CNNs
    or a neuron in fully connected layers for MLPs.
    """
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.neuron_info = self._extract_neuron_info()
        self.num_neurons = sum(info['num_neurons'] for info in self.neuron_info)
        
        # Cumulative mask tracking which neurons are frozen
        self.cumulative_mask = torch.zeros(self.num_neurons, dtype=torch.bool, device=device)
        self.task_masks = {}  # Store mask for each task
        
    def _extract_neuron_info(self) -> List[Dict]:
        """Extract information about neurons (filters/units) in each layer."""
        neuron_info = []
        neuron_idx = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                num_neurons = module.out_channels
                neuron_info.append({
                    'name': name,
                    'module': module,
                    'type': 'conv',
                    'num_neurons': num_neurons,
                    'start_idx': neuron_idx,
                    'end_idx': neuron_idx + num_neurons
                })
                neuron_idx += num_neurons
            elif isinstance(module, nn.Linear) and 'fc' not in name and 'classifier' not in name:
                # Include hidden linear layers but not the final classifier
                num_neurons = module.out_features
                neuron_info.append({
                    'name': name,
                    'module': module,
                    'type': 'linear',
                    'num_neurons': num_neurons,
                    'start_idx': neuron_idx,
                    'end_idx': neuron_idx + num_neurons
                })
                neuron_idx += num_neurons
                
        return neuron_info
    
    def get_neuron_indices_for_layer(self, layer_name: str) -> Tuple[int, int]:
        """Get start and end indices for neurons in a specific layer."""
        for info in self.neuron_info:
            if info['name'] == layer_name:
                return info['start_idx'], info['end_idx']
        raise ValueError(f"Layer {layer_name} not found")
    
    def create_gradient_mask(self) -> Dict[str, torch.Tensor]:
        """
        Create parameter-level gradient mask from neuron-level cumulative mask.
        
        Returns a dictionary mapping parameter names to binary masks where:
        - 0: parameter belongs to a frozen neuron (gradient blocked)
        - 1: parameter is free to update
        """
        gradient_masks = {}
        
        for info in self.neuron_info:
            module = info['module']
            start_idx = info['start_idx']
            
            for param_name, param in module.named_parameters():
                full_name = f"{info['name']}.{param_name}"
                mask = torch.ones_like(param, dtype=torch.float32, device=self.device)
                
                if info['type'] == 'conv':
                    # For conv layers, weight shape is [out_channels, in_channels, H, W]
                    for i in range(info['num_neurons']):
                        if self.cumulative_mask[start_idx + i]:
                            if 'weight' in param_name:
                                mask[i] = 0
                            elif 'bias' in param_name:
                                mask[i] = 0
                                
                elif info['type'] == 'linear':
                    # For linear layers, weight shape is [out_features, in_features]
                    for i in range(info['num_neurons']):
                        if self.cumulative_mask[start_idx + i]:
                            if 'weight' in param_name:
                                mask[i] = 0
                            elif 'bias' in param_name:
                                mask[i] = 0
                
                gradient_masks[full_name] = mask
                
        return gradient_masks
    
    def update_cumulative_mask(self, task_id: int, task_mask: torch.Tensor):
        """
        Update cumulative mask with new task's important neurons.
        
        Args:
            task_id: Identifier for the current task
            task_mask: Binary mask indicating important neurons for this task
        """
        self.task_masks[task_id] = task_mask.clone()
        self.cumulative_mask = self.cumulative_mask | task_mask
        
    def get_available_neurons(self) -> torch.Tensor:
        """Get indices of neurons not yet frozen."""
        return ~self.cumulative_mask
    
    def get_capacity_used(self) -> float:
        """Calculate percentage of neurons currently frozen."""
        return self.cumulative_mask.sum().item() / self.num_neurons * 100


class MeanActivationComputer:
    """
    Computes and stores mean activations for neurons.
    
    When zeroing out a neuron, we replace its output with the mean response
    over validation data to preserve signal statistics for subsequent layers.
    """
    
    def __init__(self, model: nn.Module, neuron_info: List[Dict], device: torch.device):
        self.model = model
        self.neuron_info = neuron_info
        self.device = device
        self.mean_activations = {}
        self.hooks = []
        self.activation_sums = {}
        self.activation_counts = {}
        
    def _register_hooks(self):
        """Register forward hooks to capture activations."""
        self.hooks = []
        self.activation_sums = {}
        self.activation_counts = {}
        
        for info in self.neuron_info:
            name = info['name']
            self.activation_sums[name] = None
            self.activation_counts[name] = 0
            
            def hook_fn(module, input, output, name=name):
                if self.activation_sums[name] is None:
                    # Initialize with zeros matching output shape for channel dimension
                    if len(output.shape) == 4:  # Conv output [B, C, H, W]
                        self.activation_sums[name] = torch.zeros(
                            output.shape[1], device=self.device
                        )
                    else:  # Linear output [B, F]
                        self.activation_sums[name] = torch.zeros(
                            output.shape[1], device=self.device
                        )
                
                # Sum over batch and spatial dimensions
                if len(output.shape) == 4:
                    # Mean over spatial dimensions, sum over batch
                    self.activation_sums[name] += output.mean(dim=(2, 3)).sum(dim=0)
                    self.activation_counts[name] += output.shape[0]
                else:
                    self.activation_sums[name] += output.sum(dim=0)
                    self.activation_counts[name] += output.shape[0]
            
            hook = info['module'].register_forward_hook(hook_fn)
            self.hooks.append(hook)
    
    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def compute_mean_activations(self, dataloader: torch.utils.data.DataLoader):
        """
        Compute mean activations for all neurons over the validation set.
        
        Args:
            dataloader: Validation data loader
        """
        self._register_hooks()
        self.model.eval()
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0].to(self.device)
                else:
                    x = batch.to(self.device)
                _ = self.model(x)
        
        # Compute means
        for name in self.activation_sums:
            if self.activation_counts[name] > 0:
                self.mean_activations[name] = (
                    self.activation_sums[name] / self.activation_counts[name]
                )
        
        self._remove_hooks()
        
    def get_mean_activation(self, layer_name: str) -> torch.Tensor:
        """Get mean activation for a specific layer."""
        return self.mean_activations.get(layer_name, None)


class ShapleyNeuronEstimator:
    """
    Estimates Shapley Neuron Values using Monte Carlo sampling with
    multi-armed bandit acceleration.
    
    The Shapley value for neuron i is:
    φ_i = E_π[V(S_i^π ∪ {i}) - V(S_i^π)]
    
    where π is a random permutation and S_i^π is the set of neurons
    appearing before i in permutation π.
    """
    
    def __init__(
        self,
        model: nn.Module,
        neuron_info: List[Dict],
        mean_activations: Dict[str, torch.Tensor],
        device: torch.device,
        truncation_threshold: float = 0.1,
        confidence_level: float = 0.95
    ):
        self.model = model
        self.neuron_info = neuron_info
        self.mean_activations = mean_activations
        self.device = device
        self.truncation_threshold = truncation_threshold
        self.confidence_level = confidence_level
        
        # Z-score for confidence interval
        from scipy import stats
        self.z_alpha = stats.norm.ppf((1 + confidence_level) / 2)
        
        self.num_neurons = sum(info['num_neurons'] for info in neuron_info)
        
    def _create_neuron_mask_hooks(
        self, 
        active_neurons: torch.Tensor
    ) -> List[torch.utils.hooks.RemovableHandle]:
        """
        Create hooks that mask out inactive neurons by replacing with mean activations.
        
        As per paper Section 3: "This is done by replacing a filter's output with 
        its mean response over a set of validation data. This procedure blocks the 
        flow of information through that filter while preserving the average 
        statistics of the signal passed to subsequent layers."
        
        Args:
            active_neurons: Binary tensor indicating which neurons are active (in S)
                           Neurons NOT in active_neurons are replaced with mean activations
        """
        hooks = []
        neuron_idx = 0
        
        for info in self.neuron_info:
            layer_name = info['name']
            num_neurons = info['num_neurons']
            layer_active = active_neurons[neuron_idx:neuron_idx + num_neurons].clone()
            mean_act = self.mean_activations.get(layer_name, None)
            
            if mean_act is not None:
                # Clone mean_act to avoid issues with closures
                mean_act_clone = mean_act.clone()
                
                def hook_fn(module, input, output, layer_active=layer_active, mean_act=mean_act_clone):
                    # Create a new tensor to avoid in-place modification issues
                    modified_output = output.clone()
                    
                    # Replace inactive neuron outputs with mean activations
                    # Inactive neurons are those where layer_active[i] == False
                    if len(output.shape) == 4:  # Conv output [B, C, H, W]
                        for i in range(len(layer_active)):
                            if not layer_active[i]:
                                # Replace entire spatial output with mean value
                                modified_output[:, i, :, :] = mean_act[i]
                    else:  # Linear output [B, F]
                        for i in range(len(layer_active)):
                            if not layer_active[i]:
                                modified_output[:, i] = mean_act[i]
                    
                    return modified_output
                
                hook = info['module'].register_forward_hook(hook_fn)
                hooks.append(hook)
            
            neuron_idx += num_neurons
            
        return hooks
    
    def _evaluate_subset(
        self,
        active_neurons: torch.Tensor,
        dataloader: torch.utils.data.DataLoader,
        criterion: Callable = None
    ) -> float:
        """
        Evaluate model performance V(S) with only a subset S of neurons active.
        
        As per paper: "V(S) to denote the model's performance after all Neurons 
        in M minus S have been zeroed out. The model is not retrained after this 
        modification; all parameters remain fixed, and we directly evaluate 
        the test performance V(S)."
        
        Args:
            active_neurons: Binary tensor indicating active neurons (set S)
                           True = neuron is active, False = neuron is zeroed (mean-replaced)
            dataloader: Validation data loader
            criterion: Unused, kept for API compatibility
            
        Returns:
            Performance score V(S) - accuracy on validation set
        """
        hooks = self._create_neuron_mask_hooks(active_neurons)
        self.model.eval()
        
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    x, y = batch[0].to(self.device), batch[1].to(self.device)
                else:
                    raise ValueError("Expected (x, y) batch format")
                
                outputs = self.model(x)
                _, predicted = outputs.max(1)
                total_correct += predicted.eq(y).sum().item()
                total_samples += y.size(0)
        
        # Remove hooks after evaluation
        for hook in hooks:
            hook.remove()
        
        accuracy = total_correct / total_samples if total_samples > 0 else 0
        return accuracy
    
    def estimate_shapley_values(
        self,
        dataloader: torch.utils.data.DataLoader,
        max_iterations: int = 1000,
        early_stop_threshold: float = 0.01,
        verbose: bool = True
    ) -> torch.Tensor:
        """
        Estimate Shapley values using Monte Carlo sampling with MAB acceleration.
        
        Implements the three optimizations from the paper:
        
        i. Monte Carlo Estimation:
           φ_i = E_π[V(S_i^π ∪ {i}) - V(S_i^π)]
           where S_i^π is the set of elements appearing before i in permutation π
        
        ii. Truncation:
           Skip marginal computations when V(S_i^π) < τ (performance threshold)
        
        iii. Multi-Armed Bandit:
           Restrict sampling to neurons whose confidence intervals overlap 
           with the top-k threshold
        
        Args:
            dataloader: Validation data loader
            max_iterations: Maximum Monte Carlo iterations
            early_stop_threshold: Convergence threshold
            verbose: Show progress
            
        Returns:
            Tensor of estimated Shapley values for each neuron
        """
        # Initialize running estimates using Welford's algorithm
        shapley_estimates = torch.zeros(self.num_neurons, device=self.device)
        shapley_counts = torch.zeros(self.num_neurons, device=self.device)
        shapley_sq_sums = torch.zeros(self.num_neurons, device=self.device)  # For variance
        
        # Active set for MAB - neurons still being sampled
        active_set = torch.ones(self.num_neurons, dtype=torch.bool, device=self.device)
        
        # Compute k for top-k selection (will be set later based on sparsity)
        # For now, we estimate all Shapley values
        
        iterator = range(max_iterations)
        if verbose:
            iterator = tqdm(iterator, desc="Estimating Shapley Values")
        
        for iteration in iterator:
            # i. Monte Carlo: Sample random permutation π
            perm = torch.randperm(self.num_neurons, device=self.device)
            
            # S_i^π starts as empty set
            current_subset = torch.zeros(self.num_neurons, dtype=torch.bool, device=self.device)
            
            # V(∅) - performance with no neurons active (all zeroed)
            prev_performance = self._evaluate_subset(current_subset, dataloader)
            
            # Process each neuron in permutation order
            for j, neuron_idx in enumerate(perm):
                neuron_idx = neuron_idx.item()
                
                # iii. MAB: Skip if neuron not in active set
                if not active_set[neuron_idx]:
                    # Still add to subset but don't compute marginal
                    current_subset[neuron_idx] = True
                    continue
                
                # ii. Truncation: Skip if current subset performance too low
                # "When S_i^π is small, V(S_i^π) degrades toward zero"
                if prev_performance < self.truncation_threshold:
                    current_subset[neuron_idx] = True
                    continue
                
                # Compute V(S_i^π ∪ {i})
                current_subset[neuron_idx] = True
                current_performance = self._evaluate_subset(current_subset, dataloader)
                
                # Marginal contribution: V(S_i^π ∪ {i}) - V(S_i^π)
                marginal = current_performance - prev_performance
                
                # Update running estimates using Welford's online algorithm
                shapley_counts[neuron_idx] += 1
                n = shapley_counts[neuron_idx]
                
                delta = marginal - shapley_estimates[neuron_idx]
                shapley_estimates[neuron_idx] += delta / n
                delta2 = marginal - shapley_estimates[neuron_idx]
                shapley_sq_sums[neuron_idx] += delta * delta2
                
                # Update for next iteration
                prev_performance = current_performance
            
            # iii. MAB: Update active set based on confidence intervals
            if (iteration + 1) % 10 == 0 and iteration > 0:
                active_set = self._update_active_set_mab(
                    shapley_estimates,
                    shapley_counts,
                    shapley_sq_sums,
                    active_set
                )
                
                # Check for convergence - all neurons confidently classified
                if active_set.sum() == 0:
                    if verbose:
                        print(f"\nMAB converged at iteration {iteration + 1}")
                    break
                
                # Also check if confidence widths are small enough
                variances = shapley_sq_sums / torch.clamp(shapley_counts - 1, min=1)
                std_errors = torch.sqrt(variances / torch.clamp(shapley_counts, min=1))
                confidence_widths = self.z_alpha * std_errors
                
                if confidence_widths[shapley_counts > 1].max() < early_stop_threshold:
                    if verbose:
                        print(f"\nConverged at iteration {iteration + 1}")
                    break
        
        return shapley_estimates
    
    def _update_active_set_mab(
        self,
        estimates: torch.Tensor,
        counts: torch.Tensor,
        sq_sums: torch.Tensor,
        current_active: torch.Tensor
    ) -> torch.Tensor:
        """
        Update active set for Multi-Armed Bandit optimization.
        
        From paper: "we restrict sampling to those Neurons whose current 
        confidence intervals still overlap with the top-k largest estimated value"
        
        A neuron remains active if its confidence interval overlaps with the
        region where the top-k threshold might be.
        
        Args:
            estimates: Current Shapley value estimates
            counts: Number of samples per neuron
            sq_sums: Sum of squared deviations for variance
            current_active: Current active set
            
        Returns:
            Updated active set
        """
        # Compute confidence intervals
        # Variance using Welford's: var = sq_sums / (n - 1)
        variances = sq_sums / torch.clamp(counts - 1, min=1)
        std_errors = torch.sqrt(variances / torch.clamp(counts, min=1))
        confidence_widths = self.z_alpha * std_errors
        
        # Upper and lower bounds of confidence intervals
        upper_bounds = estimates + confidence_widths
        lower_bounds = estimates - confidence_widths
        
        # Find the approximate top-k threshold
        # We want neurons whose intervals might include the k-th largest value
        sorted_estimates, sorted_indices = torch.sort(estimates, descending=True)
        
        # Estimate where the top-k boundary might be
        # Use a heuristic: consider neurons whose upper bound exceeds the
        # lower bound of estimated top-k neurons, or whose lower bound
        # is below the upper bound of neurons just outside top-k
        
        # For robustness, keep neurons active if there's any uncertainty
        # about whether they're in top-k
        
        # Get the k-th largest estimate (approximate threshold)
        # k will be determined by sparsity ratio during selection
        # For now, use median as a proxy for identifying uncertain neurons
        median_estimate = torch.median(estimates)
        
        # A neuron stays active if:
        # 1. Its confidence interval is wide (high uncertainty)
        # 2. Its interval straddles the median (could go either way)
        
        new_active = torch.zeros_like(current_active)
        
        for i in range(len(estimates)):
            if counts[i] < 2:
                # Not enough samples, keep active
                new_active[i] = True
            elif confidence_widths[i] > 0.01:  # Still uncertain
                # Check if interval overlaps with decision boundary region
                if lower_bounds[i] <= median_estimate <= upper_bounds[i]:
                    new_active[i] = True
                elif upper_bounds[i] >= sorted_estimates[min(len(sorted_estimates)//4, len(sorted_estimates)-1)]:
                    # Could be in top quartile
                    new_active[i] = True
                elif lower_bounds[i] <= sorted_estimates[min(3*len(sorted_estimates)//4, len(sorted_estimates)-1)]:
                    # Could be in bottom quartile - still need to distinguish
                    new_active[i] = True
        
        return new_active
    
    def select_top_k_neurons(
        self,
        shapley_values: torch.Tensor,
        sparsity_ratio: float,
        available_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Select top-k neurons based on Shapley values.
        
        Args:
            shapley_values: Estimated Shapley values
            sparsity_ratio: Fraction of neurons to select (c in paper)
            available_mask: Binary mask of neurons still available
            
        Returns:
            Binary mask indicating selected neurons
        """
        k = int(sparsity_ratio * self.num_neurons)
        
        if available_mask is not None:
            # Only consider available neurons
            masked_values = shapley_values.clone()
            masked_values[~available_mask] = float('-inf')
        else:
            masked_values = shapley_values
        
        # Get top-k indices
        _, top_indices = torch.topk(masked_values, k)
        
        # Create binary mask
        mask = torch.zeros(self.num_neurons, dtype=torch.bool, device=self.device)
        mask[top_indices] = True
        
        return mask


class SNVContinualLearner:
    """
    Main class for Shapley Neuron Valuation continual learning.
    
    Implements Algorithm 2 from the paper.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        sparsity_ratio: float = 0.1,
        truncation_threshold: float = 0.1,
        confidence_level: float = 0.95,
        lr: float = 0.001
    ):
        """
        Args:
            model: Neural network model
            device: Torch device
            sparsity_ratio: Fraction of neurons to allocate per task (c)
            truncation_threshold: Performance threshold for truncation (τ)
            confidence_level: Confidence level for MAB (α)
            lr: Learning rate
        """
        self.model = model.to(device)
        self.device = device
        self.sparsity_ratio = sparsity_ratio
        self.truncation_threshold = truncation_threshold
        self.confidence_level = confidence_level
        self.lr = lr
        
        # Initialize mask manager
        self.mask_manager = NeuronMaskManager(model, device)
        
        # Task-specific heads storage
        self.task_heads = {}
        
        # Training history
        self.history = defaultdict(list)
        
    def _apply_gradient_mask(self, optimizer: torch.optim.Optimizer):
        """
        Apply gradient mask to implement the update rule from the paper:
        
        θ ← θ - η * (∂L/∂θ ⊙ M_{t-1})
        
        where M_{t-1} is 0 for frozen neurons and 1 otherwise.
        """
        gradient_masks = self.mask_manager.create_gradient_mask()
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # Find matching mask by checking for exact parameter name match
                matched = False
                for mask_name, mask in gradient_masks.items():
                    # Extract the base name (e.g., "layer1.0.conv1.weight")
                    # mask_name format: "layer1.0.conv1.weight"
                    # name format might be same or slightly different
                    if name.endswith(mask_name.split('.')[-2] + '.' + mask_name.split('.')[-1]):
                        param.grad = param.grad * mask.to(param.grad.device)
                        matched = True
                        break
                    elif mask_name in name or name == mask_name:
                        param.grad = param.grad * mask.to(param.grad.device)
                        matched = True
                        break
    
    def train_task(
        self,
        task_id: int,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        num_epochs: int = 200,
        patience: int = 20,
        verbose: bool = True
    ) -> Dict:
        """
        Train on a single task using the SNV framework.
        
        Args:
            task_id: Task identifier
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum training epochs
            patience: Early stopping patience
            verbose: Whether to show progress
            
        Returns:
            Dictionary with training history
        """
        # Setup optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        
        # Get gradient mask for frozen neurons
        gradient_masks = self.mask_manager.create_gradient_mask()
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        epoch_iterator = range(num_epochs)
        if verbose:
            epoch_iterator = tqdm(epoch_iterator, desc=f"Task {task_id}")
        
        for epoch in epoch_iterator:
            # Training phase
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch in train_loader:
                x, y = batch[0].to(self.device), batch[1].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(x)
                loss = criterion(outputs, y)
                loss.backward()
                
                # Apply gradient mask
                self._apply_gradient_mask(optimizer)
                
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_correct += predicted.eq(y).sum().item()
                train_total += y.size(0)
            
            train_acc = train_correct / train_total
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    x, y = batch[0].to(self.device), batch[1].to(self.device)
                    outputs = self.model(x)
                    loss = criterion(outputs, y)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_correct += predicted.eq(y).sum().item()
                    val_total += y.size(0)
            
            val_acc = val_correct / val_total
            avg_val_loss = val_loss / len(val_loader)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = copy.deepcopy(self.model.state_dict())
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                break
            
            if verbose:
                epoch_iterator.set_postfix({
                    'train_acc': f'{train_acc:.4f}',
                    'val_acc': f'{val_acc:.4f}'
                })
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        # Compute mean activations for Shapley value estimation
        mean_computer = MeanActivationComputer(
            self.model, self.mask_manager.neuron_info, self.device
        )
        mean_computer.compute_mean_activations(val_loader)
        
        # Estimate Shapley values
        shapley_estimator = ShapleyNeuronEstimator(
            self.model,
            self.mask_manager.neuron_info,
            mean_computer.mean_activations,
            self.device,
            self.truncation_threshold,
            self.confidence_level
        )
        
        if verbose:
            print(f"\nEstimating Shapley values for task {task_id}...")
        
        shapley_values = shapley_estimator.estimate_shapley_values(
            val_loader,
            max_iterations=500,
            verbose=verbose
        )
        
        # Select important neurons
        available_mask = self.mask_manager.get_available_neurons()
        task_mask = shapley_estimator.select_top_k_neurons(
            shapley_values, self.sparsity_ratio, available_mask
        )
        
        # Update cumulative mask
        self.mask_manager.update_cumulative_mask(task_id, task_mask)
        
        if verbose:
            capacity_used = self.mask_manager.get_capacity_used()
            print(f"Task {task_id}: Capacity used = {capacity_used:.2f}%")
        
        return {
            'shapley_values': shapley_values,
            'task_mask': task_mask,
            'capacity_used': self.mask_manager.get_capacity_used()
        }
    
    def evaluate(
        self,
        test_loader: torch.utils.data.DataLoader,
        task_id: Optional[int] = None
    ) -> float:
        """
        Evaluate model on test data.
        
        Args:
            test_loader: Test data loader
            task_id: Optional task ID for Task-IL scenario
            
        Returns:
            Test accuracy
        """
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                x, y = batch[0].to(self.device), batch[1].to(self.device)
                outputs = self.model(x)
                _, predicted = outputs.max(1)
                correct += predicted.eq(y).sum().item()
                total += y.size(0)
        
        return correct / total if total > 0 else 0
    
    def evaluate_all_tasks(
        self,
        test_loaders: List[torch.utils.data.DataLoader],
        current_task: int
    ) -> np.ndarray:
        """
        Evaluate on all tasks seen so far.
        
        Args:
            test_loaders: List of test loaders for each task
            current_task: Current task index (0-indexed)
            
        Returns:
            Array of accuracies for each task
        """
        accuracies = []
        for task_id in range(current_task + 1):
            acc = self.evaluate(test_loaders[task_id], task_id)
            accuracies.append(acc)
        return np.array(accuracies)

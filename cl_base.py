"""
Shared scaffolding for every continual-learning method in this repo.

All methods -- SNV and the baselines alike -- run through one training loop so
that comparisons differ only in the algorithm, never in the optimiser, the
early-stopping rule, the head layout or the evaluation protocol.

Head convention (identical to SNV, see snv_core.SNVContinualLearner):
  TIL  -- train and evaluate through the head of the task in question.
  CIL  -- train and evaluate over the concatenation of every head created so
          far, whose order reproduces the global class index.
"""

import copy
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class ContinualMethod:
    """Base class: subclasses override the hooks, not the loop."""

    name = 'base'

    def __init__(self, model: nn.Module, device: torch.device, scenario: str = 'class_il',
                 lr: float = 1e-3, weight_decay: float = 0.0,
                 freeze_old_heads: bool = False, **kwargs):
        self.model = model.to(device)
        self.device = device
        self.scenario = scenario
        self.lr = lr
        self.weight_decay = weight_decay
        self.criterion = nn.CrossEntropyLoss()
        # models.ContinualLearningModel.freeze_heads_before documents the head
        # convention as "h_1..h_{t-1} are stored, not retrained"; only SNV calls
        # it, because for SNV that freezing IS the method.  Exposed here as an
        # option, OFF by default so each baseline keeps its published semantics
        # (LwF, for one, is defined with the old heads updating under
        # distillation).  MEASURED: turning it on does NOT lift previous-task
        # accuracy off 0.00 -- SGD CIL still gives [[65.9, 0.1], [0.0, 69.5]] --
        # because the zeros come from backbone drift, not from the old heads
        # being pushed down.  A pinned head reading drifted features is just as
        # useless as a retrained one.
        self.freeze_old_heads = freeze_old_heads
        self.config = kwargs
        self.history: List[Dict] = []

    # -- hooks -------------------------------------------------------------- #
    def before_task(self, task_id: int, train_loader, val_loader) -> None:
        pass

    def after_task(self, task_id: int, train_loader, val_loader) -> Dict:
        return {}

    def extra_loss(self, x, y, logits, task_id) -> torch.Tensor:
        """Regularisation / replay term added to the cross-entropy."""
        return torch.zeros((), device=self.device)

    def after_backward(self, task_id: int) -> None:
        pass

    def before_loss_backward(self, task_loss, task_id: int) -> None:
        """Observe the task loss separately from any stability regularizer."""
        pass

    def training_state_dict(self):
        return None

    def load_training_state_dict(self, state):
        pass

    def after_step(self, task_id: int) -> None:
        pass

    def after_epoch(self, task_id: int, epoch: int) -> None:
        """Optional topology/schedule update between training epochs."""
        pass

    def after_best_state_loaded(self, task_id: int) -> None:
        """Synchronise non-state_dict bookkeeping after early-stop restore."""
        pass

    def build_optimizer(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        from training_policy import optimizer_for
        return optimizer_for(self, params)

    # -- forward ------------------------------------------------------------ #
    def logits(self, x, task_id: int, for_training: bool = True):
        if self.scenario == 'task_il':
            return self.model(x, task_id)
        return self.model(x)

    # -- loop --------------------------------------------------------------- #
    def train_task(self, task_id: int, train_loader, val_loader, num_epochs: int = 200,
                   patience: int = 20, verbose: bool = True) -> Dict:
        if hasattr(self.model, 'ensure_head'):
            self.model.ensure_head(task_id)
            if self.freeze_old_heads and hasattr(self.model, 'freeze_heads_before'):
                self.model.freeze_heads_before(task_id)
        self.before_task(task_id, train_loader, val_loader)
        optimizer = self.build_optimizer()
        from training_policy import EpochPolicy
        policy = EpochPolicy(self, optimizer, num_epochs, patience)
        epoch_log, best_epoch = [], None

        best_loss, best_state, waited = float('inf'), None, 0
        best_training_state = None
        bar = tqdm(range(num_epochs), desc=f'{self.name} T{task_id}', disable=not verbose)

        for epoch in bar:
            self.model.train()
            for batch in train_loader:
                x, y = batch[0].to(self.device), batch[1].to(self.device)
                optimizer.zero_grad(set_to_none=True)
                out = self.logits(x, task_id)
                task_loss = self.criterion(out, y)
                loss = task_loss + self.extra_loss(x, y, out, task_id)
                self.before_loss_backward(task_loss, task_id)
                loss.backward()
                self.after_backward(task_id)
                optimizer.step()
                self.after_step(task_id)

            val_loss, val_acc = self.validate(val_loader, task_id)
            if not np.isfinite(val_loss):
                raise FloatingPointError(f'Nonfinite validation loss at task {task_id}, epoch {epoch}')
            epoch_log.append(dict(epoch=epoch+1, validation_loss=val_loss,
                validation_accuracy=val_acc, lr=optimizer.param_groups[0]['lr']))
            if val_loss < best_loss - 1e-6:
                best_loss, waited = val_loss, 0
                best_epoch = epoch+1
                best_state = copy.deepcopy(self.model.state_dict())
                best_training_state = copy.deepcopy(self.training_state_dict())
            else:
                waited += 1
                if policy.should_stop(epoch+1, waited):
                    break
            policy.step()
            if epoch < num_epochs - 1:
                self.after_epoch(task_id, epoch)
            bar.set_postfix({'val_loss': f'{val_loss:.4f}', 'val_acc': f'{val_acc:.4f}'})
        bar.close()

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.load_training_state_dict(best_training_state)
            self.after_best_state_loaded(task_id)

        result = self.after_task(task_id, train_loader, val_loader) or {}
        result.setdefault('task_id', task_id)
        result.update(epochs_run=len(epoch_log), epoch_cap=num_epochs, best_epoch=best_epoch,
            best_validation_loss=best_loss, epoch_log=epoch_log,
            stop_reason='early_stopping' if len(epoch_log)<num_epochs else 'epoch_cap',
            cap_reached_while_improving=len(epoch_log)==num_epochs and waited<patience)
        self.history.append(result)
        return result

    def validate(self, loader, task_id: int):
        self.model.eval()
        loss_sum = correct = total = 0
        with torch.no_grad():
            for batch in loader:
                x, y = batch[0].to(self.device), batch[1].to(self.device)
                out = self.logits(x, task_id, for_training=False)
                loss_sum += self.criterion(out, y).item() * y.numel()
                correct += out.argmax(1).eq(y).sum().item()
                total += y.numel()
        return loss_sum / max(total, 1), correct / max(total, 1)

    # -- evaluation --------------------------------------------------------- #
    def evaluate(self, loader, task_id: Optional[int] = None) -> float:
        self.model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in loader:
                x, y = batch[0].to(self.device), batch[1].to(self.device)
                out = self.predict(x, task_id)
                correct += out.argmax(1).eq(y).sum().item()
                total += y.numel()
        return correct / total if total else 0.0

    def predict(self, x, task_id: Optional[int]):
        if self.scenario == 'task_il' and task_id is not None:
            return self.model(x, task_id)
        return self.model(x)

    def evaluate_all_tasks(self, test_loaders, current_task: int) -> np.ndarray:
        """Row t of the accuracy matrix: every task, past AND future.

        Cells with k > current_task are the model's accuracy on tasks it has not
        been trained on yet.  They are what forward transfer is actually measured
        from; the previous version filled only k <= current_task plus a single
        zero-shot cell, so most of the matrix was never populated.
        """
        model = getattr(self, 'model', None)
        row = np.full(len(test_loaders), np.nan)
        for k in range(current_task + 1):
            row[k] = self.evaluate(test_loaders[k], k)
        # Future-task diagnostics require their labels in the output space,
        # but must not change the denominator of standard seen-class accuracy.
        if hasattr(model, 'full_output_space'):
            with model.full_output_space(len(test_loaders)):
                for k in range(current_task + 1, len(test_loaders)):
                    row[k] = self.evaluate(test_loaders[k], k)
        return row

    # -- utilities ---------------------------------------------------------- #
    def backbone_parameters(self):
        for name, p in self.model.named_parameters():
            if not name.startswith('heads.'):
                yield name, p

    def regularized_parameters(self):
        """Learned parameters, including classifier rows of observed tasks."""
        active = set(self.model.active_tasks()) if hasattr(self.model, 'active_tasks') else None
        for name, p in self.model.named_parameters():
            if name.startswith('heads.') and active is not None:
                if int(name.split('.')[1]) not in active:
                    continue
            yield name, p


class Reservoir:
    """Reservoir-sampled episodic buffer holding (x, y, task_id, logits).

    The task id of each row is stored so that a consumer under TIL can route a
    replayed sample through the head of the task it came from.  Replaying old
    samples through the *current* task's head trains that head on data whose
    labels belong to a different label space, which is worse than not replaying
    at all.

    Under CIL the logit vector widens as heads are added, so rows recorded early
    are narrower than rows recorded later.  Stored logits are right-padded to the
    widest seen and ``zw`` keeps each row's valid width, so a consumer can mask
    the padding instead of regressing against zeros that were never predictions.
    """

    def __init__(self, capacity: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.x: Optional[torch.Tensor] = None
        self.y: Optional[torch.Tensor] = None
        self.z: Optional[torch.Tensor] = None
        self.zw: Optional[torch.Tensor] = None
        self.t: Optional[torch.Tensor] = None
        self.seen = 0

    def __len__(self):
        return 0 if self.x is None else self.x.shape[0]

    def _widen(self, width: int):
        if self.z is None or self.z.shape[1] >= width:
            return
        pad = torch.zeros(self.z.shape[0], width - self.z.shape[1], dtype=self.z.dtype)
        self.z = torch.cat([self.z, pad], dim=1)

    def add(self, x, y, z=None, task_id=None):
        x, y = x.detach().cpu(), y.detach().cpu()
        z = None if z is None else z.detach().cpu()
        original_width = None if z is None else z.shape[1]
        if self.x is None:
            self.x = torch.empty((0, *x.shape[1:]), dtype=x.dtype)
            self.y = torch.empty((0,), dtype=y.dtype)
            self.t = torch.empty((0,), dtype=torch.long)
            if z is not None:
                self.z = torch.empty((0, z.shape[1]), dtype=z.dtype)
                self.zw = torch.empty((0,), dtype=torch.long)

        if z is not None:
            self._widen(z.shape[1])
            width = self.z.shape[1]
            if z.shape[1] < width:
                z = torch.cat([z, torch.zeros(z.shape[0], width - z.shape[1], dtype=z.dtype)], 1)

        for i in range(x.shape[0]):
            if len(self) < self.capacity:
                self.x = torch.cat([self.x, x[i:i + 1]])
                self.y = torch.cat([self.y, y[i:i + 1]])
                self.t = torch.cat([self.t, torch.tensor([-1 if task_id is None
                                                          else int(task_id)])])
                if z is not None:
                    self.z = torch.cat([self.z, z[i:i + 1]])
                    self.zw = torch.cat([self.zw, torch.tensor([original_width])])
            else:
                j = np.random.randint(0, self.seen + 1)
                if j < self.capacity:
                    self.x[j] = x[i]
                    self.y[j] = y[i]
                    self.t[j] = -1 if task_id is None else int(task_id)
                    if z is not None:
                        self.z[j] = z[i]
                        self.zw[j] = original_width
            self.seen += 1

    def sample(self, n: int):
        """Returns (x, y, z, zw); z and zw are None when no logits are stored."""
        if len(self) == 0:
            return None
        idx = torch.from_numpy(np.random.choice(len(self), min(n, len(self)), replace=False))
        z = None if self.z is None else self.z[idx].to(self.device)
        zw = None if self.zw is None else self.zw[idx].to(self.device)
        self._last_idx = idx
        return self.x[idx].to(self.device), self.y[idx].to(self.device), z, zw

    def sample_with_tasks(self, n: int):
        """Like ``sample`` but also returns the task id each row was stored under."""
        out = self.sample(n)
        if out is None:
            return None
        t = self.t[self._last_idx].to(self.device) if self.t is not None else None
        return (*out, t)

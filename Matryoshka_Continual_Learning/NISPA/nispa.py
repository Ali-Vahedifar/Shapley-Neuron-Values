"""Faithful phase-wise NISPA adaptation for CIFAR-100."""

import copy
import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from cl_base import ContinualMethod


class _MaskedMixin:
    def _init_masks(self):
        self.register_buffer('weight_mask', torch.ones_like(self.weight))
        if self.bias is not None:
            self.register_buffer('bias_mask', torch.ones_like(self.bias))

    def set_mask(self, weight, bias=None):
        self.weight_mask.copy_(weight.to(self.weight_mask))
        self.weight.data.mul_(self.weight_mask)
        if self.bias is not None and bias is not None:
            self.bias_mask.copy_(bias.to(self.bias_mask))
            self.bias.data.mul_(self.bias_mask)


class MaskedConv2d(_MaskedMixin, nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_masks()

    def forward(self, x):
        return F.conv2d(x, self.weight * self.weight_mask,
                        None if self.bias is None else self.bias * self.bias_mask,
                        self.stride, self.padding, self.dilation, self.groups)


class MaskedLinear(_MaskedMixin, nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_masks()

    def forward(self, x):
        return F.linear(x, self.weight * self.weight_mask,
                        None if self.bias is None else self.bias * self.bias_mask)


class NISPANet(nn.Module):
    """Official CIFAR NISPA ConvNet: 64,64,128,128,1024,100."""

    def __init__(self, classes_per_task=5, num_classes=100, scenario='class_il'):
        super().__init__()
        self.classes_per_task = classes_per_task
        self.num_classes = num_classes
        self.scenario = scenario
        self.current_task = 0
        self.conv1 = MaskedConv2d(3, 64, 3, padding=1)
        self.conv2 = MaskedConv2d(64, 64, 3, padding=1)
        self.conv3 = MaskedConv2d(64, 128, 3, padding=1)
        self.conv4 = MaskedConv2d(128, 128, 3, padding=1)
        self.fc1 = MaskedLinear(128 * 8 * 8, 1024)
        self.fc2 = MaskedLinear(1024, num_classes)
        self.feature_dim = 1024
        for layer in self.sparse_layers():
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            nn.init.zeros_(layer.bias)

    def sparse_layers(self):
        return [self.conv1, self.conv2, self.conv3, self.conv4, self.fc1, self.fc2]

    def ensure_head(self, task_id, num_classes=None):
        self.current_task = max(self.current_task, int(task_id))
        return self.fc2

    def active_tasks(self):
        return list(range(self.current_task + 1))

    def forward_activations(self, x):
        a1 = F.relu(self.conv1(x))
        a2 = F.max_pool2d(F.relu(self.conv2(a1)), 2)
        a3 = F.relu(self.conv3(a2))
        a4 = F.max_pool2d(F.relu(self.conv4(a3)), 2)
        a5 = F.relu(self.fc1(a4.flatten(1)))
        out = self.fc2(a5)
        return out, [a1, a2, a3, a4, a5, out]

    def get_features(self, x):
        return self.forward_activations(x)[1][-2]

    def forward(self, x, task_id=None):
        out = self.forward_activations(x)[0]
        if task_id is not None:
            start = int(task_id) * self.classes_per_task
            return out[:, start:start + self.classes_per_task]
        seen = (self.current_task + 1) * self.classes_per_task
        return out[:, :seen] if self.scenario == 'class_il' else out


def build_nispa_model(dataset, classes_per_task, scenario):
    if dataset != 'cifar100':
        raise ValueError('the faithful NISPA ConvNet is CIFAR-100 only')
    return NISPANet(classes_per_task, 100, scenario)


class NISPA(ContinualMethod):
    """ICML 2022 NISPA with selection/rewiring cycles inside every task."""

    name = 'nispa'

    def __init__(self, *args, prune_perc=90.0, recovery_perc=2.5,
                 phase_epochs=5, max_phases=30, grow_init='normal', **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.model, NISPANet):
            raise TypeError('faithful NISPA requires NISPANet')
        self.prune_perc = float(prune_perc)
        self.recovery_perc = float(recovery_perc) / 100.0
        self.phase_epochs = int(phase_epochs)
        self.max_phases = int(max_phases)
        self.grow_init = grow_init
        self.stable_sets: List[set] = []
        self.freeze_masks = None
        self._random_prune()

    def _random_prune(self):
        density = 1.0 - self.prune_perc / 100.0
        for index, layer in enumerate(self.model.sparse_layers()):
            if isinstance(layer, MaskedConv2d):
                connections = (torch.ones(layer.out_channels, layer.in_channels)
                               if index == 0 else
                               (torch.rand(layer.out_channels, layer.in_channels) < density).float())
                mask = connections[:, :, None, None].expand_as(layer.weight).clone()
            else:
                mask = (torch.rand_like(layer.weight) < density).float()
            layer.set_mask(mask, torch.ones_like(layer.bias))

    def logits(self, x, task_id, for_training=True):
        self.model.current_task = task_id
        out = self.model.forward_activations(x)[0]
        if self.scenario == 'task_il':
            start = task_id * self.model.classes_per_task
            return out[:, start:start + self.model.classes_per_task]
        return out[:, :(task_id + 1) * self.model.classes_per_task]

    def predict(self, x, task_id):
        task = int(task_id) if self.scenario == 'task_il' else self.model.current_task
        return self.logits(x, task, False)

    def _train_phase(self, task_id, loader, epochs, verbose):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                                     weight_decay=self.weight_decay)
        bar = tqdm(range(epochs), desc=f'nispa T{task_id} phase', disable=not verbose)
        for _ in bar:
            self.model.train()
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                F.cross_entropy(self.logits(x, task_id), y).backward()
                if self.freeze_masks is not None:
                    for layer, (frozen, frozen_bias) in zip(
                            self.model.sparse_layers(), self.freeze_masks):
                        if layer.weight.grad is not None:
                            layer.weight.grad.masked_fill_(frozen, 0)
                        if layer.bias.grad is not None:
                            layer.bias.grad.masked_fill_(frozen_bias, 0)
                optimizer.step()
                for layer in self.model.sparse_layers():
                    layer.weight.data.mul_(layer.weight_mask)
        bar.close()

    @torch.no_grad()
    def _accuracy(self, loader, task_id):
        self.model.eval()
        correct = total = 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            correct += self.logits(x, task_id, False).argmax(1).eq(y).sum().item()
            total += y.numel()
        return correct / max(total, 1)

    @torch.no_grad()
    def _activations(self, loader):
        sums = None
        self.model.eval()
        for x, _ in loader:
            _, values = self.model.forward_activations(x.to(self.device))
            batch = [v.sum((0, 2, 3)) if v.dim() == 4 else v.sum(0) for v in values]
            sums = batch if sums is None else [a + b for a, b in zip(sums, batch)]
        return sums

    @staticmethod
    def _activation_mass(values, fraction):
        order = torch.argsort(values, descending=True)
        target = values.sum() * fraction
        if target <= 0:
            return set()
        count = int(torch.searchsorted(torch.cumsum(values[order], 0), target)) + 1
        return set(order[:count].cpu().tolist())

    def _candidates(self, loader, fraction, task_id):
        values = self._activations(loader)
        result = [set(range(3))]
        result.extend(self._activation_mass(v, fraction) for v in values[:-1])
        result.append(set(range((task_id + 1) * self.model.classes_per_task)))
        return result

    def _expanded_sources(self, layer_index, source, width, device):
        if layer_index != 4:
            return torch.tensor(sorted(source), device=device, dtype=torch.long)
        if not source:
            return torch.empty(0, device=device, dtype=torch.long)
        return torch.cat([torch.arange(channel * 64, (channel + 1) * 64,
                                       device=device) for channel in sorted(source)])

    def _freeze_drop(self, stable):
        frozen_all, dropped_all = [], []
        for index, layer in enumerate(self.model.sparse_layers()):
            mask = layer.weight_mask
            target = torch.tensor(sorted(stable[index + 1]), device=mask.device,
                                  dtype=torch.long)
            source = self._expanded_sources(index, stable[index], mask.shape[1],
                                            mask.device)
            frozen = torch.zeros_like(mask, dtype=torch.bool)
            drop = torch.zeros_like(mask, dtype=torch.bool)
            if target.numel() and source.numel():
                if mask.dim() == 4:
                    frozen[target[:, None], source[None, :], :, :] = True
                else:
                    frozen[target[:, None], source[None, :]] = True
                plastic = torch.tensor(sorted(set(range(mask.shape[1])) -
                                              set(source.cpu().tolist())),
                                       device=mask.device, dtype=torch.long)
                if plastic.numel():
                    if mask.dim() == 4:
                        drop[target[:, None], plastic[None, :], :, :] = True
                    else:
                        drop[target[:, None], plastic[None, :]] = True
            frozen &= mask.bool()
            dropping = drop & mask.bool()
            amount = int(dropping.sum())
            mask[dropping] = 0
            layer.weight.data[dropping] = 0
            frozen_bias = torch.zeros_like(layer.bias, dtype=torch.bool)
            if target.numel():
                frozen_bias[target] = True
            frozen_all.append((frozen, frozen_bias))
            dropped_all.append(amount)
        return frozen_all, dropped_all

    def _new_values(self, layer, shape):
        active = layer.weight.data[layer.weight_mask.bool()]
        if self.grow_init == 'zero' or not active.numel():
            return torch.zeros(shape, device=layer.weight.device)
        if self.grow_init == 'uniform':
            return torch.empty(shape, device=layer.weight.device).uniform_(
                float(active.min()), float(active.max()))
        return torch.empty(shape, device=layer.weight.device).normal_(
            float(active.mean()), max(float(active.std()), 1e-8))

    def _grow(self, stable, dropped):
        for index, (layer, amount) in enumerate(zip(self.model.sparse_layers(), dropped)):
            if amount <= 0:
                continue
            mask = layer.weight_mask
            plastic_target = torch.tensor(
                sorted(set(range(mask.shape[0])) - stable[index + 1]),
                device=mask.device, dtype=torch.long)
            if not plastic_target.numel():
                continue
            possible = torch.zeros_like(mask, dtype=torch.bool)
            possible[plastic_target] = True
            possible &= ~mask.bool()
            if mask.dim() == 4:
                pairs = torch.nonzero(possible[..., 0, 0])
                count = min(len(pairs), amount // (mask.shape[2] * mask.shape[3]))
                if count:
                    chosen = pairs[torch.randperm(len(pairs), device=mask.device)[:count]]
                    rows, cols = chosen[:, 0], chosen[:, 1]
                    mask[rows, cols] = 1
                    layer.weight.data[rows, cols] = self._new_values(
                        layer, (count, mask.shape[2], mask.shape[3]))
            else:
                pairs = torch.nonzero(possible)
                count = min(len(pairs), amount)
                if count:
                    chosen = pairs[torch.randperm(len(pairs), device=mask.device)[:count]]
                    rows, cols = chosen[:, 0], chosen[:, 1]
                    mask[rows, cols] = 1
                    layer.weight.data[rows, cols] = self._new_values(layer, (count,))

    def _reinitialize(self):
        for layer, (frozen, frozen_bias) in zip(
                self.model.sparse_layers(), self.freeze_masks):
            old, old_bias = layer.weight.data.clone(), layer.bias.data.clone()
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            layer.weight.data[frozen] = old[frozen]
            layer.weight.data.mul_(layer.weight_mask)
            nn.init.zeros_(layer.bias)
            layer.bias.data[frozen_bias] = old_bias[frozen_bias]

    def train_task(self, task_id, train_loader, val_loader, num_epochs=150,
                   patience=20, verbose=True):
        self.model.ensure_head(task_id)
        inherited = [set(v) for v in self.stable_sets]
        best = 0.0
        accepted_state = None
        accepted_stable = None
        phase_stable = inherited
        accepted = 0
        phase_limit = min(self.max_phases,
                          max(3, math.ceil(max(num_epochs, self.phase_epochs) /
                                           self.phase_epochs)))
        threshold = 1.0
        for phase in range(1, phase_limit + 1):
            self._train_phase(task_id, train_loader, self.phase_epochs, verbose)
            accuracy = self._accuracy(val_loader, task_id)
            threshold = 0.5 * (1 + math.cos(phase * math.pi / self.max_phases))
            if ((phase > 2 and accuracy < best - self.recovery_perc)
                    or threshold <= 0.05):
                break
            best = max(best, accuracy)
            accepted_state = copy.deepcopy(self.model.state_dict())
            # The cached model was trained with the connectivity produced by
            # the *previous* phase.  Associate it with that stable set, as in
            # CacheM[p] / CacheS[p-1] in Algorithm 1; pairing it with the newly
            # selected candidates would freeze an untrained subnetwork.
            accepted_stable = [set(v) for v in phase_stable]
            candidate = self._candidates(train_loader, threshold, task_id)
            next_stable = (candidate if not inherited else
                           [a | b for a, b in zip(inherited, candidate)])
            _, dropped = self._freeze_drop(next_stable)
            self._grow(next_stable, dropped)
            phase_stable = next_stable
            accepted += 1
        if accepted_state is not None:
            self.model.load_state_dict(accepted_state)
        if accepted_stable is None:
            accepted_stable = self._candidates(train_loader, 1.0, task_id)
            if inherited:
                accepted_stable = [a | b for a, b in zip(inherited, accepted_stable)]
        self.stable_sets = accepted_stable
        self.freeze_masks, final_dropped = self._freeze_drop(self.stable_sets)
        self._grow(self.stable_sets, final_dropped)
        self._reinitialize()
        used = sum(len(v) for v in self.stable_sets[1:-1])
        total = 64 + 64 + 128 + 128 + 1024
        result = {'task_id': task_id, 'phases': accepted,
                  'capacity_used': 100.0 * used / total,
                  'stable_threshold': threshold, 'validation_accuracy': best}
        self.history.append(result)
        return result

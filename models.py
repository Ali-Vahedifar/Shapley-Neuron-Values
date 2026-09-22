"""
Backbones and the multi-head wrapper used by every method in this repo.

  PMNIST         4-layer MLP, 200 units per layer
  CIFAR-100      ResNet-18, 3x3 stem, no max-pool     (32 x 32)
  TinyImageNet   ResNet-18, 3x3 stem, no max-pool     (64 x 64)
  ImageNet-1k    ResNet-18, 7x7 stride-2 stem + pool  (224 x 224)

He (Kaiming) initialisation throughout, per the experimental setup.

The wrapper keeps one head h_t per task, as the Train procedure requires.  A
backbone therefore exposes ``get_features`` and never its own classifier.
"""

import math
from contextlib import contextmanager
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Backbones
# --------------------------------------------------------------------------- #
class MLP(nn.Module):
    """4 hidden layers of 200 ReLU units (PMNIST)."""

    def __init__(self, input_dim: int = 784, hidden_dim: int = 200, num_layers: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.feature_dim = hidden_dim
        layers = []
        in_features = input_dim
        for _ in range(num_layers):
            layer = nn.Linear(in_features, hidden_dim)
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(layer.bias)
            layers.append(layer)
            in_features = hidden_dim
        self.hidden_layers = nn.ModuleList(layers)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.reshape(x.size(0), -1)
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        return x

    def forward(self, x):
        return self.get_features(x)


def conv3x3(cin, cout, stride=1):
    return nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)


def conv1x1(cin, cout, stride=1):
    return nn.Conv2d(cin, cout, 1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class Bottleneck(nn.Module):
    """ResNet-50's 1x1 -> 3x3 -> 1x1 block, expansion 4.

    Only conv3/bn3 (and the downsample) carry the block's *output* channel
    index, so anything that treats a channel index as a feature index -- MCL's
    nesting scale, SNV's neuron mask -- must key on those, not on conv2 the way
    it does for BasicBlock.
    """

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet(nn.Module):
    """ResNet-18 (BasicBlock, [2,2,2,2]) or ResNet-50 (Bottleneck, [3,4,6,3])."""

    def __init__(self, initial_channels: int = 64, input_size: int = 32,
                 block=BasicBlock, layers=(2, 2, 2, 2)):
        super().__init__()
        self.inplanes = initial_channels
        self.block = block
        self.layers_cfg = tuple(layers)
        self.feature_dim = 512 * block.expansion
        self.input_size = input_size

        if input_size <= 64:
            self.conv1 = nn.Conv2d(3, initial_channels, 3, stride=1, padding=1, bias=False)
            self.maxpool = nn.Identity()
        else:
            self.conv1 = nn.Conv2d(3, initial_channels, 7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.bn1 = nn.BatchNorm2d(initial_channels)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(64, layers[0], 1)
        self.layer2 = self._make_layer(128, layers[1], 2)
        self.layer3 = self._make_layer(256, layers[2], 2)
        self.layer4 = self._make_layer(512, layers[3], 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._initialize_weights()

    def _make_layer(self, planes, blocks, stride):
        block = self.block
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion))
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        layers += [block(self.inplanes, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_features(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return torch.flatten(self.avgpool(x), 1)

    def forward(self, x):
        return self.get_features(x)


def ResNet18(initial_channels: int = 64, input_size: int = 32):
    return ResNet(initial_channels, input_size, BasicBlock, (2, 2, 2, 2))


def ResNet50(initial_channels: int = 64, input_size: int = 32):
    return ResNet(initial_channels, input_size, Bottleneck, (3, 4, 6, 3))

# --------------------------------------------------------------------------- #
# Multi-head wrapper
# --------------------------------------------------------------------------- #
class ContinualLearningModel(nn.Module):
    """Backbone plus one head h_t per task.

    ``forward(x, t)``  -> logits of head t                       (TIL)
    ``forward(x)``     -> logits of every head created so far,
                          concatenated in task order              (CIL)

    The concatenation order reproduces the global class index the CIL benchmark
    assigns, so an argmax over it is an argmax over all classes seen so far.
    """

    def __init__(self, backbone: nn.Module, feature_dim: int, classes_per_task: int,
                 num_tasks: int, scenario: str = 'class_il'):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.classes_per_task = classes_per_task
        self.num_tasks = num_tasks
        self.scenario = scenario
        self.heads = nn.ModuleDict()
        # How many tasks the model has actually been trained on.  Kept separate
        # from which heads exist so the full T x T accuracy matrix can be filled:
        # evaluating a *future* task needs its head to exist, but the CIL softmax
        # during training must stay at the width of the tasks seen so far, or
        # creating heads early would silently change every method's loss.
        self.seen_upto = -1

    # -- heads -------------------------------------------------------------- #
    def ensure_head(self, task_id: int, num_classes: Optional[int] = None) -> nn.Linear:
        key = str(task_id)
        if key not in self.heads:
            head = nn.Linear(self.feature_dim, num_classes or self.classes_per_task)
            nn.init.kaiming_normal_(head.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(head.bias)
            self.heads[key] = head.to(next(self.backbone.parameters()).device)
        self.seen_upto = max(self.seen_upto, task_id)
        return self.heads[key]

    def freeze_heads_before(self, task_id: int) -> None:
        """h_1..h_{t-1} are stored, not retrained."""
        for key, head in self.heads.items():
            requires_grad = int(key) >= task_id
            for p in head.parameters():
                p.requires_grad_(requires_grad)

    def active_tasks(self) -> List[int]:
        """Tasks whose logits take part in the CIL concatenation."""
        tasks = sorted(int(k) for k in self.heads.keys())
        if self.seen_upto >= 0:
            tasks = [t for t in tasks if t <= self.seen_upto]
        return tasks

    def all_heads(self) -> List[int]:
        return sorted(int(k) for k in self.heads.keys())

    @contextmanager
    def full_output_space(self, num_tasks: Optional[int] = None):
        """Temporarily widen the CIL output to every head, for future-task eval.

        A[t, k] with k > t is only defined if task k's classes are in the output
        space.  This widens it for the measurement and restores it afterwards, so
        training semantics are untouched.
        """
        previous = self.seen_upto
        for t in range(num_tasks or self.num_tasks):
            self.ensure_head(t)
        self.to(next(self.backbone.parameters()).device)
        self.seen_upto = (num_tasks or self.num_tasks) - 1
        try:
            yield self
        finally:
            self.seen_upto = previous

    # -- forward ------------------------------------------------------------ #
    def forward(self, x: torch.Tensor, task_id: Optional[int] = None) -> torch.Tensor:
        features = self.backbone.get_features(x)
        if task_id is not None:
            return self.heads[str(task_id)](features)
        tasks = self.active_tasks()
        if not tasks:
            raise RuntimeError('no task head has been created yet')
        return torch.cat([self.heads[str(t)](features) for t in tasks], dim=1)

    def get_features(self, x):
        return self.backbone.get_features(x)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
# Backbone per dataset.  ResNet-18 is adequate at 32px; at 64px and 224px the
# depth matters more than the saving, so Tiny-ImageNet and ImageNet-1k use
# ResNet-50 by default.  Overridable with --backbone.
_INPUT_SIZE = {'cifar10': 32, 'cifar100': 32, 'cifar20': 32, 'tinyimagenet': 64,
               'imagenet1k': 224}
_DEFAULT_BACKBONE = {'cifar10': 'resnet18', 'cifar100': 'resnet18', 'cifar20': 'resnet18',
                     'tinyimagenet': 'resnet50',
                     'imagenet1k': 'resnet50'}
_RESNETS = {'resnet18': ResNet18, 'resnet50': ResNet50}


def _BACKBONES_get(dataset, backbone=None):
    if dataset == 'pmnist':
        return MLP(784, 200, 4)
    name = backbone or _DEFAULT_BACKBONE[dataset]
    if name not in _RESNETS:
        raise ValueError(f'unknown backbone {name!r}; expected one of {sorted(_RESNETS)}')
    return _RESNETS[name](64, input_size=_INPUT_SIZE[dataset])


def create_backbone(dataset: str, backbone: Optional[str] = None) -> nn.Module:
    dataset = dataset.lower()
    if dataset not in _DEFAULT_BACKBONE and dataset != 'pmnist':
        raise ValueError(f'unknown dataset {dataset!r}')
    return _BACKBONES_get(dataset, backbone)


def create_model(dataset: str, classes_per_task: int, num_tasks: int,
                 scenario: str = 'class_il',
                 backbone_name: Optional[str] = None) -> ContinualLearningModel:
    """Build the multi-head model used for both CIL and TIL."""
    backbone = create_backbone(dataset, backbone_name)
    return ContinualLearningModel(
        backbone=backbone,
        feature_dim=backbone.feature_dim,
        classes_per_task=classes_per_task,
        num_tasks=num_tasks,
        scenario=scenario,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_neurons(model: nn.Module) -> int:
    """N = sum_l C_l over the backbone (classifier heads excluded)."""
    from snv_core import build_neuron_index
    return sum(g.num_neurons for g in build_neuron_index(model))

"""
Model architectures for SNV Continual Learning experiments.

Architectures:
- MLP: 4-layer MLP with 200 neurons per layer for PMNIST
- ResNet-18: Standard ResNet-18 for CIFAR-100 and TinyImageNet

Anonymous submission for ICML 2026.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class MLP(nn.Module):
    """
    4-layer MLP with 200 neurons per layer for PMNIST experiments.
    
    Architecture:
    - Input: 784 (28x28 flattened)
    - Hidden 1: 200 neurons, ReLU
    - Hidden 2: 200 neurons, ReLU
    - Hidden 3: 200 neurons, ReLU
    - Hidden 4: 200 neurons, ReLU
    - Output: num_classes
    
    Uses He initialization for weights.
    """
    
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 200,
        num_layers: int = 4,
        num_classes: int = 10
    ):
        super(MLP, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # Build hidden layers
        layers = []
        in_features = input_dim
        
        for i in range(num_layers):
            layer = nn.Linear(in_features, hidden_dim)
            # He initialization
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(layer.bias)
            layers.append(layer)
            in_features = hidden_dim
        
        self.hidden_layers = nn.ModuleList(layers)
        
        # Output layer (classifier)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        nn.init.kaiming_normal_(self.classifier.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.classifier.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten input if needed
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        
        # Forward through hidden layers
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        
        # Output
        x = self.classifier(x)
        return x
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get features before the classifier."""
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        
        return x


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution."""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False
    )


class BasicBlock(nn.Module):
    """
    Basic building block for ResNet-18.
    
    Structure:
    - Conv 3x3 -> BN -> ReLU
    - Conv 3x3 -> BN
    - Shortcut connection
    - ReLU
    """
    expansion = 1
    
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ):
        super(BasicBlock, self).__init__()
        
        # First convolution
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        
        # Second convolution
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNet18(nn.Module):
    """
    ResNet-18 architecture for CIFAR-100 and TinyImageNet.
    
    Architecture:
    - Conv 3x3, 64 filters
    - Layer 1: 2 BasicBlocks, 64 channels
    - Layer 2: 2 BasicBlocks, 128 channels
    - Layer 3: 2 BasicBlocks, 256 channels
    - Layer 4: 2 BasicBlocks, 512 channels
    - Global Average Pooling
    - FC classifier
    
    Channel progression: {64, 64, 128, 128, 256, 256, 512, 512}
    Uses He initialization for all convolutional layers.
    """
    
    def __init__(
        self,
        num_classes: int = 100,
        initial_channels: int = 64,
        input_size: int = 32
    ):
        super(ResNet18, self).__init__()
        
        self.inplanes = initial_channels
        self.num_classes = num_classes
        
        # Initial convolution - adapted for smaller images (CIFAR/TinyImageNet)
        if input_size <= 32:
            # For CIFAR-100 (32x32)
            self.conv1 = nn.Conv2d(
                3, initial_channels, kernel_size=3, stride=1, padding=1, bias=False
            )
        else:
            # For TinyImageNet (64x64) or larger
            self.conv1 = nn.Conv2d(
                3, initial_channels, kernel_size=7, stride=2, padding=3, bias=False
            )
        
        self.bn1 = nn.BatchNorm2d(initial_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Max pooling only for larger images
        if input_size > 32:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.maxpool = nn.Identity()
        
        # ResNet layers
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)
        
        # He initialization
        self._initialize_weights()
        
    def _make_layer(
        self,
        planes: int,
        blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        """Create a ResNet layer with multiple BasicBlocks."""
        downsample = None
        
        if stride != 1 or self.inplanes != planes * BasicBlock.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * BasicBlock.expansion, stride),
                nn.BatchNorm2d(planes * BasicBlock.expansion),
            )
        
        layers = []
        layers.append(BasicBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * BasicBlock.expansion
        
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get features before the classifier."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x


class ContinualLearningModel(nn.Module):
    """
    Wrapper model for continual learning that supports dynamic head expansion.
    
    For Class-IL: Single output head for all classes
    For Task-IL: Separate heads per task (optional)
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        initial_classes: int = 10,
        scenario: str = 'class_il'
    ):
        super(ContinualLearningModel, self).__init__()
        
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.scenario = scenario
        self.total_classes = initial_classes
        
        # Single unified head for Class-IL
        self.classifier = nn.Linear(feature_dim, initial_classes)
        nn.init.kaiming_normal_(self.classifier.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.classifier.bias)
        
        # Task heads for Task-IL (optional)
        self.task_heads = nn.ModuleDict()
        
    def expand_classifier(self, new_classes: int):
        """
        Expand classifier to accommodate new classes.
        
        Args:
            new_classes: Number of new classes to add
        """
        old_classes = self.total_classes
        self.total_classes += new_classes
        
        # Create new classifier
        new_classifier = nn.Linear(self.feature_dim, self.total_classes)
        nn.init.kaiming_normal_(new_classifier.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(new_classifier.bias)
        
        # Copy old weights
        with torch.no_grad():
            new_classifier.weight[:old_classes] = self.classifier.weight
            new_classifier.bias[:old_classes] = self.classifier.bias
        
        self.classifier = new_classifier
        
    def add_task_head(self, task_id: int, num_classes: int):
        """Add a task-specific head for Task-IL."""
        head = nn.Linear(self.feature_dim, num_classes)
        nn.init.kaiming_normal_(head.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(head.bias)
        self.task_heads[str(task_id)] = head
        
    def forward(
        self,
        x: torch.Tensor,
        task_id: Optional[int] = None
    ) -> torch.Tensor:
        # Get features from backbone
        features = self.backbone.get_features(x)
        
        if self.scenario == 'task_il' and task_id is not None:
            # Use task-specific head
            if str(task_id) in self.task_heads:
                return self.task_heads[str(task_id)](features)
        
        # Default: use unified classifier
        return self.classifier(features)


def create_model(
    dataset: str,
    num_classes: int,
    scenario: str = 'class_il'
) -> nn.Module:
    """
    Factory function to create appropriate model for dataset.
    
    Args:
        dataset: Dataset name ('pmnist', 'cifar100', 'tinyimagenet')
        num_classes: Total number of classes
        scenario: 'class_il' or 'task_il'
        
    Returns:
        Initialized model
    """
    dataset = dataset.lower()
    
    if dataset == 'pmnist':
        return MLP(
            input_dim=784,
            hidden_dim=200,
            num_layers=4,
            num_classes=num_classes
        )
    elif dataset == 'cifar100':
        return ResNet18(
            num_classes=num_classes,
            initial_channels=64,
            input_size=32
        )
    elif dataset == 'tinyimagenet':
        return ResNet18(
            num_classes=num_classes,
            initial_channels=64,
            input_size=64
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_neurons(model: nn.Module) -> int:
    """
    Count total number of neurons (filters/units) in the model.
    
    For SNV, a neuron is defined as:
    - A convolutional filter for Conv layers
    - A hidden unit for Linear layers (excluding classifier)
    """
    total_neurons = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            total_neurons += module.out_channels
        elif isinstance(module, nn.Linear):
            # Exclude final classifier layer
            if 'fc' not in name and 'classifier' not in name:
                total_neurons += module.out_features
    
    return total_neurons

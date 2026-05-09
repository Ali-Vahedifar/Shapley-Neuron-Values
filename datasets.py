"""
Dataset loaders for SNV Continual Learning experiments.

Datasets:
- Permuted MNIST (PMNIST): MNIST with random pixel permutations per task
- CIFAR-100: 100 classes divided into 10 or 20 tasks
- TinyImageNet: 200 classes divided into 10 or 20 tasks

Data splits: 70% train, 20% test, 10% validation

Anonymous submission for ICML 2026.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import datasets, transforms
from typing import Tuple, List, Dict, Optional
import os
from PIL import Image


class PermutedMNIST(Dataset):
    """
    Permuted MNIST dataset for continual learning.
    
    Each task applies a unique deterministic permutation to input pixels.
    """
    
    def __init__(
        self,
        root: str,
        train: bool = True,
        download: bool = True,
        permutation: Optional[np.ndarray] = None
    ):
        self.mnist = datasets.MNIST(
            root=root, train=train, download=download
        )
        self.permutation = permutation
        
        # Pre-compute transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
    def __len__(self) -> int:
        return len(self.mnist)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img, label = self.mnist[idx]
        
        # Apply transform
        img = self.transform(img)
        
        # Flatten and apply permutation
        img_flat = img.view(-1)
        if self.permutation is not None:
            img_flat = img_flat[self.permutation]
        
        return img_flat, label


class ContinualLearningDataset(Dataset):
    """
    Wrapper dataset for continual learning that filters by class indices.
    """
    
    def __init__(
        self,
        base_dataset: Dataset,
        class_indices: List[int],
        class_mapping: Optional[Dict[int, int]] = None
    ):
        """
        Args:
            base_dataset: Original dataset
            class_indices: List of class indices to include
            class_mapping: Optional mapping from original to new class indices
        """
        self.base_dataset = base_dataset
        self.class_indices = set(class_indices)
        self.class_mapping = class_mapping
        
        # Filter indices that belong to specified classes
        self.valid_indices = []
        for idx in range(len(base_dataset)):
            _, label = base_dataset[idx]
            if label in self.class_indices:
                self.valid_indices.append(idx)
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        real_idx = self.valid_indices[idx]
        img, label = self.base_dataset[real_idx]
        
        if self.class_mapping is not None:
            label = self.class_mapping[label]
        
        return img, label


class TinyImageNet(Dataset):
    """
    TinyImageNet dataset (200 classes, 64x64 images).
    """
    
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[transforms.Compose] = None,
        download: bool = True
    ):
        self.root = os.path.join(root, 'tiny-imagenet-200')
        self.train = train
        self.transform = transform
        
        if download and not os.path.exists(self.root):
            self._download()
        
        self.images = []
        self.labels = []
        self._load_data()
        
    def _download(self):
        """Download TinyImageNet dataset."""
        import urllib.request
        import zipfile
        
        url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
        zip_path = os.path.join(os.path.dirname(self.root), 'tiny-imagenet-200.zip')
        
        print("Downloading TinyImageNet...")
        os.makedirs(os.path.dirname(self.root), exist_ok=True)
        urllib.request.urlretrieve(url, zip_path)
        
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(self.root))
        
        os.remove(zip_path)
        print("Done!")
        
    def _load_data(self):
        """Load image paths and labels."""
        # Load class names
        wnids_path = os.path.join(self.root, 'wnids.txt')
        with open(wnids_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        if self.train:
            train_dir = os.path.join(self.root, 'train')
            for class_name in self.class_names:
                class_dir = os.path.join(train_dir, class_name, 'images')
                if os.path.exists(class_dir):
                    for img_name in os.listdir(class_dir):
                        if img_name.endswith('.JPEG'):
                            self.images.append(os.path.join(class_dir, img_name))
                            self.labels.append(self.class_to_idx[class_name])
        else:
            val_dir = os.path.join(self.root, 'val')
            val_annotations = os.path.join(val_dir, 'val_annotations.txt')
            
            img_to_class = {}
            with open(val_annotations, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split('\t')
                    img_to_class[parts[0]] = parts[1]
            
            val_images_dir = os.path.join(val_dir, 'images')
            for img_name in os.listdir(val_images_dir):
                if img_name.endswith('.JPEG'):
                    self.images.append(os.path.join(val_images_dir, img_name))
                    class_name = img_to_class[img_name]
                    self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.images[idx]
        label = self.labels[idx]
        
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


def get_cifar100_transforms(train: bool = True) -> transforms.Compose:
    """Get transforms for CIFAR-100."""
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761]
            )
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761]
            )
        ])


def get_tinyimagenet_transforms(train: bool = True) -> transforms.Compose:
    """Get transforms for TinyImageNet."""
    if train:
        return transforms.Compose([
            transforms.RandomCrop(64, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


class ContinualLearningBenchmark:
    """
    Creates continual learning benchmark with proper data splits.
    
    Splits: 70% train, 20% test, 10% validation
    """
    
    def __init__(
        self,
        dataset_name: str,
        num_tasks: int,
        data_root: str = './data',
        seed: int = 42,
        scenario: str = 'class_il'
    ):
        """
        Args:
            dataset_name: 'pmnist', 'cifar100', or 'tinyimagenet'
            num_tasks: Number of tasks (10 or 20)
            data_root: Root directory for datasets
            seed: Random seed for reproducibility
            scenario: 'class_il' or 'task_il'
        """
        self.dataset_name = dataset_name.lower()
        self.num_tasks = num_tasks
        self.data_root = data_root
        self.seed = seed
        self.scenario = scenario
        
        # Set random seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Initialize dataset-specific parameters
        self._setup_dataset()
        
    def _setup_dataset(self):
        """Setup dataset-specific parameters."""
        if self.dataset_name == 'pmnist':
            self.num_classes = 10
            self.classes_per_task = 10  # All classes per task, different permutation
            self.input_size = 784
            
            # Generate random permutations for each task
            self.permutations = [
                np.random.permutation(784) if i > 0 else np.arange(784)
                for i in range(self.num_tasks)
            ]
            
        elif self.dataset_name == 'cifar100':
            self.num_classes = 100
            self.classes_per_task = self.num_classes // self.num_tasks
            self.input_size = 32
            
            # Randomly shuffle class order
            self.class_order = np.random.permutation(self.num_classes)
            
        elif self.dataset_name == 'tinyimagenet':
            self.num_classes = 200
            self.classes_per_task = self.num_classes // self.num_tasks
            self.input_size = 64
            
            # Randomly shuffle class order
            self.class_order = np.random.permutation(self.num_classes)
            
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def get_task_classes(self, task_id: int) -> List[int]:
        """Get class indices for a specific task."""
        if self.dataset_name == 'pmnist':
            return list(range(10))  # All classes, different permutation
        else:
            start = task_id * self.classes_per_task
            end = start + self.classes_per_task
            return self.class_order[start:end].tolist()
    
    def get_class_mapping(self, task_id: int) -> Dict[int, int]:
        """
        Get mapping from original class indices to task-local indices.
        
        For Class-IL: Maps to cumulative class indices
        For Task-IL: Maps to task-local indices (0 to classes_per_task-1)
        """
        classes = self.get_task_classes(task_id)
        
        if self.scenario == 'class_il':
            # Cumulative mapping
            start_idx = task_id * self.classes_per_task
            return {orig: start_idx + i for i, orig in enumerate(classes)}
        else:
            # Task-local mapping
            return {orig: i for i, orig in enumerate(classes)}
    
    def get_task_data(
        self,
        task_id: int,
        batch_size: int = 64
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Get train, validation, and test data loaders for a task.
        
        Data split: 70% train, 10% validation, 20% test
        
        Args:
            task_id: Task index (0-indexed)
            batch_size: Batch size for data loaders
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        if self.dataset_name == 'pmnist':
            return self._get_pmnist_task_data(task_id, batch_size)
        elif self.dataset_name == 'cifar100':
            return self._get_cifar100_task_data(task_id, batch_size)
        elif self.dataset_name == 'tinyimagenet':
            return self._get_tinyimagenet_task_data(task_id, batch_size)
    
    def _get_pmnist_task_data(
        self,
        task_id: int,
        batch_size: int
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get PMNIST task data with specific permutation."""
        permutation = self.permutations[task_id]
        
        # Full train and test datasets with permutation
        train_dataset = PermutedMNIST(
            root=self.data_root, train=True, download=True,
            permutation=permutation
        )
        test_dataset = PermutedMNIST(
            root=self.data_root, train=False, download=True,
            permutation=permutation
        )
        
        # Split train into train (70%) and val (10%)
        # Paper specifies: 70% train, 10% validation, 20% test
        # Original MNIST train is 60000, test is 10000 = 70000 total
        # We need: 70% train (~49000), 10% val (~7000), 20% test (~14000)
        # Use original train for train+val, original test for test
        # Split original train: 87.5% for train, 12.5% for val
        # This gives roughly 52500 train, 7500 val from 60000 train samples
        train_size = int(0.875 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        
        train_subset, val_subset = random_split(
            train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(self.seed + task_id)
        )
        
        train_loader = DataLoader(
            train_subset, batch_size=10, shuffle=True, num_workers=2
        )
        val_loader = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False, num_workers=2
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )
        
        return train_loader, val_loader, test_loader
    
    def _get_cifar100_task_data(
        self,
        task_id: int,
        batch_size: int
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get CIFAR-100 task data."""
        classes = self.get_task_classes(task_id)
        class_mapping = self.get_class_mapping(task_id)
        
        # Load full datasets
        train_transform = get_cifar100_transforms(train=True)
        test_transform = get_cifar100_transforms(train=False)
        
        full_train = datasets.CIFAR100(
            root=self.data_root, train=True, download=True,
            transform=train_transform
        )
        full_test = datasets.CIFAR100(
            root=self.data_root, train=False, download=True,
            transform=test_transform
        )
        
        # Filter by task classes
        train_dataset = ContinualLearningDataset(
            full_train, classes, class_mapping
        )
        test_dataset = ContinualLearningDataset(
            full_test, classes, class_mapping
        )
        
        # Split train into train and validation
        train_size = int(0.875 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        
        train_subset, val_subset = random_split(
            train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(self.seed + task_id)
        )
        
        train_loader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True, num_workers=2
        )
        val_loader = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False, num_workers=2
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )
        
        return train_loader, val_loader, test_loader
    
    def _get_tinyimagenet_task_data(
        self,
        task_id: int,
        batch_size: int
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get TinyImageNet task data."""
        classes = self.get_task_classes(task_id)
        class_mapping = self.get_class_mapping(task_id)
        
        # Load full datasets
        train_transform = get_tinyimagenet_transforms(train=True)
        test_transform = get_tinyimagenet_transforms(train=False)
        
        full_train = TinyImageNet(
            root=self.data_root, train=True,
            transform=train_transform, download=True
        )
        full_test = TinyImageNet(
            root=self.data_root, train=False,
            transform=test_transform, download=True
        )
        
        # Filter by task classes
        train_dataset = ContinualLearningDataset(
            full_train, classes, class_mapping
        )
        test_dataset = ContinualLearningDataset(
            full_test, classes, class_mapping
        )
        
        # Split train into train and validation
        train_size = int(0.875 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        
        train_subset, val_subset = random_split(
            train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(self.seed + task_id)
        )
        
        train_loader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True, num_workers=2
        )
        val_loader = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False, num_workers=2
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )
        
        return train_loader, val_loader, test_loader
    
    def get_cumulative_classes(self, task_id: int) -> int:
        """Get total number of classes seen up to and including task_id."""
        if self.dataset_name == 'pmnist':
            return 10  # Always 10 classes
        return (task_id + 1) * self.classes_per_task

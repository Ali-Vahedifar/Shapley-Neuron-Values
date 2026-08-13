"""
Continual-learning benchmarks.

Datasets: Permuted MNIST, CIFAR-100, TinyImageNet, ImageNet-1k.
Task splits: 10, 20 or 50 tasks (ImageNet-1k is the only one evaluated at 50).

Splits follow the experimental setup: for every task the samples of that task's
classes are pooled from the dataset's native train and test partitions and then
re-split 70 % train / 10 % validation / 20 % test.  Re-pooling matters -- the
native partitions are not in that ratio (TinyImageNet ships 500 train and 50
validation images per class, i.e. 91 / 9), so slicing the native train set alone
cannot produce the stated proportions.

Labels are remapped per scenario: cumulative global indices for CIL, task-local
indices 0..C-1 for TIL.
"""

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.10, 0.20


# --------------------------------------------------------------------------- #
# Raw pools (no transform; transforms are applied by the subset wrapper)
# --------------------------------------------------------------------------- #
class _ConcatPool:
    """Two raw datasets viewed as one, with a single ``targets`` array."""

    def __init__(self, first, second, first_targets, second_targets):
        self.first, self.second = first, second
        self.offset = len(first)
        self.targets = np.concatenate([np.asarray(first_targets), np.asarray(second_targets)])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return self.first[i] if i < self.offset else self.second[i - self.offset]


class _TaskSubset(Dataset):
    """A task's slice of a pool: applies the transform, remaps the label."""

    def __init__(self, pool, indices: np.ndarray, transform, label_map: Dict[int, int],
                 permutation: Optional[np.ndarray] = None):
        self.pool = pool
        self.indices = indices
        self.transform = transform
        self.label_map = label_map
        self.permutation = permutation

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, label = self.pool[int(self.indices[i])]
        if self.transform is not None:
            img = self.transform(img)
        if self.permutation is not None:
            img = img.reshape(-1)[self.permutation]
        return img, self.label_map[int(label)]


class TinyImageNet:
    """TinyImageNet-200 as a raw pool of (PIL, label)."""

    URL = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'

    def __init__(self, root: str, train: bool = True, download: bool = True):
        self.root = os.path.join(root, 'tiny-imagenet-200')
        if download and not os.path.exists(self.root):
            self._download(root)
        self.paths: List[str] = []
        labels: List[int] = []

        with open(os.path.join(self.root, 'wnids.txt')) as f:
            wnids = [line.strip() for line in f if line.strip()]
        self.class_to_idx = {w: i for i, w in enumerate(wnids)}

        if train:
            for wnid in wnids:
                d = os.path.join(self.root, 'train', wnid, 'images')
                if not os.path.isdir(d):
                    continue
                for fn in sorted(os.listdir(d)):
                    if fn.endswith('.JPEG'):
                        self.paths.append(os.path.join(d, fn))
                        labels.append(self.class_to_idx[wnid])
        else:
            ann = os.path.join(self.root, 'val', 'val_annotations.txt')
            mapping = {}
            with open(ann) as f:
                for line in f:
                    parts = line.strip().split('\t')
                    mapping[parts[0]] = parts[1]
            d = os.path.join(self.root, 'val', 'images')
            for fn in sorted(os.listdir(d)):
                if fn.endswith('.JPEG'):
                    self.paths.append(os.path.join(d, fn))
                    labels.append(self.class_to_idx[mapping[fn]])

        self.targets = np.asarray(labels)

    def _download(self, root):
        import urllib.request, zipfile
        os.makedirs(root, exist_ok=True)
        zip_path = os.path.join(root, 'tiny-imagenet-200.zip')
        print('Downloading TinyImageNet-200 ...')
        urllib.request.urlretrieve(self.URL, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(root)
        os.remove(zip_path)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return Image.open(self.paths[i]).convert('RGB'), int(self.targets[i])


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
_STATS = {
    'cifar100': ([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
    'tinyimagenet': ([0.4802, 0.4481, 0.3975], [0.2770, 0.2691, 0.2821]),
    'imagenet1k': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
}


def build_transforms(dataset: str, train: bool):
    if dataset == 'pmnist':
        return transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))])
    mean, std = _STATS[dataset]
    norm = transforms.Normalize(mean, std)
    if dataset == 'cifar100':
        aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
        base = []
    elif dataset == 'tinyimagenet':
        aug = [transforms.RandomCrop(64, padding=8), transforms.RandomHorizontalFlip()]
        base = []
    else:  # imagenet1k
        aug = [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()]
        base = [transforms.Resize(256), transforms.CenterCrop(224)]
    steps = (aug if train else base) + [transforms.ToTensor(), norm]
    return transforms.Compose(steps)


# --------------------------------------------------------------------------- #
# Benchmark
# --------------------------------------------------------------------------- #
class ContinualLearningBenchmark:
    """Sequential tasks with 70 / 10 / 20 splits."""

    NUM_CLASSES = {'pmnist': 10, 'cifar100': 100, 'tinyimagenet': 200, 'imagenet1k': 1000}

    def __init__(self, dataset_name: str, num_tasks: int, data_root: str = './data',
                 seed: int = 42, scenario: str = 'class_il', num_workers: int = 4,
                 download: bool = True):
        self.dataset_name = dataset_name.lower()
        if self.dataset_name not in self.NUM_CLASSES:
            raise ValueError(f'unknown dataset {dataset_name!r}')
        self.num_tasks = num_tasks
        self.data_root = data_root
        self.seed = seed
        self.scenario = scenario
        self.num_workers = num_workers
        self.download = download

        self.num_classes = self.NUM_CLASSES[self.dataset_name]
        rng = np.random.RandomState(seed)

        if self.dataset_name == 'pmnist':
            self.classes_per_task = 10
            self.permutations = [np.arange(784) if t == 0 else rng.permutation(784)
                                 for t in range(num_tasks)]
            self.class_order = np.arange(10)
        else:
            if self.num_classes % num_tasks:
                raise ValueError(
                    f'{self.num_classes} classes do not divide evenly into {num_tasks} tasks')
            self.classes_per_task = self.num_classes // num_tasks
            self.permutations = [None] * num_tasks
            self.class_order = rng.permutation(self.num_classes)

        self._pool = None
        self._split_cache: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    # -- pool --------------------------------------------------------------- #
    def _get_pool(self):
        if self._pool is not None:
            return self._pool
        name, root = self.dataset_name, self.data_root

        if name == 'pmnist':
            tr = datasets.MNIST(root, train=True, download=self.download)
            te = datasets.MNIST(root, train=False, download=self.download)
            pool = _ConcatPool(tr, te, tr.targets.numpy(), te.targets.numpy())
        elif name == 'cifar100':
            tr = datasets.CIFAR100(root, train=True, download=self.download)
            te = datasets.CIFAR100(root, train=False, download=self.download)
            pool = _ConcatPool(tr, te, tr.targets, te.targets)
        elif name == 'tinyimagenet':
            tr = TinyImageNet(root, train=True, download=self.download)
            te = TinyImageNet(root, train=False, download=self.download)
            pool = _ConcatPool(tr, te, tr.targets, te.targets)
        else:
            base = os.path.join(root, 'imagenet')
            train_dir, val_dir = os.path.join(base, 'train'), os.path.join(base, 'val')
            if not (os.path.isdir(train_dir) and os.path.isdir(val_dir)):
                raise FileNotFoundError(
                    f'ImageNet-1k not found. Expected ImageFolder layouts at\n'
                    f'  {train_dir}\n  {val_dir}\n'
                    'ImageNet cannot be downloaded automatically; fetch it from '
                    'https://image-net.org and arrange it as class-per-directory.')
            tr = datasets.ImageFolder(train_dir)
            te = datasets.ImageFolder(val_dir)
            pool = _ConcatPool(tr, te, tr.targets, te.targets)

        self._pool = pool
        return pool

    # -- task definition ---------------------------------------------------- #
    def get_task_classes(self, task_id: int) -> List[int]:
        if self.dataset_name == 'pmnist':
            return list(range(10))
        start = task_id * self.classes_per_task
        return self.class_order[start:start + self.classes_per_task].tolist()

    def get_class_mapping(self, task_id: int) -> Dict[int, int]:
        classes = self.get_task_classes(task_id)
        if self.scenario == 'class_il':
            base = task_id * self.classes_per_task
            return {int(c): base + i for i, c in enumerate(classes)}
        return {int(c): i for i, c in enumerate(classes)}

    def _split_indices(self, task_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if task_id in self._split_cache:
            return self._split_cache[task_id]
        pool = self._get_pool()
        classes = np.asarray(self.get_task_classes(task_id))
        idx = np.nonzero(np.isin(pool.targets, classes))[0]
        rng = np.random.RandomState(self.seed * 1000 + task_id)
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(TRAIN_FRAC * n))
        n_val = int(round(VAL_FRAC * n))
        split = (idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:])
        self._split_cache[task_id] = split
        return split

    # -- loaders ------------------------------------------------------------ #
    def get_task_data(self, task_id: int, batch_size: int = 64
                      ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        pool = self._get_pool()
        train_idx, val_idx, test_idx = self._split_indices(task_id)
        mapping = self.get_class_mapping(task_id)
        perm = self.permutations[task_id]

        tf_train = build_transforms(self.dataset_name, train=True)
        tf_eval = build_transforms(self.dataset_name, train=False)

        def make(indices, transform, shuffle):
            ds = _TaskSubset(pool, indices, transform, mapping, perm)
            return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                              num_workers=self.num_workers,
                              pin_memory=torch.cuda.is_available(),
                              persistent_workers=self.num_workers > 0)

        return (make(train_idx, tf_train, True),
                make(val_idx, tf_eval, False),
                make(test_idx, tf_eval, False))

    def get_joint_data(self, batch_size: int = 64) -> Tuple[DataLoader, DataLoader]:
        """All tasks at once -- the joint-training upper bound."""
        pool = self._get_pool()
        mapping: Dict[int, int] = {}
        train_all, test_all = [], []
        for t in range(self.num_tasks):
            tr, va, te = self._split_indices(t)
            train_all += [tr, va]
            test_all.append(te)
            mapping.update(self.get_class_mapping(t))
        tf_train = build_transforms(self.dataset_name, train=True)
        tf_eval = build_transforms(self.dataset_name, train=False)
        train_ds = _TaskSubset(pool, np.concatenate(train_all), tf_train, mapping,
                               self.permutations[0])
        test_ds = _TaskSubset(pool, np.concatenate(test_all), tf_eval, mapping,
                              self.permutations[0])
        kw = dict(batch_size=batch_size, num_workers=self.num_workers,
                  pin_memory=torch.cuda.is_available())
        return DataLoader(train_ds, shuffle=True, **kw), DataLoader(test_ds, shuffle=False, **kw)

    def get_cumulative_classes(self, task_id: int) -> int:
        if self.dataset_name == 'pmnist':
            return 10
        return (task_id + 1) * self.classes_per_task

    def split_report(self, task_id: int = 0) -> Dict[str, float]:
        tr, va, te = self._split_indices(task_id)
        n = len(tr) + len(va) + len(te)
        return {'train': len(tr) / n, 'val': len(va) / n, 'test': len(te) / n, 'n': n}

# datasets.py

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data.sampler import Sampler
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np
import gzip
import random
import tarfile
from typing import Optional, Sequence, Union
from utils import apply_transform

DATASET_NORMALIZATION_STATS = {
    "cifar10": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2023, 0.1994, 0.2010),
    },
    "cifar100": {
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
    },
    "emnist_digits": {
        "mean": (0.1307,),
        "std": (0.3081,),
    },
}

DEFAULT_TRANSFORM_SPECS = {
    "cifar10": {
        "train": [
            {"name": "RandomCrop", "params": {"size": 32, "padding": 4}},
            {"name": "RandomHorizontalFlip", "params": {}},
            {"name": "Normalize", "params": {"mean": DATASET_NORMALIZATION_STATS["cifar10"]["mean"],
                                             "std": DATASET_NORMALIZATION_STATS["cifar10"]["std"]}},
        ],
        "test": [
            {"name": "Normalize", "params": {"mean": DATASET_NORMALIZATION_STATS["cifar10"]["mean"],
                                             "std": DATASET_NORMALIZATION_STATS["cifar10"]["std"]}},
        ],
    },
    "cifar100": {
        "train": [
            {"name": "RandomCrop", "params": {"size": 32, "padding": 4}},
            {"name": "RandomHorizontalFlip", "params": {}},
            {"name": "Normalize", "params": {"mean": DATASET_NORMALIZATION_STATS["cifar100"]["mean"],
                                             "std": DATASET_NORMALIZATION_STATS["cifar100"]["std"]}},
        ],
        "test": [
            {"name": "Normalize", "params": {"mean": DATASET_NORMALIZATION_STATS["cifar100"]["mean"],
                                             "std": DATASET_NORMALIZATION_STATS["cifar100"]["std"]}},
        ],
    },
    "emnist_digits": {
        "train": [
            {"name": "Resize", "params": {"size": 32}},
            {"name": "Normalize", "params": {"mean": DATASET_NORMALIZATION_STATS["emnist_digits"]["mean"],
                                             "std": DATASET_NORMALIZATION_STATS["emnist_digits"]["std"]}},
        ],
        "test": [
            {"name": "Resize", "params": {"size": 32}},
            {"name": "Normalize", "params": {"mean": DATASET_NORMALIZATION_STATS["emnist_digits"]["mean"],
                                             "std": DATASET_NORMALIZATION_STATS["emnist_digits"]["std"]}},
        ],
    },
}

TRANSFORM_REGISTRY = {
    "RandomCrop": transforms.RandomCrop,
    "RandomHorizontalFlip": transforms.RandomHorizontalFlip,
    "RandomRotation": transforms.RandomRotation,
    "ColorJitter": transforms.ColorJitter,
    "GaussianBlur": transforms.GaussianBlur,
    "Normalize": transforms.Normalize,
    "Resize": transforms.Resize,
    "CenterCrop": transforms.CenterCrop,
    "RandomResizedCrop": transforms.RandomResizedCrop,
    "Grayscale": transforms.Grayscale,
}

def _describe_transform(transform) -> str:
    if transform is None:
        return "None"
    if isinstance(transform, transforms.Compose):
        return " -> ".join(type(t).__name__ for t in transform.transforms)
    return type(transform).__name__


def format_noise_rate(noise_rate: float) -> str:
    """Format noise rate for directory/file naming."""
    rate_str = f"{noise_rate:.10g}"
    return rate_str.replace(".", "_") if "." in rate_str and rate_str.endswith(".0") else rate_str


def _load_raw_cifar10():
    data_root = './data'
    cifar_train = datasets.CIFAR10(root=data_root, train=True, download=True)
    x_train = torch.from_numpy(cifar_train.data).permute(0, 3, 1, 2).float() / 255.0
    y_train = torch.tensor(cifar_train.targets, dtype=torch.long)

    cifar_test = datasets.CIFAR10(root=data_root, train=False, download=True)
    x_test = torch.from_numpy(cifar_test.data).permute(0, 3, 1, 2).float() / 255.0
    y_test = torch.tensor(cifar_test.targets, dtype=torch.long)

    return TensorDataset(x_train, y_train), TensorDataset(x_test, y_test)


def _load_raw_cifar100():
    data_root = './data'
    cifar_train = datasets.CIFAR100(root=data_root, train=True, download=True)
    x_train = torch.from_numpy(cifar_train.data).permute(0, 3, 1, 2).float() / 255.0
    y_train = torch.tensor(cifar_train.targets, dtype=torch.long)

    cifar_test = datasets.CIFAR100(root=data_root, train=False, download=True)
    x_test = torch.from_numpy(cifar_test.data).permute(0, 3, 1, 2).float() / 255.0
    y_test = torch.tensor(cifar_test.targets, dtype=torch.long)

    return TensorDataset(x_train, y_train), TensorDataset(x_test, y_test)


def _load_raw_emnist_digits():
    data_root = './data/EMNIST'
    image_files = {
        "train": os.path.join(data_root, "emnist-digits-train-images-idx3-ubyte.gz"),
        "test": os.path.join(data_root, "emnist-digits-test-images-idx3-ubyte.gz"),
    }
    label_files = {
        "train": os.path.join(data_root, "emnist-digits-train-labels-idx1-ubyte.gz"),
        "test": os.path.join(data_root, "emnist-digits-test-labels-idx1-ubyte.gz"),
    }

    def _load_gz(path, is_image):
        with gzip.open(path, 'rb') as f:
            if is_image:
                data = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(-1, 28, 28)
            else:
                data = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
        return data

    try:
        x_train = _load_gz(image_files["train"], is_image=True).copy()
        y_train = _load_gz(label_files["train"], is_image=False).copy()
        x_test = _load_gz(image_files["test"], is_image=True).copy()
        y_test = _load_gz(label_files["test"], is_image=False).copy()
    except FileNotFoundError:
        # Fallback to torchvision dataset download if local files are unavailable.
        torchvision_root = './data'
        train_ds = datasets.EMNIST(root=torchvision_root, split='digits', train=True, download=True)
        test_ds = datasets.EMNIST(root=torchvision_root, split='digits', train=False, download=True)
        x_train = train_ds.data.numpy().copy()
        y_train = train_ds.targets.numpy().copy()
        x_test = test_ds.data.numpy().copy()
        y_test = test_ds.targets.numpy().copy()
    except Exception as exc:
        raise RuntimeError(f"Failed to load EMNIST digits data: {exc}") from exc

    x_train_tensor = torch.from_numpy(x_train).unsqueeze(1).float() / 255.0
    y_train_tensor = torch.from_numpy(y_train).long()

    x_test_tensor = torch.from_numpy(x_test).unsqueeze(1).float() / 255.0
    y_test_tensor = torch.from_numpy(y_test).long()

    return TensorDataset(x_train_tensor, y_train_tensor), TensorDataset(x_test_tensor, y_test_tensor)


DATASET_CACHE_CONFIG = {
    "cifar10": {
        "base_dir": "/workspace/data/cifar10",
        "noise_subdir": "cifar10_noise_{rate}",
        "transform_path": "/workspace/data/cifar10/transform.json",
        "raw_loader": _load_raw_cifar10,
        "imagesize": (32, 32),
        "num_classes": 10,
        "in_channels": 3,
        "num_digits": 10,
        "num_colors": 1,
        "default_transforms": DEFAULT_TRANSFORM_SPECS["cifar10"],
    },
    "cifar100": {
        "base_dir": "/workspace/data/cifar100",
        "noise_subdir": "cifar100_noise_{rate}",
        "transform_path": "/workspace/data/cifar100/transform.json",
        "raw_loader": _load_raw_cifar100,
        "imagesize": (32, 32),
        "num_classes": 100,
        "in_channels": 3,
        "num_digits": 100,
        "num_colors": 1,
        "default_transforms": DEFAULT_TRANSFORM_SPECS["cifar100"],
    },
    "emnist_digits": {
        "base_dir": "/workspace/data/EMNIST",
        "noise_subdir": "emnist_digits_noise_{rate}",
        "transform_path": "/workspace/data/EMNIST/transform_digits.json",
        "raw_loader": _load_raw_emnist_digits,
        "imagesize": (32, 32),
        "num_classes": 10,
        "in_channels": 1,
        "num_digits": 10,
        "num_colors": 1,
        "default_transforms": DEFAULT_TRANSFORM_SPECS["emnist_digits"],
    },
}


def _normalize_spec_if_missing(spec, dataset_key):
    stats = DATASET_NORMALIZATION_STATS.get(dataset_key)
    if stats is None:
        return list(spec)
    has_normalize = any(step.get("name") == "Normalize" for step in spec)
    if has_normalize:
        return list(spec)
    mean = stats["mean"]
    std = stats["std"]
    spec_with_norm = list(spec) + [{"name": "Normalize", "params": {"mean": mean, "std": std}}]
    return spec_with_norm


def _ensure_transform_file(dataset_key, config):
    path = config["transform_path"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        default_spec = config.get("default_transforms", {"train": [], "test": []})
        with open(path, "w") as f:
            json.dump(default_spec, f, indent=2)
    return path


def _build_transform_from_spec(spec, dataset_key):
    if not spec:
        return None
    transform_objects = []
    for entry in spec:
        name = entry.get("name")
        params = entry.get("params", {})
        if name not in TRANSFORM_REGISTRY:
            print(f"Warning: transform '{name}' は未登録です。スキップします。")
            continue
        transform_cls = TRANSFORM_REGISTRY[name]
        transform_objects.append(transform_cls(**params))
    if not transform_objects:
        return None
    return transforms.Compose(transform_objects)


def load_transforms_for_dataset(dataset_key, config):
    path = _ensure_transform_file(dataset_key, config)
    with open(path, "r") as f:
        spec_data = json.load(f)

    default_specs = config.get("default_transforms", {"train": [], "test": []})

    if "train" in spec_data or "test" in spec_data:
        train_spec = spec_data.get("train", default_specs.get("train", []))
        test_spec = spec_data.get("test", default_specs.get("test", spec_data.get("train", [])))
    elif "transforms" in spec_data:
        train_spec = spec_data["transforms"]
        test_spec = default_specs.get("test", train_spec)
    else:
        train_spec = default_specs.get("train", [])
        test_spec = default_specs.get("test", train_spec)

    train_spec = _normalize_spec_if_missing(train_spec, dataset_key)
    test_spec = _normalize_spec_if_missing(test_spec, dataset_key)

    return (
        _build_transform_from_spec(train_spec, dataset_key),
        _build_transform_from_spec(test_spec, dataset_key),
    )
class NoisyDataset(Dataset):
    """
    Custom dataset that includes noise information.
    """
    def __init__(self, dataset, noise_info, transform=None):
        self.dataset = dataset
        self.noise_info = noise_info
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        input, label = self.dataset[idx]
        if isinstance(input, torch.Tensor):
            input = input.clone()
        if self.transform is not None:
            input = self.transform(input)
        noise_label = self.noise_info[idx]
        return input, label, noise_label

    def set_transform(self, transform):
        self.transform = transform

class BalancedBatchSampler(Sampler):
    """
    Custom sampler to create balanced batches of clean and noisy samples.
    """
    def __init__(self, clean_indices, noisy_indices, batch_size, drop_last):
        self.clean_indices = clean_indices
        self.noisy_indices = noisy_indices
        self.batch_size = batch_size
        self.drop_last = drop_last

        assert batch_size % 2 == 0, "Batch size must be even for balanced batches"
        self.num_samples_per_class = batch_size // 2

    def __iter__(self):
        # Shuffle the indices
        random.shuffle(self.clean_indices)
        random.shuffle(self.noisy_indices)

        # Calculate the number of batches
        min_len = min(len(self.clean_indices), len(self.noisy_indices))
        num_batches = min_len // self.num_samples_per_class

        for i in range(num_batches):
            clean_batch = self.clean_indices[i * self.num_samples_per_class: (i + 1) * self.num_samples_per_class]
            noisy_batch = self.noisy_indices[i * self.num_samples_per_class: (i + 1) * self.num_samples_per_class]
            batch = clean_batch + noisy_batch
            random.shuffle(batch)
            yield batch

        if not self.drop_last:
            # Handle remaining samples
            remaining_clean = self.clean_indices[num_batches * self.num_samples_per_class:]
            remaining_noisy = self.noisy_indices[num_batches * self.num_samples_per_class:]

            if len(remaining_clean) >= self.num_samples_per_class and len(remaining_noisy) >= self.num_samples_per_class:
                batch = remaining_clean[:self.num_samples_per_class] + remaining_noisy[:self.num_samples_per_class]
                random.shuffle(batch)
                yield batch

def load_datasets(dataset, target, gray_scale, args):
    """Load dataset with optional grayscale conversion.

    Parameters
    ----------
    dataset : str
        Dataset name.
    target : str
        Target type for special datasets such as ``colored_emnist``.
    gray_scale : bool
        Convert images to gray scale when ``True``.
    args : argparse.Namespace
        Additional command line arguments.

    Returns
    -------
    tuple
        ``(train_dataset, test_dataset, imagesize, num_classes, in_channels)``
    """

    dataset = dataset.lower()

    def _load_mnist(_, __, ___):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        def loader():
            train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
            test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
            return train, test

        # train_ds, test_ds = _safe_load(loader, (32, 32), 10, 1)
        return train_ds, test_ds, (32, 32), 10, 1

    def _load_emnist(_, __, ___):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        def loader():
            train = datasets.EMNIST(root='./data', split='balanced', train=True, download=True, transform=transform)
            test = datasets.EMNIST(root='./data', split='balanced', train=False, download=True, transform=transform)
            return train, test

        train_ds, test_ds = _safe_load(loader, (32, 32), 47, 1)
        return train_ds, test_ds, (32, 32), 47, 1

    def _load_emnist_digits(_, __, ___):
        emnist_path = './data/EMNIST'

        def load_gz_file(file_path, is_image=True):
            with gzip.open(file_path, 'rb') as f:
                if is_image:
                    return np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(-1, 28, 28)
                return np.frombuffer(f.read(), dtype=np.uint8, offset=8)

        try:
            print
            x_train = load_gz_file(os.path.join(emnist_path, 'emnist-digits-train-images-idx3-ubyte.gz'))
            print("x_train shape:", x_train.shape)
            y_train = load_gz_file(os.path.join(emnist_path, 'emnist-digits-train-labels-idx1-ubyte.gz'), is_image=False)
            print("y_train shape:", y_train.shape)
            x_test = load_gz_file(os.path.join(emnist_path, 'emnist-digits-test-images-idx3-ubyte.gz'))
            y_test = load_gz_file(os.path.join(emnist_path, 'emnist-digits-test-labels-idx1-ubyte.gz'), is_image=False)
        except Exception as exc:
            print(f"Warning: failed to load EMNIST digits ({exc}). Using dummy data.")
            # train_ds = _create_dummy_dataset(1000, (32, 32), 10, 1)
            # test_ds = _create_dummy_dataset(200, (32, 32), 10, 1)
            # return train_ds, test_ds, (32, 32), 10, 1

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        x_train_tensor = apply_transform(x_train, transform)
        x_test_tensor = apply_transform(x_test, transform)

        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        train_ds = TensorDataset(x_train_tensor, y_train_tensor)
        test_ds = TensorDataset(x_test_tensor, y_test_tensor)
        return train_ds, test_ds, (32, 32), 10, 1

    def _load_colored_emnist(target_local, _, args_local):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        try:
            x_train = np.load('data/colored_EMNIST/x_train_colored.npy')
            x_test = np.load('data/colored_EMNIST/x_test_colored.npy')

            if target_local == 'color':
                y_train = np.load('data/colored_EMNIST/y_train_colors.npy')
                y_test = np.load('data/colored_EMNIST/y_test_colors.npy')
            elif target_local == 'digit':
                y_train = np.load('data/colored_EMNIST/y_train_digits.npy')
                y_test = np.load('data/colored_EMNIST/y_test_digits.npy')
            else:
                y_train = np.load('data/colored_EMNIST/y_train_combined.npy')
                y_test = np.load('data/colored_EMNIST/y_test_combined.npy')
        except Exception as exc:
            print(f"Warning: failed to load colored EMNIST ({exc}). Using dummy data.")
            # num_classes = 10 if target_local in ['color', 'digit'] else 100
            # train_ds = _create_dummy_dataset(1000, (32, 32), num_classes, 3)
            # test_ds = _create_dummy_dataset(200, (32, 32), num_classes, 3)
            # return train_ds, test_ds, (32, 32), num_classes, 3

        x_train_tensor = apply_transform(x_train, transform)
        x_test_tensor = apply_transform(x_test, transform)

        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        train_ds = TensorDataset(x_train_tensor, y_train_tensor)
        test_ds = TensorDataset(x_test_tensor, y_test_tensor)

        num_classes = 10 if target_local in ['color', 'digit'] else 100
        return train_ds, test_ds, (32, 32), num_classes, 3

    def _load_distribution_colored_emnist(target_local, _, args_local):
        seed = args_local.fix_seed
        variance = args_local.variance
        correlation = args_local.correlation

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        base_path = f'data/distribution_colored_EMNIST_Seed{seed}_Var{variance}_Corr{correlation}'
        try:
            x_train = np.load(os.path.join(base_path, 'x_train_colored.npy'))
            x_test = np.load(os.path.join(base_path, 'x_test_colored.npy'))

            if target_local == 'color':
                y_train = np.load(os.path.join(base_path, 'y_train_colors.npy'))
                y_test = np.load(os.path.join(base_path, 'y_test_colors.npy'))
            elif target_local == 'digit':
                y_train = np.load(os.path.join(base_path, 'y_train_digits.npy'))
                y_test = np.load(os.path.join(base_path, 'y_test_digits.npy'))
            else:
                y_train = np.load(os.path.join(base_path, 'y_train_combined.npy'))
                y_test = np.load(os.path.join(base_path, 'y_test_combined.npy'))
        except Exception as exc:
            print(f"Warning: failed to load distribution colored EMNIST ({exc}). Using dummy data.")
            # num_classes = 10 if target_local in ['color', 'digit'] else 100
            # train_ds = _create_dummy_dataset(1000, (32, 32), num_classes, 3)
            # test_ds = _create_dummy_dataset(200, (32, 32), num_classes, 3)
            # return train_ds, test_ds, (32, 32), num_classes, 3

        x_train_tensor = apply_transform(x_train, transform)
        x_test_tensor = apply_transform(x_test, transform)

        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        train_ds = TensorDataset(x_train_tensor, y_train_tensor)
        test_ds = TensorDataset(x_test_tensor, y_test_tensor)

        num_classes = 10 if target_local in ['color', 'digit'] else 100
        return train_ds, test_ds, (32, 32), num_classes, 3

    def _load_cifar10(_, __, ___):
        data_root = './data'
        archive_path = os.path.join(data_root, 'cifar-10-python.tar.gz')
        extracted_dir = os.path.join(data_root, 'cifar-10-batches-py')

        if not os.path.isdir(extracted_dir):
            if not os.path.isfile(archive_path):
                raise FileNotFoundError(f"CIFAR-10 archive not found at {archive_path}")

            def _is_within_directory(directory, target):
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
                return os.path.commonpath([abs_directory, abs_target]) == abs_directory

            with tarfile.open(archive_path, 'r:gz') as tar:
                for member in tar.getmembers():
                    member_path = os.path.join(data_root, member.name)
                    if not _is_within_directory(data_root, member_path):
                        raise RuntimeError(f"Unsafe path detected in archive member {member.name}")
                tar.extractall(path=data_root)

        # 0-1 range float tensors without any augmentation/normalization
        cifar_train = datasets.CIFAR10(root=data_root, train=True, download=False)
        x_train = torch.from_numpy(cifar_train.data).permute(0, 3, 1, 2).float() / 255.0
        y_train = torch.tensor(cifar_train.targets, dtype=torch.long)
        train_ds = TensorDataset(x_train, y_train)

        cifar_test = datasets.CIFAR10(root=data_root, train=False, download=False)
        x_test = torch.from_numpy(cifar_test.data).permute(0, 3, 1, 2).float() / 255.0
        y_test = torch.tensor(cifar_test.targets, dtype=torch.long)
        test_ds = TensorDataset(x_test, y_test)

        return train_ds, test_ds, (32, 32), 10, 3

    def _load_cifar100(_, __, ___):
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        def loader():
            train = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
            test = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
            return train, test

        train_ds, test_ds = _safe_load(loader, (32, 32), 100, 3)
        return train_ds, test_ds, (32, 32), 100, 3

    def _load_tiny_imagenet(_, __, ___):
        transform = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))
        ])
        def loader():
            train = datasets.ImageFolder(root='./data/tiny-imagenet-200/train', transform=transform)
            test = datasets.ImageFolder(root='./data/tiny-imagenet-200/val', transform=transform)
            return train, test

        train_ds, test_ds = _safe_load(loader, (64, 64), 200, 3)
        return train_ds, test_ds, (64, 64), 200, 3

    def _load_distribution_to_normal(target_local, _, args_local):
        seed = args_local.fix_seed
        variance = args_local.variance
        correlation = args_local.correlation

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        base_path = f'data/distribution_colored_EMNIST_Seed{seed}_Var{variance}_Corr{correlation}'
        try:
            x_train = np.load(os.path.join(base_path, 'x_train_colored.npy'))
            y_train = np.load(os.path.join(base_path, 'y_train_combined.npy'))
            x_test = np.load('data/colored_EMNIST/x_test_colored.npy')
            y_test = np.load('data/colored_EMNIST/y_test_combined.npy')
        except Exception as exc:
            print(f"Warning: failed to load distribution_to_normal ({exc}). Using dummy data.")
            # train_ds = _create_dummy_dataset(1000, (32, 32), 100, 3)
            # test_ds = _create_dummy_dataset(200, (32, 32), 100, 3)
            # return train_ds, test_ds, (32, 32), 100, 3

        x_train_tensor = apply_transform(x_train, transform)
        x_test_tensor = apply_transform(x_test, transform)

        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        train_ds = TensorDataset(x_train_tensor, y_train_tensor)
        test_ds = TensorDataset(x_test_tensor, y_test_tensor)
        return train_ds, test_ds, (32, 32), 100, 3

    loader_map = {
        'mnist': _load_mnist,
        'emnist': _load_emnist,
        'emnist_digits': _load_emnist_digits,
        'colored_emnist': _load_colored_emnist,
        'distribution_colored_emnist': _load_distribution_colored_emnist,
        'cifar10': _load_cifar10,
        'cifar100': _load_cifar100,
        'tinyimagenet': _load_tiny_imagenet,
        'distribution_to_normal': _load_distribution_to_normal,
    }

    if dataset not in loader_map:
        raise ValueError(f"Invalid dataset name: {dataset}")

    train_dataset, test_dataset, imagesize, num_classes, in_channels = loader_map[dataset](target, gray_scale, args)

    if gray_scale:
        gs_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        if isinstance(train_dataset, TensorDataset):
            train_dataset = TensorDataset(*[
                gs_transform(img) for img in train_dataset.tensors[0]
            ], train_dataset.tensors[1])
        if isinstance(test_dataset, TensorDataset):
            test_dataset = TensorDataset(*[
                gs_transform(img) for img in test_dataset.tensors[0]
            ], test_dataset.tensors[1])

    return train_dataset, test_dataset, imagesize, num_classes, in_channels


import os
import torch
import random
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, Dataset

# --- 既存の関数群 ---
def add_label_noise(targets, label_noise_rate, num_digits, num_colors):
    """
    Add label noise to the targets.
    """
    noisy_targets = targets.clone()
    num_noisy = int(label_noise_rate * len(targets))
    noisy_indices = torch.randperm(len(targets))[:num_noisy]
    noise_info = torch.zeros(len(targets), dtype=torch.int)  # Initialize as clean

    if num_digits == 10 and num_colors == 1:
        for idx in noisy_indices:
            original_label = targets[idx].item()
            new_label = random.randint(0, num_digits - 1)
            while new_label == original_label:
                new_label = random.randint(0, num_digits - 1)
            noisy_targets[idx] = new_label
            noise_info[idx] = 1  # Mark as noisy

    elif num_digits == 10 and num_colors == 10:
        for idx in noisy_indices:
            original_label = targets[idx].item()
            original_digit = original_label // num_colors
            original_color = original_label % num_colors

            new_digit = random.randint(0, num_digits - 1)
            new_color = random.randint(0, num_colors - 1)
            new_label = new_digit * num_colors + new_color
            while new_label == original_label:
                new_digit = random.randint(0, num_digits - 1)
                new_color = random.randint(0, num_colors - 1)
                new_label = new_digit * num_colors + new_color

            noisy_targets[idx] = new_label
            noise_info[idx] = 1  # Mark as noisy

    return noisy_targets, noise_info


def add_targeted_label_noise(
    targets: torch.Tensor,
    total_noise_rate: float,
    class_subset: Union[int, Sequence[int]],
    num_classes: int = 10,
    random_state: Optional[int] = None,
):
    """
    Add label noise by restricting perturbations to a subset of classes while matching
    a desired overall noise rate.

    Args:
        targets (torch.Tensor): 1D tensor of original labels.
        total_noise_rate (float): Desired noisy proportion across the entire dataset (0.0〜1.0).
        class_subset (Union[int, Sequence[int]]): Class ID or list of class IDs that are eligible
            for label corruption.
        num_classes (int): Total number of classes in the dataset.
        random_state (Optional[int]): Seed for the internal RNG.

        The function automatically computes the per-class corruption rate so that
        `total_noise_rate` is achieved overall. For a balanced dataset, this matches
        `total_noise_rate * num_classes / len(class_subset)`.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, dict]:
            - noisy_targets: Tensor of labels after noise injection.
            - noise_info: Tensor marking whether each sample was noised (1) or clean (0).
            - meta: Dictionary with bookkeeping information such as achieved noise rates.
    """
    if targets.dim() != 1:
        raise ValueError("targets must be a 1D tensor of class indices.")

    if not 0.0 <= total_noise_rate <= 1.0:
        raise ValueError("total_noise_rate must be within [0.0, 1.0].")

    if isinstance(class_subset, int):
        parsed_subset = [class_subset]
    else:
        parsed_subset = list(class_subset)

    if len(parsed_subset) == 0:
        raise ValueError("class_subset must contain at least one class index.")

    unique_subset = sorted(set(int(cls) for cls in parsed_subset))
    for cls in unique_subset:
        if cls < 0 or cls >= num_classes:
            raise ValueError(f"class ID {cls} is outside the valid range [0, {num_classes}).")

    if num_classes < 2:
        raise ValueError("num_classes must be at least 2 to apply label noise.")

    total_samples = targets.numel()
    desired_noisy = int(round(total_noise_rate * total_samples))

    if desired_noisy == 0:
        meta = {
            "achieved_noise_rate": 0.0,
            "per_class_noise_rate": 0.0,
            "num_noisy_samples": 0,
            "target_classes": unique_subset,
        }
        return targets.clone(), torch.zeros_like(targets, dtype=torch.long), meta

    targets_cpu = targets.detach().cpu().clone()
    mask = torch.zeros_like(targets_cpu, dtype=torch.bool)
    for cls in unique_subset:
        mask |= targets_cpu == cls
    eligible_indices = torch.nonzero(mask, as_tuple=False).flatten()

    eligible_count = int(eligible_indices.numel())
    if eligible_count == 0:
        raise ValueError("No samples belong to the specified class_subset.")

    if desired_noisy > eligible_count:
        raise ValueError(
            f"Requested {desired_noisy} noisy samples but only {eligible_count} samples "
            "belong to the specified class_subset."
        )

    rng = np.random.default_rng(random_state)
    eligible_indices_np = eligible_indices.numpy()
    rng.shuffle(eligible_indices_np)
    selected_indices = eligible_indices_np[:desired_noisy]

    noisy_targets_cpu = targets_cpu.clone()
    noise_info_cpu = torch.zeros_like(targets_cpu, dtype=torch.long)

    for idx in selected_indices:
        idx = int(idx)
        original_label = int(noisy_targets_cpu[idx].item())
        rand_value = int(rng.integers(0, num_classes - 1))
        if rand_value >= original_label:
            rand_value += 1
        noisy_targets_cpu[idx] = rand_value
        noise_info_cpu[idx] = 1

    noisy_targets = noisy_targets_cpu.to(targets.device, non_blocking=True)
    noise_info = noise_info_cpu.to(targets.device, non_blocking=True)

    achieved_noise_rate = desired_noisy / total_samples
    per_class_noise_rate = desired_noisy / eligible_count
    meta = {
        "achieved_noise_rate": achieved_noise_rate,
        "per_class_noise_rate": per_class_noise_rate,
        "num_noisy_samples": desired_noisy,
        "target_classes": unique_subset,
    }
    return noisy_targets, noise_info, meta


def apply_label_noise_to_dataset(dataset, noise_rate, num_digits=10, num_colors=1):
    """
    指定されたデータセットにラベルノイズを加える。
    ノイズは一様にランダムなクラスに置き換える方式。
    - 入力: PyTorch Dataset (image, label) or (image, label, ...)
    - 出力: NoisyDataset (image, noisy_label, noise_flag)

    Args:
        dataset (Dataset): 元データセット（PyTorch Dataset）
        noise_rate (float): ノイズ率（0〜1）
        num_digits (int): 数字クラスの数（combined ターゲットに使う）
        num_colors (int): 色クラスの数（combined ターゲットに使う）

    Returns:
        noisy_dataset (NoisyDataset)
    """
    x_list, y_list, noise_info_list = [], [], []
    num_classes = num_digits * num_colors

    for i in range(len(dataset)):
        sample = dataset[i]
        if isinstance(sample, tuple) and len(sample) == 2:
            x, y = sample
        elif isinstance(sample, tuple) and len(sample) >= 3:
            x, y = sample[0], sample[1]
        else:
            raise ValueError("Unsupported dataset format")

        if np.random.rand() < noise_rate:
            y_noisy = np.random.randint(num_classes)
            while y_noisy == y:
                y_noisy = np.random.randint(num_classes)
            y_list.append(y_noisy)
            noise_info_list.append(1)
        else:
            y_list.append(y)
            noise_info_list.append(0)

        x_list.append(x)

    # Tensor へ変換
    x_tensor = torch.stack(x_list) if isinstance(x_list[0], torch.Tensor) else torch.tensor(np.stack(x_list))
    y_tensor = torch.tensor(y_list, dtype=torch.long)
    noise_tensor = torch.tensor(noise_info_list, dtype=torch.long)

    base_dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    return NoisyDataset(base_dataset, noise_tensor)


def _load_or_create_cached_dataset(dataset_key, noise_rate, return_type="torch"):
    config = DATASET_CACHE_CONFIG[dataset_key]
    rate_str = format_noise_rate(noise_rate)
    cache_dir = os.path.join(config["base_dir"], config["noise_subdir"].format(rate=rate_str))
    os.makedirs(cache_dir, exist_ok=True)

    train_file = os.path.join(cache_dir, "train_data.pt")
    test_file = os.path.join(cache_dir, "test_data.pt")
    meta_file = os.path.join(cache_dir, "meta.pt")

    cache_exists = all(os.path.exists(path) for path in (train_file, test_file, meta_file))

    if cache_exists:
        print(f"[DatasetCache] Loaded cached dataset '{dataset_key}' (noise_rate={noise_rate}) from {cache_dir}")
    else:
        print(f"[DatasetCache] Creating cached dataset '{dataset_key}' (noise_rate={noise_rate}) at {cache_dir}")
        raw_train, raw_test = config["raw_loader"]()
        noisy_train = apply_label_noise_to_dataset(
            raw_train, noise_rate, num_digits=config["num_digits"], num_colors=config["num_colors"]
        )

        x_train, y_train = noisy_train.dataset.tensors
        noise_info = noisy_train.noise_info.clone()

        x_test, y_test = raw_test.tensors

        torch.save({
            "x_train": x_train.clone(),
            "y_train": y_train.clone(),
            "noise_info": noise_info.clone(),
        }, train_file)

        torch.save({
            "x_test": x_test.clone(),
            "y_test": y_test.clone(),
        }, test_file)

        meta = {
            "imagesize": config["imagesize"],
            "num_classes": config["num_classes"],
            "in_channels": config["in_channels"],
            "noise_rate": noise_rate,
        }
        torch.save(meta, meta_file)
        print(f"[DatasetCache] Saved new cache for '{dataset_key}' to {cache_dir}")

    train_data = torch.load(train_file)
    test_data = torch.load(test_file)
    meta = torch.load(meta_file)

    x_train = train_data["x_train"].float()
    y_train = train_data["y_train"].long()
    noise_info = train_data["noise_info"].long()

    x_test = test_data["x_test"].float()
    y_test = test_data["y_test"].long()

    stats = DATASET_NORMALIZATION_STATS.get(dataset_key)
    if stats is not None:
        mean_tensor = torch.tensor(stats["mean"], dtype=x_train.dtype).view(1, -1, 1, 1)
        std_tensor = torch.tensor(stats["std"], dtype=x_train.dtype).view(1, -1, 1, 1)

        def _maybe_denormalize(tensor):
            if tensor.max() <= 1.0 and tensor.min() >= 0.0:
                return tensor, False
            restored = tensor * std_tensor + mean_tensor
            return restored.clamp(0.0, 1.0), True

        x_train, _ = _maybe_denormalize(x_train)
        x_test, _ = _maybe_denormalize(x_test)

    train_tensor_dataset = TensorDataset(x_train, y_train)
    train_dataset = NoisyDataset(train_tensor_dataset, noise_info)
    test_dataset = TensorDataset(x_test, y_test)

    train_transform, test_transform = load_transforms_for_dataset(dataset_key, config)
    if train_transform is not None:
        train_dataset.set_transform(train_transform)
    print(f"[DatasetCache] Train transform for '{dataset_key}': {_describe_transform(train_transform)}")
    if test_transform is not None:
        test_dataset = TransformedDataset(test_dataset, test_transform)
    print(f"[DatasetCache] Test transform for '{dataset_key}': {_describe_transform(test_transform)}")

    if return_type == "npy":
        return (
            x_train.numpy(),
            y_train.numpy(),
            noise_info.numpy(),
            x_test.numpy(),
            y_test.numpy(),
            meta,
        )

    return train_dataset, test_dataset, meta

# -----------------------------------------
# ② 既存Datasetに transform をかけ直すためのラッパー
# -----------------------------------------
class TransformedDataset(Dataset):
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __getitem__(self, idx):
        data = self.base_dataset[idx]
        # (x, y) or (x, y, noise_flag) を考慮
        x, y = data[:2]
        if isinstance(x, torch.Tensor):
            x = x.clone()
        # x に transform を適用 (Tensor -> PIL に変換不要な augment のみ推奨)
        x = self.transform(x)
        # 3つ目の要素 (noise_flag) がある場合もそのまま返却
        return (x, y) if len(data) == 2 else (x, y, data[2])

    def __len__(self):
        return len(self.base_dataset)

def load_or_create_noisy_dataset(dataset, target, gray_scale, args, return_type="torch"):
    """
    指定されたデータセットについてラベルノイズを付与した学習データを保存／読み込みし、
    テストデータ・meta情報も取得する。

    Parameters:
    ----------
    dataset : str
        データセット名（"cifar10"、"emnist"など）。
    target : str
        EMNISTなどで使用するターゲット種別や、coloredの際の判定用。
    gray_scale : bool
        入力データをグレイスケール化するかどうか。
    args : argparse.Namespace
        label_noise_rateやfix_seedなどの実行時引数。
    return_type : {"torch", "npy"}
        返り値の型を指定。"torch"の場合はDataset形式、"npy"の場合はnumpy配列。
    """
    label_noise_rate = args.label_noise_rate

    dataset_key = dataset.lower()

    if dataset_key in DATASET_CACHE_CONFIG:
        return _load_or_create_cached_dataset(dataset_key, label_noise_rate, return_type)

    # 既存ロジックはその他のデータセット向けに残しておく
    if dataset_key in ["emnist", "emnist_digits"]:
        return _handle_emnist_digits(target, gray_scale, args, return_type)
    elif "distribution_colored_emnist" in dataset_key:
        print("distribution_colored_emnist")
        print(target)
        return _handle_distribution_colored_emnist(dataset, target, gray_scale, args, return_type)
    else:
        raise ValueError("Unsupported dataset type for noisy dataset loader")


def _handle_cifar10(args, return_type):
    """
    CIFAR-10用の保存／読み込みの設定を行い、_load_or_save_datasetに渡す。
    """
    base_dir = os.path.join("/workspace/data", "cifar10")
    train_dir = os.path.join(base_dir, f"cifar10_noise_{args.label_noise_rate}")
    test_dir = base_dir
    noise_num_colors = 1
    return _load_or_save_dataset(
        dataset="cifar10",
        train_dir=train_dir,
        test_dir=test_dir,
        noise_num_colors=noise_num_colors,
        gray_scale=False,
        args=args,
        return_type=return_type
    )


def _handle_emnist_digits(target, gray_scale, args, return_type):
    """
    EMNIST(またはEMNISTの数字部分)用の保存／読み込み設定を行い、_load_or_save_datasetに渡す。
    """
    base_dir = os.path.join("/workspace/data", "EMNIST")
    train_dir = os.path.join(base_dir, f"emnist_noise_{args.label_noise_rate}")
    test_dir = base_dir
    noise_num_colors = 1
    return _load_or_save_dataset(
        dataset="emnist_digits",
        train_dir=train_dir,
        test_dir=test_dir,
        noise_num_colors=noise_num_colors,
        gray_scale=gray_scale,
        args=args,
        return_type=return_type
    )


def _handle_distribution_colored_emnist(dataset, target, gray_scale, args, return_type):
    """
    Distribution Colored EMNIST, あるいは通常のColored EMNIST用の設定を行い、
    _load_or_save_datasetに渡す。
    """
    if dataset.lower() == "distribution_colored_emnist":
        base_dir = os.path.join(
            "/workspace/data",
            f"distribution_colored_EMNIST_Seed{args.fix_seed}_Var{args.variance}_Corr{args.correlation}"
        )
        train_dir = os.path.join(base_dir, f"{os.path.basename(base_dir)}_noise_{args.label_noise_rate}")
    else:
        base_dir = os.path.join("/workspace/data", "colored_EMNIST")
        train_dir = base_dir + f"_noise_{args.label_noise_rate}"

    test_dir = base_dir
    noise_num_colors = 10 if target == "combined" else 1
    return _load_or_save_dataset(
        dataset=dataset,
        train_dir=train_dir,
        test_dir=test_dir,
        noise_num_colors=noise_num_colors,
        gray_scale=gray_scale,
        args=args,
        return_type=return_type
    )


def _check_files_exist(directory, file_list):
    """
    指定したディレクトリ内に、file_listで与えられた全てのファイルが存在するかを確認する。

    Parameters:
    ----------
    directory : str
        チェック対象のディレクトリ。
    file_list : list of str
        存在確認を行うファイル名のリスト。

    Returns:
    -------
    bool
        全てのファイルが存在すればTrue、それ以外はFalse。
    """
    if not os.path.exists(directory):
        return False
    return all(os.path.exists(os.path.join(directory, f)) for f in file_list)


def _load_or_save_dataset(dataset,train_dir, test_dir, noise_num_colors, gray_scale, args, return_type):
    from torch.utils.data import TensorDataset
    label_noise_rate = args.label_noise_rate

    train_file = os.path.join(train_dir, "train_data.pt")
    test_file = os.path.join(test_dir, "test_data.pt")
    meta_file = os.path.join(test_dir, "meta.pt")

    train_files_exist = os.path.exists(train_file)
    test_files_exist = os.path.exists(test_file) and os.path.exists(meta_file)

    if not (train_files_exist and test_files_exist):
        missing_files = []
        if not os.path.exists(train_file):
            missing_files.append(train_file)
        if not os.path.exists(test_file):
            missing_files.append(test_file)
        if not os.path.exists(meta_file):
            missing_files.append(meta_file)
        print("以下のファイルが存在しないため新規生成します：")
        for f in missing_files:
            print(f"  - {f}")
        
        full_train_dataset, full_test_dataset, imagesize, num_classes, in_channels = load_datasets(
            dataset, "combined", gray_scale, args
        )

        # 学習データの準備
        if isinstance(full_train_dataset, TensorDataset):
            x_train, y_train = full_train_dataset.tensors
        else:
            x_list, y_list = zip(*[(x, y) for x, y in full_train_dataset])
            x_train = torch.stack(x_list)
            y_train = torch.tensor(y_list)
        x_train = x_train.float()

        # ラベルノイズの付加
        print(f"Adding label noise with rate: {label_noise_rate}")
        y_train_noisy, noise_info = add_label_noise(
            y_train, label_noise_rate, num_digits=10, num_colors=noise_num_colors
        )

        train_dataset = TensorDataset(x_train, y_train_noisy)
        train_dataset = NoisyDataset(train_dataset, noise_info)

        meta_local = {
            "imagesize": imagesize,
            "num_classes": num_classes,
            "in_channels": in_channels
        }

        # 保存（.pt 形式でまとめて保存）
        if not train_files_exist:
            os.makedirs(train_dir, exist_ok=True)
            torch.save({
                "x_train": x_train,
                "y_train": y_train_noisy,
                "noise_info": noise_info
            }, train_file)

        if not test_files_exist:
            if isinstance(full_test_dataset, TensorDataset):
                x_test, y_test = full_test_dataset.tensors
            else:
                x_test_list, y_test_list = zip(*[(x, y) for x, y in full_test_dataset])
                x_test = torch.stack(list(x_test_list))
                y_test = torch.tensor(y_test_list)
            x_test = x_test.float()

            os.makedirs(test_dir, exist_ok=True)
            torch.save({
                "x_test": x_test,
                "y_test": y_test
            }, test_file)
            torch.save(meta_local, meta_file)

        meta = meta_local
    else:
        # 読み込み（.pt形式から）
        print("以下のファイルからデータを読み込みます：")
        print(f"  - {train_file}")
        print(f"  - {test_file}")
        print(f"  - {meta_file}")
        train_data = torch.load(train_file)
        x_train = train_data["x_train"].float()
        y_train = train_data["y_train"]
        noise_info = train_data["noise_info"]

        test_data = torch.load(test_file)
        x_test = test_data["x_test"].float()
        y_test = test_data["y_test"]

        meta = torch.load(meta_file)

        if dataset.lower() == "cifar10":
            stats = DATASET_NORMALIZATION_STATS["cifar10"]
            mean = torch.tensor(stats["mean"], dtype=x_train.dtype).view(1, -1, 1, 1)
            std = torch.tensor(stats["std"], dtype=x_train.dtype).view(1, -1, 1, 1)

            def _maybe_denormalize(tensor):
                if tensor.min() < 0 or tensor.max() > 1:
                    denorm = tensor * std + mean
                    return denorm.clamp(0.0, 1.0), True
                return tensor, False

            x_train, converted_train = _maybe_denormalize(x_train)
            x_test, converted_test = _maybe_denormalize(x_test)

            if converted_train or converted_test:
                torch.save({
                    "x_train": x_train,
                    "y_train": y_train,
                    "noise_info": noise_info
                }, train_file)
                torch.save({
                    "x_test": x_test,
                    "y_test": y_test
                }, test_file)

    if return_type == "npy":
        # NumPy配列として返却
        return (
            x_train.numpy(), y_train.numpy(), noise_info.numpy(),
            x_test.numpy(), y_test.numpy(), meta
        )

    elif return_type == "torch":
        train_tensor_dataset = TensorDataset(x_train, y_train)
        
        # ラベルノイズがなくても NoisyDataset を使う（noise_info は全て0）
        train_dataset = NoisyDataset(train_tensor_dataset, noise_info)

        test_dataset = TensorDataset(x_test, y_test)

        return train_dataset, test_dataset, meta

    else:
        raise ValueError("return_type must be either 'npy' or 'torch'")

# その他の関数 (add_label_noise, NoisyDataset, load_datasets) は元のまま保持

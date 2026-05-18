import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def _default_num_workers_for_device(device: torch.device) -> int:
    cpu_count = os.cpu_count() or 2
    if device.type == "cuda":
        return min(16, max(4, cpu_count - 1))
    if device.type == "mps":
        return min(4, max(2, cpu_count // 3))
    return 2


def _default_batch_size_for_device(device: torch.device, dataset: str = "cifar10") -> int:
    """Auto-tune batch size based on device and dataset."""
    if device.type == "cuda":
        if dataset == "cifar10":
            return 256
        elif dataset == "cifar100":
            return 128
        else:  # tinyimagenet
            return 64
    elif device.type == "mps":
        if dataset == "cifar10":
            return 64
        elif dataset == "cifar100":
            return 32
        else:  # tinyimagenet
            return 16
    else:  # cpu
        return 32


def get_dataset_loaders(
    dataset: str = "cifar10",
    batch_size: int | None = None,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
    data_dir: str = "./data",
    device: torch.device | None = None,
    normalize: bool = False,
    download: bool = True,
):
    """
    Return train and test DataLoaders for CIFAR-10, CIFAR-100, or TinyImageNet.

    Args:
        dataset: "cifar10", "cifar100", or "tinyimagenet"
        batch_size: if None, chosen based on device and dataset
        num_workers: if None, chosen based on CPU count and device
        pin_memory: if None, set to True when using CUDA
        data_dir: dataset root directory
        device: if provided, used to infer num_workers/pin_memory/batch_size
        normalize: whether to apply dataset-specific normalization
        download: whether to download the dataset
    """
    dataset = dataset.lower()
    if dataset not in ["cifar10", "cifar100", "tinyimagenet"]:
        raise ValueError(f"Unknown dataset: {dataset}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    if batch_size is None:
        batch_size = _default_batch_size_for_device(device, dataset)

    if num_workers is None:
        num_workers = _default_num_workers_for_device(device)

    if pin_memory is None:
        pin_memory = True if device.type == "cuda" else False

    os.makedirs(data_dir, exist_ok=True)

    # Dataset-specific transforms and normalization
    if dataset == "cifar10":
        num_classes = 10
        img_size = 32
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
    elif dataset == "cifar100":
        num_classes = 100
        img_size = 32
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    else:  # tinyimagenet
        num_classes = 200
        img_size = 64
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

    if normalize:
        norm = transforms.Normalize(mean, std)
    else:
        norm = None

    train_transforms = [
        transforms.RandomCrop(img_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
    if norm is not None:
        train_transforms.append(norm)

    test_transforms = [transforms.ToTensor()]
    if norm is not None:
        test_transforms.append(norm)

    transform_train = transforms.Compose(train_transforms)
    transform_test = transforms.Compose(test_transforms)

    # Load dataset
    if dataset == "cifar10":
        train_ds = datasets.CIFAR10(root=data_dir, train=True, download=download, transform=transform_train)
        test_ds = datasets.CIFAR10(root=data_dir, train=False, download=download, transform=transform_test)
    elif dataset == "cifar100":
        train_ds = datasets.CIFAR100(root=data_dir, train=True, download=download, transform=transform_train)
        test_ds = datasets.CIFAR100(root=data_dir, train=False, download=download, transform=transform_test)
    else:  # tinyimagenet
        train_ds = datasets.ImageNet(
            root=data_dir, split='train', download=download, transform=transform_train
        )
        test_ds = datasets.ImageNet(
            root=data_dir, split='val', download=download, transform=transform_test
        )

    persistent = num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )

    return train_loader, test_loader, num_classes


def get_cifar10_loaders(
    batch_size: int = 128,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
    data_dir: str = "./data",
    device: torch.device | None = None,
    normalize: bool = False,
    download: bool = True,
):
    """
    Return CIFAR-10 train and test DataLoaders.
    (Kept for backward compatibility)
    """
    train_loader, test_loader, _ = get_dataset_loaders(
        dataset="cifar10",
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        data_dir=data_dir,
        device=device,
        normalize=normalize,
        download=download,
    )
    return train_loader, test_loader

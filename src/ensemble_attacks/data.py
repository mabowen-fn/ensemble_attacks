import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def _default_num_workers_for_device(device: torch.device) -> int:
    cpu_count = os.cpu_count() or 2
    if device.type == "cuda":
        return min(16, max(4, cpu_count - 1))
    if device.type == "mps":
        return min(8, max(2, cpu_count // 2))
    return 2


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
    Return CIFAR-10 train and test DataLoaders with sensible defaults for
    GPU/cloud environments.

    Args:
        batch_size: batch size
        num_workers: if None, chosen based on CPU count and device
        pin_memory: if None, set to True when using CUDA
        data_dir: dataset root
        device: if provided, used to infer num_workers/pin_memory
        normalize: whether to apply CIFAR-10 mean/std normalization
        download: whether to download the dataset
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    if num_workers is None:
        num_workers = _default_num_workers_for_device(device)

    if pin_memory is None:
        pin_memory = True if device.type == "cuda" else False

    os.makedirs(data_dir, exist_ok=True)

    if normalize:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
        norm = transforms.Normalize(mean, std)
    else:
        norm = None

    train_transforms = [
        transforms.RandomCrop(32, padding=4),
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

    train_ds = datasets.CIFAR10(root=data_dir, train=True, download=download, transform=transform_train)
    test_ds = datasets.CIFAR10(root=data_dir, train=False, download=download, transform=transform_test)

    # Use persistent_workers when appropriate to improve throughput in long runs
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

    return train_loader, test_loader

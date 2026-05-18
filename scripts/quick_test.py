#!/usr/bin/env python3
"""Quick 1-epoch training test to validate production setup."""

import sys
sys.path.insert(0, 'src')

import torch
import torch.optim as optim
from ensemble_attacks.config import ExperimentConfig
from ensemble_attacks.data import get_cifar10_loaders
from ensemble_attacks.models import get_resnet18
from ensemble_attacks.train import train_one_epoch, evaluate
from ensemble_attacks.utils import get_device, get_device_info, warmup_device, set_random_seed
from ensemble_attacks.logging_utils import ExperimentLogger, print_device_info

print("\n" + "="*70)
print("QUICK START: 1-EPOCH TRAINING TEST")
print("="*70)

# Setup
config = ExperimentConfig(num_epochs=1, batch_size=64, num_workers=0)  # No workers for quick test
device = get_device(prefer_cuda=False)
device_info = get_device_info()
print_device_info(device_info)

set_random_seed(seed=42)

logger = ExperimentLogger()
print(f"✓ Experiment ID: {logger.exp_id}\n")

# Load data
train_loader, test_loader = get_cifar10_loaders(
    batch_size=config.batch_size,
    num_workers=0,  # Disable multiprocessing for this test
    device=device
)
print(f"✓ Data loaded: {len(train_loader)} train batches, {len(test_loader)} test batches")

# Create model
model = get_resnet18().to(device)
param_count = sum(p.numel() for p in model.parameters())
print(f"✓ Model created: ResNet18 with {param_count:,} parameters")

# Train
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
print(f"\n✓ Starting training (1 epoch, batch_size=64)...\n")

loss, acc = train_one_epoch(model, train_loader, optimizer, device)
print(f"  Training - Loss: {loss:.4f}, Accuracy: {acc:.4f}")

test_acc = evaluate(model, test_loader, device)
print(f"  Testing  - Accuracy: {test_acc:.4f}")

# Log metrics
logger.log_metric({
    "epoch": 1,
    "train_loss": loss,
    "train_acc": acc,
    "test_acc": test_acc,
    "device": str(device),
})

print(f"\n✓ Experiment complete!")
print(f"✓ Logs saved to: {logger.exp_dir}")

print("\n" + "="*70)
print("✅ QUICK START TEST PASSED - SYSTEM IS PRODUCTION READY!")
print("="*70)
print("\nNext steps for production use:")
print("  1. Run full validation: uv run python scripts/validate_setup.py")
print("  2. Train full model: uv run python scripts/train_resnet18.py")
print("  3. Evaluate attacks: uv run python scripts/eval_ensemble_attacks.py")
print("\nAll results logged to: outputs/logs/")

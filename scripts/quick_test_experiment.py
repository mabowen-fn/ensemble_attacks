#!/usr/bin/env python3
"""Quick test of the large-scale experiment script."""

import os
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ensemble_attacks.data import get_dataset_loaders
from ensemble_attacks.models import get_model
from ensemble_attacks.utils import get_device, set_random_seed
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def quick_test():
    """Run a quick test of the experiment pipeline."""
    
    logger.info("="*80)
    logger.info("QUICK TEST: EXPERIMENT PIPELINE")
    logger.info("="*80)
    
    # Setup
    device = get_device(prefer_cuda=False)
    logger.info(f"\nDevice: {device}")
    set_random_seed(42)
    
    # Load data
    logger.info("\nLoading CIFAR-10...")
    _, test_loader, num_classes = get_dataset_loaders(
        dataset="cifar10",
        batch_size=32,
        num_workers=0,
        device=device,
    )
    
    # Load models
    logger.info("Loading models...")
    surrogate = get_model("resnet18", num_classes=10).to(device).eval()
    target = get_model("vgg16", num_classes=10).to(device).eval()
    
    logger.info("✓ Models loaded")
    
    # Test single image
    logger.info("\nTesting on single image...")
    x_batch, y_batch = next(iter(test_loader))
    x = x_batch[0].to(device, dtype=torch.float32)
    true_label = y_batch[0].item()
    
    logger.info(f"  Image shape: {x.shape}")
    logger.info(f"  True label: {true_label}")
    
    # Test forward pass
    with torch.no_grad():
        logits_surr = surrogate(x.unsqueeze(0))
        logits_tgt = target(x.unsqueeze(0))
    
    logger.info(f"  Surrogate output: {logits_surr.shape}")
    logger.info(f"  Target output: {logits_tgt.shape}")
    
    # Test gradient computation
    logger.info("\nTesting gradient computation...")
    x_grad = x.clone().requires_grad_(True)
    logits = surrogate(x_grad.unsqueeze(0))
    loss = F.cross_entropy(logits, torch.tensor([true_label], device=device))
    surrogate.zero_grad()
    loss.backward()
    grad = x_grad.grad.clone().detach()
    
    logger.info(f"  Gradient shape: {grad.shape}")
    logger.info(f"  Gradient norm: {grad.norm().item():.6f}")
    
    # Test adversarial perturbation
    logger.info("\nTesting adversarial perturbation...")
    epsilon = 8.0 / 255.0
    x_adv = x + epsilon * torch.sign(grad)
    x_adv = torch.clamp(x_adv, 0, 1)
    
    linf = (x_adv - x).abs().max().item()
    logger.info(f"  L∞ perturbation: {linf:.6f}")
    logger.info(f"  ε: {epsilon:.6f}")
    logger.info(f"  Within budget: {linf <= epsilon + 1e-6}")
    
    # Test model predictions
    with torch.no_grad():
        logits_clean = target(x.unsqueeze(0))
        logits_adv = target(x_adv.unsqueeze(0))
        pred_clean = logits_clean.argmax(dim=1).item()
        pred_adv = logits_adv.argmax(dim=1).item()
    
    logger.info(f"  Clean prediction: {pred_clean}")
    logger.info(f"  Adversarial prediction: {pred_adv}")
    logger.info(f"  Attack success: {pred_clean != pred_adv}")
    
    logger.info("\n" + "="*80)
    logger.info("✓ QUICK TEST PASSED")
    logger.info("="*80)
    
    return True


if __name__ == "__main__":
    try:
        success = quick_test()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

#!/usr/bin/env python3
"""Quick validation of large-scale experiment infrastructure."""

import os
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ensemble_attacks.data import get_dataset_loaders
from ensemble_attacks.models import get_model
from ensemble_attacks.utils import get_device

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_setup():
    """Validate all components work."""
    
    logger.info("=" * 80)
    logger.info("VALIDATING LARGE-SCALE EXPERIMENT SETUP")
    logger.info("=" * 80)
    
    # 1. Device
    logger.info("\n1. Device detection...")
    try:
        device = get_device(prefer_cuda=False)
        logger.info(f"   ✓ Device: {device}")
    except Exception as e:
        logger.error(f"   ✗ Failed: {e}")
        return False
    
    # 2. Dataset loading
    logger.info("\n2. Dataset loading (CIFAR-10)...")
    try:
        train_loader, test_loader, num_classes = get_dataset_loaders(
            dataset="cifar10",
            batch_size=32,
            num_workers=0,
            device=device,
        )
        logger.info(f"   ✓ Loaded CIFAR-10 (num_classes={num_classes})")
        
        # Get one batch
        x_batch, y_batch = next(iter(test_loader))
        logger.info(f"   ✓ Batch shape: {x_batch.shape}, labels: {y_batch.shape}")
    except Exception as e:
        logger.error(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 3. Model loading
    logger.info("\n3. Model loading...")
    try:
        for model_name in ["resnet18", "vgg16", "densenet121"]:
            model = get_model(model_name, num_classes=10)
            model = model.to(device)
            logger.info(f"   ✓ Loaded {model_name}")
    except Exception as e:
        logger.error(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. Forward pass
    logger.info("\n4. Forward pass...")
    try:
        x_batch = x_batch.to(device, dtype=torch.float32)
        with torch.no_grad():
            logits = model(x_batch)
        logger.info(f"   ✓ Output shape: {logits.shape}")
    except Exception as e:
        logger.error(f"   ✗ Failed: {e}")
        return False
    
    # 5. Gradient computation
    logger.info("\n5. Gradient computation...")
    try:
        model.eval()  # Ensure model is in eval mode
        x_test = x_batch[0].clone().requires_grad_(True)
        logits = model(x_test.unsqueeze(0))
        loss = logits[0].max()  # Use max logit as loss
        loss.backward()
        grad = x_test.grad
        logger.info(f"   ✓ Gradient shape: {grad.shape}")
    except Exception as e:
        logger.error(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ ALL VALIDATIONS PASSED")
    logger.info("=" * 80)
    return True


if __name__ == "__main__":
    import torch
    success = validate_setup()
    sys.exit(0 if success else 1)

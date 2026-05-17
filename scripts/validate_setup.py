#!/usr/bin/env python3
"""
Validate production setup before running experiments.
Tests CUDA availability, data loading, model loading, and attack validation.
"""

import sys
sys.path.insert(0, 'src')

import torch
from ensemble_attacks.utils import get_device, get_device_info, set_random_seed
from ensemble_attacks.config import ExperimentConfig
from ensemble_attacks.data import get_cifar10_loaders
from ensemble_attacks.models import get_resnet18
from ensemble_attacks.logging_utils import print_device_info, ExperimentLogger
from ensemble_attacks.attacks import fgsm_attack, pgd_attack, bim_attack


def test_device_detection():
    """Test CUDA/device detection."""
    print("\n" + "="*60)
    print("TEST 1: Device Detection")
    print("="*60)
    
    try:
        device = get_device(prefer_cuda=False)  # Test CPU first
        print(f"✓ CPU detection: {device}")
    except RuntimeError as e:
        print(f"✗ CPU detection failed: {e}")
        return False
    
    device_info = get_device_info()
    print_device_info(device_info)
    
    if device_info['cuda_available']:
        print("✓ CUDA is available - GPU acceleration enabled")
        try:
            cuda_device = get_device(prefer_cuda=True)
            print(f"✓ CUDA device selected: {cuda_device}")
        except RuntimeError as e:
            print(f"✗ CUDA selection failed: {e}")
            return False
    else:
        print("⚠ CUDA not available - will use CPU (slower)")
    
    return True


def test_config():
    """Test configuration system."""
    print("\n" + "="*60)
    print("TEST 2: Configuration System")
    print("="*60)
    
    try:
        config = ExperimentConfig()
        print(f"✓ Default config created")
        print(f"  - Batch size: {config.batch_size}")
        print(f"  - Epochs: {config.num_epochs}")
        print(f"  - Attack epsilon: {config.attack.epsilon:.6f}")
        print(f"  - Attack iters: {config.attack.iters}")
        print(f"  - Seed: {config.reproducibility.seed}")
        print(f"  - Deterministic: {config.reproducibility.deterministic}")
    except Exception as e:
        print(f"✗ Config creation failed: {e}")
        return False
    
    # Test config validation
    try:
        from ensemble_attacks.config import AttackConfig
        bad_config = AttackConfig(epsilon=-1)
        print("✗ Config validation failed - should reject negative epsilon")
        return False
    except AssertionError:
        print("✓ Config validation working (rejects invalid parameters)")
    
    return True


def test_reproducibility():
    """Test reproducibility settings."""
    print("\n" + "="*60)
    print("TEST 3: Reproducibility")
    print("="*60)
    
    try:
        set_random_seed(seed=42, deterministic=False)
        print("✓ Random seed setting successful")
        
        import random
        import numpy as np
        x1 = random.random()
        y1 = np.random.random()
        t1 = torch.rand(1).item()
        
        set_random_seed(seed=42, deterministic=False)
        x2 = random.random()
        y2 = np.random.random()
        t2 = torch.rand(1).item()
        
        if abs(x1 - x2) < 1e-10 and abs(y1 - y2) < 1e-10 and abs(t1 - t2) < 1e-10:
            print("✓ Reproducible random number generation (seed=42)")
        else:
            print("⚠ Seed setting may not be fully reproducible")
    except Exception as e:
        print(f"✗ Reproducibility test failed: {e}")
        return False
    
    return True


def test_data_loading():
    """Test CIFAR-10 data loading."""
    print("\n" + "="*60)
    print("TEST 4: Data Loading")
    print("="*60)
    
    try:
        device = get_device(prefer_cuda=False)
        print(f"Loading CIFAR-10 on device: {device}")
        
        train_loader, test_loader = get_cifar10_loaders(
            batch_size=32,
            num_workers=None,  # Auto
            pin_memory=None,   # Auto
            device=device,
            download=True
        )
        
        print(f"✓ Data loaders created")
        print(f"  - Train batches: {len(train_loader)}")
        print(f"  - Test batches: {len(test_loader)}")
        
        # Load one batch
        x, y = next(iter(train_loader))
        print(f"✓ Sample batch loaded")
        print(f"  - Images shape: {x.shape} (should be [32, 3, 32, 32])")
        print(f"  - Labels shape: {y.shape} (should be [32])")
        print(f"  - Image range: [{x.min().item():.4f}, {x.max().item():.4f}] (should be [0, 1])")
        
        if x.shape != torch.Size([32, 3, 32, 32]):
            print("✗ Unexpected batch shape")
            return False
        
        if x.min() < 0 or x.max() > 1:
            print("✗ Image values out of [0, 1] range")
            return False
        
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_model():
    """Test model loading."""
    print("\n" + "="*60)
    print("TEST 5: Model Loading")
    print("="*60)
    
    try:
        model = get_resnet18()
        param_count = sum(p.numel() for p in model.parameters())
        print(f"✓ ResNet18 model created")
        print(f"  - Parameters: {param_count:,}")
        
        device = get_device(prefer_cuda=False)
        model = model.to(device)
        model.eval()
        print(f"✓ Model moved to {device}")
        
        # Test forward pass
        x = torch.rand(2, 3, 32, 32).to(device)
        with torch.no_grad():
            logits = model(x)
        
        print(f"✓ Forward pass successful")
        print(f"  - Output shape: {logits.shape} (should be [2, 10])")
        
        if logits.shape != torch.Size([2, 10]):
            print("✗ Unexpected output shape")
            return False
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_attacks():
    """Test attack functions."""
    print("\n" + "="*60)
    print("TEST 6: Attack Functions")
    print("="*60)
    
    try:
        device = get_device(prefer_cuda=False)
        model = get_resnet18().to(device).eval()
        
        x = torch.rand(2, 3, 32, 32).to(device)
        y = torch.tensor([0, 1]).to(device)
        
        # Test FGSM
        x_adv_fgsm = fgsm_attack(model, x, y, epsilon=8/255)
        print(f"✓ FGSM attack successful")
        print(f"  - Output shape: {x_adv_fgsm.shape}")
        print(f"  - Range: [{x_adv_fgsm.min().item():.4f}, {x_adv_fgsm.max().item():.4f}]")
        
        if x_adv_fgsm.min() < 0 or x_adv_fgsm.max() > 1:
            print("✗ FGSM output out of [0, 1] range")
            return False
        
        # Test BIM
        x_adv_bim = bim_attack(model, x, y, epsilon=8/255, alpha=2/255, iters=5)
        print(f"✓ BIM attack successful")
        
        # Test PGD
        x_adv_pgd = pgd_attack(model, x, y, epsilon=8/255, alpha=2/255, iters=5)
        print(f"✓ PGD attack successful")
        
        if x_adv_pgd.min() < 0 or x_adv_pgd.max() > 1:
            print("✗ PGD output out of [0, 1] range")
            return False
        
    except Exception as e:
        print(f"✗ Attack test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_logging():
    """Test experiment logging."""
    print("\n" + "="*60)
    print("TEST 7: Experiment Logging")
    print("="*60)
    
    try:
        logger = ExperimentLogger(log_dir="outputs/logs")
        print(f"✓ ExperimentLogger created")
        print(f"  - Experiment ID: {logger.exp_id}")
        print(f"  - Directory: {logger.exp_dir}")
        
        logger.log_text("Test message", level="INFO")
        logger.log_metric({"test": 1.0, "value": 42})
        print(f"✓ Logging functions work")
        
    except Exception as e:
        print(f"✗ Logging test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    """Run all validation tests."""
    print("\n" + "="*60)
    print("ENSEMBLE ATTACKS - PRODUCTION SETUP VALIDATION")
    print("="*60)
    
    tests = [
        ("Device Detection", test_device_detection),
        ("Configuration System", test_config),
        ("Reproducibility", test_reproducibility),
        ("Data Loading", test_data_loading),
        ("Model Loading", test_model),
        ("Attack Functions", test_attacks),
        ("Experiment Logging", test_logging),
    ]
    
    results = []
    for test_name, test_fn in tests:
        try:
            result = test_fn()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗✗✗ {test_name} crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print("-" * 60)
    print(f"Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ All validation tests passed!")
        print("Ready to run production experiments.")
        return True
    else:
        print(f"\n❌ {total - passed} test(s) failed.")
        print("Please fix the above issues before running experiments.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

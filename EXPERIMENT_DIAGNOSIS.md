# Experiment Diagnosis: Why All Attacks Produce Identical Results

## Summary

Your experiment shows all four attacks (MI-FGSM, NES-only, Static-Hybrid, ELPD-Blend) producing **identical results** with ~98% attack success rate. This is caused by **extremely low clean accuracy** of the pretrained ImageNet models on CIFAR-10.

## Root Cause

**Pretrained ImageNet models have ~5-8% accuracy on CIFAR-10** (worse than random guessing on 10 classes!). This happens because:

1. **Domain mismatch**: ImageNet uses 224×224 images, CIFAR-10 uses 32×32 images
2. **Class mismatch**: ImageNet has 1000 classes, CIFAR-10 has 10 classes
3. **Distribution mismatch**: ImageNet features don't transfer well to low-resolution CIFAR images

## What This Means

When models have ~5% clean accuracy:
- **95% of samples are already misclassified** before any attack
- Attacks "succeed" immediately because the model is already wrong
- All attacks produce identical results because they all succeed in 1 step
- No actual adversarial optimization is happening

## Evidence

```
Clean Accuracy Test (500 samples):
- ResNet18: 7.8% (40/512)
- VGG16:    5.5% (28/512)
- Random guessing: 10%

Attack Results:
- All attacks: 98% ASR
- All attacks succeed in 1 step on same samples
- Identical L∞ perturbations (0.007843 = step_size)
```

## Solution: Fine-tune Models on CIFAR-10

You MUST train the models on CIFAR-10 before running experiments:

```bash
# Train surrogate model
uv run scripts/train_models.py --model resnet18 --dataset cifar10 --epochs 10

# Train target model  
uv run scripts/train_models.py --model vgg16 --dataset cifar10 --epochs 10
```

Expected training time: ~5-10 minutes per model on Apple Silicon MPS.


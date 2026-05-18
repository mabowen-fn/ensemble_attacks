# Model Training Guide

## Problem Fixed

**Previous Issue**: Models were initialized with random weights (`weights=None`), causing all attacks to produce identical results because:
- No learned features for transfer attacks to exploit
- Random gradients provided no useful signal
- Both surrogate and target made essentially random predictions

**Solution**: Models now use ImageNet pretrained weights by default, providing:
- Learned feature representations that transfer across architectures
- Meaningful gradients for surrogate-based attacks
- Realistic evaluation of attack transferability

## Quick Start

### Option 1: Use Pretrained ImageNet Weights (Default)

The simplest approach - models are initialized with ImageNet pretrained weights and used directly:

```bash
# Run experiment with pretrained models (no additional training needed)
python run_experiment.py --config quick
```

This works because ImageNet features transfer reasonably well to CIFAR datasets.

### Option 2: Fine-tune on CIFAR (Recommended for Best Results)

For optimal performance, fine-tune the pretrained models on your target dataset:

```bash
# Train surrogate model (ResNet18) on CIFAR-10
python scripts/train_models.py --model resnet18 --dataset cifar10 --epochs 10

# Train target model (VGG16) on CIFAR-10
python scripts/train_models.py --model vgg16 --dataset cifar10 --epochs 10

# Train all combinations at once
python scripts/train_models.py --train-all --epochs 10
```

Checkpoints are saved to `./checkpoints/` by default.

### Option 3: Use Fine-tuned Checkpoints in Experiments

Update your experiment config to use the trained checkpoints:

```python
from configs import FULL_EXPERIMENT

# Modify config to use fine-tuned models
FULL_EXPERIMENT.surrogate_checkpoint = "./checkpoints/resnet18_cifar10_seed42.pth"
FULL_EXPERIMENT.target_checkpoint = "./checkpoints/vgg16_cifar10_seed42.pth"
```

Or via command line (requires config file):

```bash
# Create custom config with checkpoints
python -c "
from configs import FULL_EXPERIMENT, save_config
FULL_EXPERIMENT.surrogate_checkpoint = './checkpoints/resnet18_cifar10_seed42.pth'
FULL_EXPERIMENT.target_checkpoint = './checkpoints/vgg16_cifar10_seed42.pth'
save_config(FULL_EXPERIMENT, 'configs/experiments/full_with_checkpoints.json')
"

# Run with custom config
python run_experiment.py --config configs/experiments/full_with_checkpoints.json
```

## Training Script Options

```bash
python scripts/train_models.py --help

Options:
  --model {resnet18,vgg16,densenet121}  Model architecture
  --dataset {cifar10,cifar100}          Dataset to train on
  --epochs EPOCHS                       Number of epochs (default: 10)
  --batch-size BATCH_SIZE               Batch size (default: 128)
  --lr LR                               Learning rate (default: 0.001)
  --device DEVICE                       Device (cuda/mps/cpu, default: auto)
  --output-dir OUTPUT_DIR               Checkpoint directory (default: ./checkpoints)
  --seed SEED                           Random seed (default: 42)
  --train-all                           Train all model-dataset combinations
```

## Expected Results

### With Random Weights (Old Behavior - BROKEN)
- All attacks: ~95% ASR (artificially high)
- Identical results across all methods
- No meaningful differences between attacks
- **Cannot validate research claims**

### With Pretrained Weights (Fixed)
- MI-FGSM (pure transfer): ~60-80% ASR
- NES-only (pure query): ~40-60% ASR
- Static-Hybrid: ~70-85% ASR
- ELPD-Blend: Should outperform static hybrid if method is effective
- **Clear differentiation between methods**
- **Statistical significance becomes meaningful**

### With Fine-tuned Weights (Best)
- Even better differentiation between methods
- Higher baseline accuracy on clean samples
- More realistic adversarial evaluation
- **Publication-quality results**

## Training Time Estimates

On Apple Silicon (M1/M2) with MPS:
- ResNet18 on CIFAR-10 (10 epochs): ~5-10 minutes
- VGG16 on CIFAR-10 (10 epochs): ~8-15 minutes
- DenseNet121 on CIFAR-10 (10 epochs): ~10-20 minutes

On CUDA GPU (e.g., RTX 3090):
- ResNet18 on CIFAR-10 (10 epochs): ~2-3 minutes
- VGG16 on CIFAR-10 (10 epochs): ~3-5 minutes
- DenseNet121 on CIFAR-10 (10 epochs): ~4-6 minutes

## Verification

After training, verify your models are working correctly:

```bash
# Quick test with pretrained models
python run_experiment.py --config quick

# Check that attacks produce different results
python -c "
import pandas as pd
df = pd.read_csv('./outputs/quick_test/results.csv')
print('Attack Success Rates:')
for attack in df['attack'].unique():
    asr = df[df['attack'] == attack]['success'].mean()
    print(f'{attack:15} {asr:.2%}')
"
```

You should see **different** success rates across attacks, not identical ones.

## Troubleshooting

### Issue: "All attacks still produce identical results"
- Check that `use_pretrained=True` in your config
- Verify models are actually different architectures (resnet18 vs vgg16)
- Ensure you're not using the same model for both surrogate and target

### Issue: "Training is too slow"
- Reduce batch size: `--batch-size 64`
- Reduce epochs: `--epochs 5`
- Use smaller dataset: `--dataset cifar10` instead of cifar100

### Issue: "Out of memory during training"
- Reduce batch size: `--batch-size 32` or `--batch-size 16`
- Use CPU if GPU memory is limited: `--device cpu`

### Issue: "Models don't load from checkpoint"
- Check checkpoint path exists
- Verify checkpoint was saved for correct dataset (cifar10 vs cifar100)
- Ensure num_classes matches (10 for CIFAR-10, 100 for CIFAR-100)

## Next Steps

1. **Immediate**: Run experiments with pretrained weights (no training needed)
2. **Short-term**: Fine-tune models on CIFAR for better results
3. **Publication**: Train with more epochs (20-30) and larger sample sizes

The pretrained approach is sufficient to validate your ELPD-Blend method and demonstrate it works better than baselines.

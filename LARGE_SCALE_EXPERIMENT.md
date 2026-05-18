# Large-Scale Power Likelihood Attacks Experiment

## Overview

This guide walks through running comprehensive large-scale experiments evaluating **ELPD-Blend** (power likelihood attacks) against baseline attacks on multiple datasets with statistical analysis.

## Quick Start

### 1. Validate Setup
```bash
# First, verify all components work on your device (MPS/CUDA/CPU)
python scripts/validate_exp_setup.py
python scripts/quick_test_experiment.py
```

Both should show ✓ ALL VALIDATIONS PASSED.

### 2. Run Small Experiment (Testing)
```bash
# Run on 20 samples (takes ~5-10 minutes on M1/M2)
python scripts/main_large_scale_experiment.py
```

This runs the default config:
- **Dataset**: CIFAR-10
- **Samples**: 50 per dataset  
- **Surrogate**: ResNet18 (white-box)
- **Target**: VGG16 (black-box)
- **Query Budget**: 1000 queries/image
- **Attacks**: MI-FGSM, NES-Only, Static Hybrid, ELPD-Blend

### 3. Run Full Experiment (Publication-Ready)
See **Configuration** section below to customize, then:
```bash
python scripts/main_large_scale_experiment.py
```

## Configuration

### Modifying Experiment Parameters

Edit `scripts/main_large_scale_experiment.py` in the `main()` function:

```python
def main():
    cfg = ExperimentConfig(
        # Datasets to test
        datasets=["cifar10", "cifar100"],  # Add "tinyimagenet" for more
        
        # Model architectures
        surrogate_model="resnet18",  # white-box model
        target_model="vgg16",         # black-box model ("densenet121" alternative)
        
        # Number of samples per dataset
        num_samples={"cifar10": 100, "cifar100": 200},
        
        # Attack parameters
        num_steps=25,              # Increase for stronger attacks
        query_budget=1000,         # Queries available per image
        epsilon=8.0 / 255.0,       # Perturbation budget
        
        # ELPD parameters
        elpd_method="waic",
        n_nes_samples=8,           # Gradient estimate samples
        nes_sigma=1.5e-2,          # NES noise level
        eta_ema_alpha=0.4,         # Blending factor smoothing
        
        # Device settings
        device_str=None,           # None = auto-detect, "mps", "cuda", "cpu"
        batch_size=None,           # None = auto-tune for device
        num_workers=None,          # None = auto-tune
        
        # Output
        output_dir="./outputs/large_scale_experiment",
    )
```

### Recommended Configurations

**Quick Test (2-5 min)**
```python
datasets=["cifar10"]
num_samples={"cifar10": 10}
num_steps=10
query_budget=100
```

**Medium Run (20-30 min)**
```python
datasets=["cifar10"]
num_samples={"cifar10": 50}
num_steps=15
query_budget=500
```

**Full Experiment (1-2 hours)**
```python
datasets=["cifar10", "cifar100"]
num_samples={"cifar10": 100, "cifar100": 200}
num_steps=25
query_budget=1000
```

**Publication-Ready (4-8 hours)**
```python
datasets=["cifar10", "cifar100"]
num_samples={"cifar10": 500, "cifar100": 500}
num_steps=30
query_budget=2000
```

## Understanding Results

### Output Structure
```
outputs/large_scale_experiment/
├── results.csv          # Raw per-sample results
├── summary.csv          # Aggregate statistics
└── experiment.log       # Full execution log
```

### Key Metrics

**ASR (Attack Success Rate)**: Percentage of adversarial examples that fool the target model

**Query Efficiency**: How many queries needed for successful attack  
- Lower = better (more efficient)

**L∞ Perturbation**: Maximum pixel change across all channels
- Lower = stealthier attack
- Must be ≤ ε

### Example Output

```
SUMMARY STATISTICS
Dataset: CIFAR-10
Attack        Success_Count  Total_Samples  ASR    Avg_Queries  Avg_L∞
mifgsm        32             100            32.0%  125.3        0.0314
nes_only      45             100            45.0%  287.5        0.0312
static_hybrid 52             100            52.0%  342.1        0.0313
elpd_blend    68             100            68.0%  415.7        0.0314

STATISTICAL COMPARISON
Dataset: CIFAR-10
  ELPD vs mifgsm: p=0.0021 (significant!)
  ELPD vs nes_only: p=0.0157 (significant)
  ELPD vs static_hybrid: p=0.0401 (borderline)
```

### Interpreting Statistical Significance

- **p < 0.05**: Statistically significant improvement (5% chance of random)
- **p < 0.01**: Highly significant (1% chance of random)
- **p > 0.05**: Not significant (may be random variation)

## Performance Optimization

### MPS (Apple Silicon) Tuning

The script auto-detects MPS and tunes batch sizes:
- **CIFAR-10**: batch=64 (2GB vram)
- **CIFAR-100**: batch=32 (4GB vram)

If you get OOM errors, reduce manually:
```python
batch_size=16  # For M1 with limited memory
```

### Parallel Processing

For distributed runs, split by dataset:
```bash
# Terminal 1: CIFAR-10
python scripts/main_large_scale_experiment.py  # Will process cifar10

# Terminal 2: CIFAR-100
# Edit script to: datasets=["cifar100"]
# Then run
python scripts/main_large_scale_experiment.py
```

### Faster Gradient Estimation

Reduce `n_nes_samples` (less accurate but faster):
```python
n_nes_samples=4  # Default 8
```

## Reproducibility

All runs use fixed seeds for reproducibility:
- **Seed**: 42 (configurable in ExperimentConfig)
- Enables: Same results every run
- Controls: Data shuffling, torch randomness, numpy randomness

To change seed:
```python
cfg = ExperimentConfig(seed=123)
```

## Troubleshooting

### Out of Memory (OOM)
```python
batch_size=16
num_workers=0
num_samples={"cifar10": 10}  # Reduce sample count
```

### Slow Dataset Loading
```python
num_workers=0  # Disable multiprocessing
pin_memory=False
```

### Timeout on Queries
```python
query_budget=500  # Reduce from 1000
num_steps=15      # Reduce from 25
```

### Model Not in Right Mode
- Models are automatically set to eval mode
- No training occurs during experiments

## Next Steps

After running experiments:

1. **Analyze Results**
   - Check `summary.csv` for aggregate metrics
   - Look for p < 0.05 in statistical tests
   - Identify which dataset/architecture combinations show best improvement

2. **Publication Checklist**
   - ✓ Multiple datasets (CIFAR-10, CIFAR-100)
   - ✓ Cross-architecture (ResNet18, VGG16, DenseNet)
   - ✓ Statistical significance (p-values)
   - ✓ Error bars / confidence intervals
   - ✓ Query budget constraints respected
   - ✓ Reproducible with seeds

3. **Generate Plots** (TODO in next phase)
   - Bar plots: ASR comparison
   - Error bars: ±1σ confidence intervals
   - Line plots: ASR vs query budget
   - Heatmaps: Per-class vulnerability

## Reference

- **Paper**: Black-and-White Power Likelihood Attacks
- **Method**: ELPD (Expected Log Predictive Density) based weighting
- **Key Idea**: Adaptively blend surrogate and target gradients using Bayesian model selection


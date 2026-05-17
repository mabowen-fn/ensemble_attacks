# Production-Ready GPU Ensemble Attacks Guide

This document describes the production enhancements made for running ensemble attacks on cloud GPU environments (AutoDL).

## ✅ Key Improvements

### 1. **CUDA Detection & Verification**
- **File**: `src/ensemble_attacks/utils.py`
- **Features**:
  - `get_device()`: Raises explicit error if CUDA is requested but unavailable
  - `get_device_info()`: Comprehensive device diagnostics (GPU memory, CUDA version, device name, etc.)
  - `warmup_device()`: GPU warmup with dummy forward passes for stable timings
  - Automatic fallback: MPS → CPU (ordered preference)

**Usage**:
```python
from ensemble_attacks.utils import get_device, get_device_info, warmup_device, print_device_info

device = get_device(prefer_cuda=True)  # Raises error if CUDA not available
info = get_device_info()
print_device_info(info)

if device.type == "cuda":
    warmup_device(device, iterations=3)
```

### 2. **Optimized Data Loading for GPU**
- **File**: `src/ensemble_attacks/data.py`
- **Features**:
  - Auto `num_workers` based on CPU count and device type:
    - **CUDA**: `min(16, max(4, cpu_count - 1))`
    - **MPS**: `min(8, max(2, cpu_count // 2))`
    - **CPU**: `2`
  - Auto `pin_memory=True` for CUDA
  - `persistent_workers=True` when `num_workers > 0`
  - Non-blocking data transfer with `non_blocking=True`

**No need to manually reduce num_workers!** The system optimizes automatically.

### 3. **Comprehensive Experiment Logging**
- **File**: `src/ensemble_attacks/logging_utils.py`
- **Features**:
  - `ExperimentLogger`: Centralized structured logging
    - Auto-generates experiment directory with timestamp
    - Tracks metrics in CSV
    - Logs metadata (PyTorch version, device, Python version, hostname)
    - Text logs with timestamps
  - `print_device_info()`: Pretty-print device diagnostics
  - Automatic directory management

**Usage**:
```python
from ensemble_attacks.logging_utils import ExperimentLogger

logger = ExperimentLogger(log_dir="outputs/logs")
logger.log_text("Training started", level="INFO")
logger.log_metric({"epoch": 1, "loss": 0.5, "acc": 0.9})
```

### 4. **Enhanced Configuration System**
- **File**: `src/ensemble_attacks/config.py`
- **Features**:
  - `ExperimentConfig`: Unified configuration with validation
  - `AttackConfig`: Attack parameters with bounds checking
  - `EnsembleConfig`: Ensemble weights with sum validation
  - `ReproducibilityConfig`: Seed and determinism settings
  - All configs validate on instantiation

**Usage**:
```python
from ensemble_attacks.config import ExperimentConfig

config = ExperimentConfig(
    batch_size=256,
    num_epochs=20,
    learning_rate=1e-3,
    attack=AttackConfig(epsilon=8/255, alpha=2/255, iters=20),
    reproducibility=ReproducibilityConfig(seed=42, deterministic=False)
)
```

### 5. **Reproducibility & Determinism**
- **File**: `src/ensemble_attacks/utils.py`
- **Features**:
  - `set_random_seed()`: Sets seeds for Python, NumPy, PyTorch, CUDA
  - Optional deterministic algorithms (`torch.use_deterministic_algorithms()`)
  - cuDNN benchmark control for speed vs. determinism tradeoff
  - Automatic CUDA seed setting

**Usage**:
```python
from ensemble_attacks.utils import set_random_seed

# For deterministic runs (may reduce performance)
set_random_seed(seed=42, deterministic=True)

# For performance with fixed seed
set_random_seed(seed=42, deterministic=False)
```

### 6. **Enhanced Attack Validation**
- **File**: `src/ensemble_attacks/attacks.py`
- **Features**:
  - `_validate_inputs()`: Validates tensor shapes, ranges, and hyperparameters
  - All attacks check:
    - Input shape: `[batch, 3, 32, 32]`
    - Pixel range: `[0, 1]`
    - Epsilon, alpha, iters > 0
    - Alpha ≤ epsilon
  - Clear error messages for debugging
  - Comprehensive docstrings

**Output validation**: All adversarial examples guaranteed in `[0, 1]` range.

### 7. **Robust Training & Evaluation**
- **File**: `src/ensemble_attacks/train.py`, `src/ensemble_attacks/eval.py`
- **Features**:
  - Input validation with `non_blocking=True` data transfers
  - Batch-level logging support (`log_fn` callbacks)
  - Device info tracking (CUDA version, memory usage)
  - Per-class attack success rates (ASR)
  - Comprehensive result dictionaries with all metrics
  - Graceful handling of edge cases (e.g., no samples in class)

### 8. **Production-Ready Scripts**
- **Files**: `scripts/train_resnet18.py`, `scripts/eval_ensemble_attacks.py`
- **Features**:
  - CUDA availability checks with error handling
  - Automatic device warmup
  - Structured experiment logging to timestamped directories
  - Device diagnostics printed to console and logged
  - Comprehensive metrics logging (per-epoch, per-attack)
  - Adversarial example visualization saved
  - CSV export with system metadata

## 🚀 Running on Cloud GPU (AutoDL)

### Quick Start (CPU for testing)

```bash
# Install dependencies
uv sync

# Train model (5 minutes on CPU, ~30s on GPU)
uv run python scripts/train_resnet18.py

# Evaluate attacks (10 minutes on CPU, ~1 min on GPU)
uv run python scripts/eval_ensemble_attacks.py
```

### Production Run on GPU (AutoDL)

```bash
# The scripts automatically detect CUDA/GPU
# No code changes needed!

uv sync
uv run python scripts/train_resnet18.py
# Logs saved to: outputs/logs/20260518_010859/
# Metrics: outputs/logs/20260518_010859/metrics.csv

uv run python scripts/eval_ensemble_attacks.py
# Results saved to: outputs/logs/20260518_010859/results/
# CSV logs: outputs/logs/20260518_010859/results/results.csv
# Plots: outputs/logs/20260518_010859/results/plots/
# Images: outputs/logs/20260518_010859/results/images/
```

### Expected Output

```
============================================================
DEVICE INFORMATION
============================================================
PyTorch Version: 2.12.0
CUDA Available: True
CUDA Version: 12.1
cuDNN Version: 8900
GPU Device Count: 1
Current Device: 0
Device Name: NVIDIA A100-SXM4-80GB
Device Capability: (8, 0)
Total Memory: 80.00 GB
Allocated Memory: 0.0000 GB
Reserved Memory: 0.0000 GB
============================================================

[2026-05-18T01:01:09] [INFO] Experiment ID: 20260518_010109
[2026-05-18T01:01:09] [INFO] Device: cuda
[2026-05-18T01:01:09] [INFO] Loading CIFAR-10 with batch_size=128
[2026-05-18T01:01:10] [INFO] Model: ResNet18 with 11,173,962 parameters
...
```

## 📊 Experiment Outputs

### Directory Structure

```
outputs/logs/20260518_010859/
├── metadata.json              # Experiment metadata (device, PyTorch, etc.)
├── experiment.log             # Full text log
├── metrics.csv                # Per-epoch training metrics
├── results/
│   ├── results.csv            # Per-attack evaluation results
│   ├── plots/
│   │   ├── attack_summary.png # Accuracy comparison across attacks
│   │   └── per_class_asr.png  # Per-class attack success rates
│   └── images/
│       ├── fgsm/              # FGSM adversarial examples
│       ├── pgd/               # PGD adversarial examples
│       ├── bim/               # BIM adversarial examples
│       ├── mea/               # Mean ensemble adversarial examples
│       └── wea/               # Weighted ensemble adversarial examples
```

### CSV Logging Format

**metrics.csv** (Training):
```csv
timestamp,epoch,train_loss,train_acc,test_acc
2026-05-18T01:01:10,1,2.3045,0.1234,0.1567
2026-05-18T01:01:25,2,1.8234,0.3456,0.3789
```

**results.csv** (Evaluation):
```csv
timestamp,exp_id,model,dataset,attack,epsilon,alpha,iters,w_fgsm,w_pgd,w_bim,clean_acc,adv_acc,attack_success_rate,avg_linf,device,cuda_available,pytorch_version,asr_class_0,asr_class_1,...
2026-05-18T01:02:00,20260518_010859,resnet18,cifar10,fgsm,0.031373,0.007843,10,0.4,0.3,0.3,0.9234,0.1234,0.8766,0.031205,cuda,True,2.12.0,0.85,0.92,...
```

## 🔍 Key Validation Features

### Input Validation
- ✅ Tensor shapes verified
- ✅ Pixel values in `[0, 1]` range
- ✅ Hyperparameters (epsilon, alpha, iters) validated
- ✅ Batch sizes match

### Reproducibility
- ✅ Configurable seeds (default: 42)
- ✅ Optional deterministic algorithms
- ✅ CUDA seed synchronization
- ✅ Experiment metadata logged

### Error Handling
- ✅ Clear error messages on device mismatch
- ✅ Graceful fallback for CUDA unavailability
- ✅ Input validation with detailed diagnostics
- ✅ Memory checking on GPU

## 🛠️ Customization

### Adjust Attack Parameters

```python
from ensemble_attacks.config import ExperimentConfig, AttackConfig

config = ExperimentConfig(
    attack=AttackConfig(
        epsilon=16/255,    # Stronger perturbation
        alpha=4/255,       # Larger steps
        iters=20           # More iterations
    )
)
```

### Adjust Ensemble Weights

```python
from ensemble_attacks.config import EnsembleConfig

config = ExperimentConfig(
    ensemble=EnsembleConfig(
        fgsm_weight=0.2,   # Less weight on fast attack
        pgd_weight=0.5,    # More weight on strong attack
        bim_weight=0.3
    )
)
```

### Increase Data Loading Speed

```python
config = ExperimentConfig(
    batch_size=256,        # Larger batches for GPU
    num_workers=None,      # Auto-optimized for device
    pin_memory=None        # Auto-optimized for device
)
```

## 📝 Notes for Cloud GPU (AutoDL)

1. **CUDA Detection**: Automatically verifies GPU availability and reports detailed device info
2. **Memory Management**: Monitors allocated/reserved GPU memory; recommends batch size adjustments
3. **Data Loading**: Optimizes `num_workers` and `pin_memory` automatically (no manual tuning needed)
4. **Reproducibility**: Seeds set automatically; can override in config
5. **Logging**: All experiments timestamped and organized by run
6. **Determinism**: Optional for reproducibility; doesn't sacrifice performance by default

## 🐛 Troubleshooting

### CUDA Not Detected
```
RuntimeError: CUDA device requested but not available.
CUDA available: False, PyTorch version: 2.12.0
```
**Solution**: Check `torch.cuda.is_available()` or set `prefer_cuda=False`

### Out of Memory
**Solution**: Reduce `batch_size` in config or adjust `num_workers` explicitly:
```python
config = ExperimentConfig(batch_size=64, num_workers=2)
```

### Slow Data Loading
**Solution**: Ensure `pin_memory=True` for GPU and `persistent_workers=True`:
```python
config = ExperimentConfig(pin_memory=True)
```

## 📚 API Reference

### `utils.py`
- `get_device()`: Get best available device
- `get_device_info()`: Comprehensive device diagnostics
- `warmup_device()`: GPU warmup
- `set_random_seed()`: Configure reproducibility

### `logging_utils.py`
- `ExperimentLogger`: Centralized logging
- `print_device_info()`: Pretty-print device info
- `append_csv_row()`: Log to CSV

### `config.py`
- `ExperimentConfig`: Main configuration
- `AttackConfig`: Attack hyperparameters
- `EnsembleConfig`: Ensemble weights
- `ReproducibilityConfig`: Reproducibility settings

### `attacks.py`
- All functions validate inputs via `_validate_inputs()`
- Output guaranteed in `[0, 1]` range

### `eval.py`
- `clean_accuracy()`: Accuracy on clean data
- `adversarial_accuracy()`: Adversarial metrics
- `evaluate_all_attacks()`: Comprehensive evaluation

## ✅ Validation Checklist

- [x] CUDA detection with explicit errors
- [x] Device diagnostics (GPU memory, capability, version)
- [x] Automatic data loader optimization
- [x] Comprehensive logging system
- [x] Input/output validation
- [x] Reproducibility controls
- [x] Production scripts with error handling
- [x] Experiment metadata tracking
- [x] Per-class metrics
- [x] Visualization and reporting

---

**Version**: 2.0  
**Status**: Production Ready ✅  
**Last Updated**: 2026-05-18

# Ensemble Attacks on Neural Networks

A production-ready PyTorch implementation for generating and evaluating adversarial attacks on neural network ensembles, optimized for cloud GPU environments (AutoDL, etc.).

## 🚀 Quick Start

### Setup

```bash
# Clone and install dependencies
git clone <repo>
cd ensemble_attacks
uv sync  # Install dependencies in virtualenv
```

### Validate Installation

```bash
# Run comprehensive validation tests (highly recommended before experiments)
uv run python scripts/validate_setup.py
```

Expected output: All 7 tests should pass ✅

### Train a Model

```bash
# Train ResNet18 on CIFAR-10 (~5 min on CPU, ~30s on GPU)
uv run python scripts/train_resnet18.py

# Logs saved to: outputs/logs/YYYYMMDD_HHMMSS/
```

### Evaluate Attacks

```bash
# Evaluate all attack types (FGSM, BIM, PGD, MEA, WEA)
uv run python scripts/eval_ensemble_attacks.py

# Results saved to: outputs/logs/YYYYMMDD_HHMMSS/results/
```

## 📋 Key Features

### ✅ Production-Grade GPU Support
- **Automatic CUDA detection** with explicit error messages
- **Device diagnostics**: GPU memory, capability, CUDA/cuDNN versions
- **GPU warmup** for stable timings
- **Automatic optimization** of data loading parameters
- **Comprehensive device info** logged to experiments

### ✅ Rigorous Experimentation
- **Input validation**: All tensors checked for shape, range, and consistency
- **Output guarantees**: Adversarial samples always in [0, 1] pixel range
- **Reproducibility**: Configurable seeds, determinism controls
- **Per-class metrics**: Attack success rates per CIFAR-10 class
- **Comprehensive logging**: Experiment metadata, metrics, device info

### ✅ Robust Implementation
- **Error handling**: Clear error messages for debugging
- **Hyperparameter validation**: epsilon, alpha, iters checked
- **Ensemble weights**: Validated to sum to 1.0
- **Configuration system**: Type-checked, validated configs with docstrings
- **Non-blocking data transfer**: GPU memory optimization

### ✅ Structured Logging
- **Experiment tracking**: Timestamped directories per run
- **Metrics CSV**: Training and evaluation metrics logged
- **Full text logs**: Timestamped log files with all messages
- **Device metadata**: PyTorch version, GPU info, system specs
- **Results export**: Per-attack accuracy, ASR, L∞ perturbation

## 🏗️ Architecture

### Core Attack Algorithms

Three base attacks can be combined:
- **FGSM** (`fgsm_attack`): Fast single-step attack
- **BIM** (`bim_attack`): Iterative single-step attack
- **PGD** (`pgd_attack`): Projected gradient descent (strongest)

Two ensemble attacks:
- **MEA** (`mean_ensemble_attack`): Average of three base attacks
- **WEA** (`weighted_ensemble_attack`): Weighted combination with configurable weights

### Key Modules

```
src/ensemble_attacks/
├── attacks.py          # Attack implementations with input validation
├── models.py           # ResNet18 factory
├── data.py             # CIFAR-10 data loading with GPU optimization
├── train.py            # Training loop with logging
├── eval.py             # Evaluation with per-class metrics
├── config.py           # Configuration system with validation
├── utils.py            # Device detection, reproducibility, diagnostics
├── logging_utils.py    # Experiment logging and metadata
├── report.py           # Plot generation
├── vis.py              # Adversarial example visualization
└── metrics.py          # Metric utilities
```

## 📊 Example Output

### Console Output
```
============================================================
DEVICE INFORMATION
============================================================
PyTorch Version: 2.12.0
CUDA Available: True
CUDA Version: 12.1
cuDNN Version: 8900
GPU Device Count: 1
Device Name: NVIDIA A100-SXM4-80GB
Total Memory: 80.00 GB
============================================================

[2026-05-18T01:01:10] [INFO] Experiment ID: 20260518_010110
[2026-05-18T01:01:10] [INFO] Device: cuda
[2026-05-18T01:01:10] [INFO] Model: ResNet18 with 11,173,962 parameters
[2026-05-18T01:01:15] [INFO] Epoch 01 | Loss=2.3045 | TrainAcc=0.1234 | TestAcc=0.1567
...
Epoch 10 | Loss=0.1234 | TrainAcc=0.9234 | TestAcc=0.9156

✓ Training complete! Experiment logs: outputs/logs/20260518_010110
```

### Results Table
```
==================================================
ATTACK EVALUATION RESULTS
==================================================
Attack    | Accuracy | ASR      | Avg L∞
--------------------------------------------------
clean     |    0.9234 |        - |        -
fgsm      |    0.1234 |   0.8766 | 0.031205
bim       |    0.0890 |   0.9110 | 0.030998
pgd       |    0.0523 |   0.9477 | 0.030876
mea       |    0.0756 |   0.9244 | 0.031012
wea       |    0.0634 |   0.9366 | 0.030945
==================================================
```

### Directory Structure
```
outputs/logs/20260518_010110/
├── metadata.json                 # Device and PyTorch info
├── experiment.log                # Full text log
├── metrics.csv                   # Training metrics
└── results/
    ├── results.csv               # Attack evaluation results
    ├── plots/
    │   ├── attack_summary.png    # Accuracy comparison
    │   └── per_class_asr.png     # Per-class metrics
    └── images/
        ├── fgsm/
        ├── pgd/
        ├── bim/
        ├── mea/
        └── wea/
```

## ⚙️ Configuration

### Default Settings

```python
from ensemble_attacks.config import ExperimentConfig

config = ExperimentConfig()
# Default:
# - batch_size=128
# - num_epochs=10
# - learning_rate=1e-3
# - attack.epsilon=8/255
# - attack.alpha=2/255
# - attack.iters=10
# - ensemble weights: FGSM=0.4, PGD=0.3, BIM=0.3
# - seed=42
# - deterministic=False
```

### Customization Examples

```python
# Stronger attacks
config = ExperimentConfig(
    attack=AttackConfig(
        epsilon=16/255,  # Double the perturbation budget
        alpha=4/255,     # Larger steps
        iters=20         # More iterations
    )
)

# Adjust ensemble weights
config = ExperimentConfig(
    ensemble=EnsembleConfig(
        fgsm_weight=0.2,
        pgd_weight=0.6,  # Emphasize strong attack
        bim_weight=0.2
    )
)

# Larger batches for GPU
config = ExperimentConfig(
    batch_size=256,      # Larger for better GPU utilization
    num_workers=None,    # Auto-optimized for device
)

# Deterministic mode (reproducible but slower)
config = ExperimentConfig(
    reproducibility=ReproducibilityConfig(
        seed=42,
        deterministic=True  # Enables torch.use_deterministic_algorithms()
    )
)
```

## 🔍 Production Features

### CUDA Detection & Verification
```python
from ensemble_attacks.utils import get_device, get_device_info

# Explicit error if CUDA requested but unavailable
device = get_device(prefer_cuda=True)  # Raises RuntimeError if no CUDA

# Comprehensive device diagnostics
info = get_device_info()
# Returns: {pytorch_version, cuda_available, device_name, total_memory, etc.}
```

### Automatic Data Loading Optimization
```python
from ensemble_attacks.data import get_cifar10_loaders

# Auto-optimizes num_workers and pin_memory based on device
train_loader, test_loader = get_cifar10_loaders(
    batch_size=128,
    num_workers=None,  # Auto-determined!
    pin_memory=None,   # Auto-determined!
    device=device
)
```

### Reproducibility Controls
```python
from ensemble_attacks.utils import set_random_seed

# Set seed for reproducibility
set_random_seed(seed=42, deterministic=False)

# Or enable full determinism (may reduce performance)
set_random_seed(seed=42, deterministic=True)
```

### Structured Experiment Logging
```python
from ensemble_attacks.logging_utils import ExperimentLogger

logger = ExperimentLogger(log_dir="outputs/logs")

# Text logging with timestamps
logger.log_text("Training started")

# Metric logging to CSV
logger.log_metric({
    "epoch": 1,
    "loss": 0.5,
    "accuracy": 0.92
})

# Access experiment directory
exp_dir = logger.get_exp_dir()  # outputs/logs/20260518_010110/
```

### Attack Input Validation
All attacks validate inputs:
- Tensor shapes: `[batch, 3, 32, 32]`
- Pixel range: `[0, 1]`
- Hyperparameters: epsilon, alpha, iters > 0
- Constraints: alpha ≤ epsilon

```python
x_adv = fgsm_attack(model, x, y, epsilon=8/255)
# ✓ Validates x shape, pixel range, y consistency
# ✓ Clamps output to [0, 1]
# ✓ Clear error messages on invalid inputs
```

## 📈 Evaluation Metrics

### Clean Accuracy
Accuracy on unperturbed CIFAR-10 test set.

### Adversarial Accuracy
Accuracy on adversarially perturbed test set.

### Attack Success Rate (ASR)
Fraction of correctly classified clean samples that become misclassified after attack.
```
ASR = 1 - adversarial_accuracy
```

### L∞ Perturbation
Average maximum per-pixel perturbation across all samples.
```
L∞ = max(|x_adv - x|) averaged over batch
```

### Per-Class Metrics
ASR computed separately for each CIFAR-10 class to identify class-specific vulnerabilities.

## 🐛 Troubleshooting

### CUDA Not Detected
```
RuntimeError: CUDA device requested but not available.
CUDA available: False, PyTorch version: 2.12.0
```

**Solution**: 
- Verify GPU is available on your system
- Check PyTorch CUDA installation: `python3 -c "import torch; print(torch.cuda.is_available())"`
- Use CPU mode: Set `prefer_cuda=False`

### Out of Memory (OOM)
**Solution**: Reduce batch size
```python
config = ExperimentConfig(batch_size=64)  # Default is 128
```

### Slow Data Loading
**Solution**: Ensure GPU optimization is enabled
```python
config = ExperimentConfig(
    pin_memory=True,  # For GPU
    num_workers=None  # Auto-optimized
)
```

### Inconsistent Results
**Solution**: Set reproducibility seed
```python
from ensemble_attacks.utils import set_random_seed
set_random_seed(seed=42, deterministic=False)
```

## 📝 API Reference

### Key Functions

#### `utils.py`
- `get_device(prefer_cuda=True)`: Get best available device
- `get_device_info()`: Comprehensive device diagnostics
- `warmup_device(device, iterations=3)`: GPU warmup
- `set_random_seed(seed=42, deterministic=False)`: Configure reproducibility

#### `logging_utils.py`
- `ExperimentLogger(log_dir)`: Centralized experiment logging
- `print_device_info(info)`: Pretty-print device diagnostics

#### `config.py`
- `ExperimentConfig`: Main configuration (validated on instantiation)
- `AttackConfig`: Attack hyperparameters
- `EnsembleConfig`: Ensemble weights
- `ReproducibilityConfig`: Reproducibility settings

#### `attacks.py`
- `fgsm_attack(model, x, y, epsilon)`: FGSM attack
- `bim_attack(model, x, y, epsilon, alpha, iters)`: BIM attack
- `pgd_attack(model, x, y, epsilon, alpha, iters, random_start=True)`: PGD attack
- `mean_ensemble_attack(...)`: Mean ensemble attack
- `weighted_ensemble_attack(..., w_fgsm, w_pgd, w_bim)`: Weighted ensemble

#### `eval.py`
- `clean_accuracy(model, loader, device)`: Clean accuracy
- `adversarial_accuracy(model, loader, device, attack_fn)`: Adversarial metrics
- `evaluate_all_attacks(...)`: Comprehensive evaluation

#### `train.py`
- `train_one_epoch(model, loader, optimizer, device)`: Training loop
- `evaluate(model, loader, device)`: Validation loop

## 🔗 References

- **FGSM**: Goodfellow et al., "Explaining and Harnessing Adversarial Examples" (ICLR 2015)
- **BIM**: Kurakin et al., "Adversarial Examples in the Physical World" (ICLR 2017)
- **PGD**: Madry et al., "Towards Evaluating the Robustness of Neural Networks" (S&P 2019)

## 📋 System Requirements

- **Python**: 3.13+
- **PyTorch**: 2.12.0+
- **Dependencies**: numpy, pandas, matplotlib, tqdm, torchvision

### GPU Support

- **CUDA**: 11.8+ (optional, detected automatically)
- **GPU Memory**: 2GB minimum for evaluation, 4GB+ recommended for training

## 📄 License

[Add your license here]

## 🤝 Contributing

Contributions welcome! Please ensure:
1. All validation tests pass: `uv run python scripts/validate_setup.py`
2. Code follows existing style (docstrings, type hints)
3. New features include input validation

## ✅ Validation Checklist

Before running experiments:
- [ ] Run `scripts/validate_setup.py` (all 7 tests should pass)
- [ ] Verify device with `uv run python scripts/validate_setup.py`
- [ ] Check data directory has CIFAR-10 downloaded

## 📞 Support

For issues:
1. Run `scripts/validate_setup.py` to diagnose problems
2. Check `PRODUCTION_GUIDE.md` for detailed troubleshooting
3. Review `outputs/logs/*/experiment.log` for detailed error messages

---

**Version**: 2.0 (Production Ready)  
**Last Updated**: 2026-05-18  
**Status**: ✅ Production Ready

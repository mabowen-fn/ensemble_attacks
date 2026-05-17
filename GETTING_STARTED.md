# Getting Started - Production-Ready Ensemble Attacks

## ✅ Pre-Launch Checklist

- [ ] Clone repository: `git clone <repo>`
- [ ] Install dependencies: `uv sync`
- [ ] Run validation: `uv run python scripts/validate_setup.py`
- [ ] All 7 tests pass ✅
- [ ] Review `PRODUCTION_GUIDE.md` for detailed info

## 🚀 Quick Start (3 Steps)

### Step 1: Validate Setup (1 minute)
```bash
uv run python scripts/validate_setup.py
```
**Expected Output**:
```
Result: 7/7 tests passed
✅ All validation tests passed!
Ready to run production experiments.
```

### Step 2: Train Model (5 min on CPU, ~30s on GPU)
```bash
uv run python scripts/train_resnet18.py
```
**Expected Output**:
```
DEVICE INFORMATION
...
Epoch 10 | Loss=0.1234 | TrainAcc=0.9234 | TestAcc=0.9156
✓ Training complete! Experiment logs: outputs/logs/20260518_010110/
```

### Step 3: Evaluate Attacks (10 min on CPU, ~1 min on GPU)
```bash
uv run python scripts/eval_ensemble_attacks.py
```
**Expected Output**:
```
ATTACK EVALUATION RESULTS
clean    |    0.9234 |        - |        -
fgsm     |    0.1234 |   0.8766 | 0.031205
pgd      |    0.0523 |   0.9477 | 0.030876
wea      |    0.0634 |   0.9366 | 0.030945
✓ Evaluation complete!
```

## 📊 Where to Find Results

After running the scripts, check:

```
outputs/logs/20260518_HHMMSS/
├── metadata.json              # Device & system info
├── experiment.log             # Full text log
├── metrics.csv                # Training metrics (train loss, accuracy)
└── results/
    ├── results.csv            # Attack evaluation results
    ├── plots/
    │   ├── attack_summary.png # Accuracy comparison
    │   └── per_class_asr.png  # Per-class attack success rates
    └── images/
        ├── fgsm/              # Adversarial examples (FGSM)
        ├── pgd/               # Adversarial examples (PGD)
        ├── bim/               # Adversarial examples (BIM)
        ├── mea/               # Adversarial examples (MEA)
        └── wea/               # Adversarial examples (WEA)
```

## 🔧 Customization

### Change Attack Parameters
Edit the attack config in your experiment:

```python
from ensemble_attacks.config import ExperimentConfig, AttackConfig

config = ExperimentConfig(
    attack=AttackConfig(
        epsilon=16/255,  # Stronger attacks
        alpha=4/255,     # Larger steps
        iters=20         # More iterations
    )
)
```

### Change Ensemble Weights
```python
from ensemble_attacks.config import EnsembleConfig

config = ExperimentConfig(
    ensemble=EnsembleConfig(
        fgsm_weight=0.2,
        pgd_weight=0.6,  # Emphasize strong attack
        bim_weight=0.2
    )
)
```

### Increase Batch Size for GPU
```python
config = ExperimentConfig(
    batch_size=256,      # 128 is default
    num_workers=None,    # Auto-optimized (don't reduce!)
)
```

## 🐛 Troubleshooting

### Q: "CUDA device requested but not available"
**A**: GPU not found. Either:
- Use CPU mode: set `prefer_cuda=False`
- Verify GPU: `uv run python -c "import torch; print(torch.cuda.is_available())"`

### Q: "Out of memory"
**A**: Reduce batch size:
```python
config = ExperimentConfig(batch_size=64)  # Default is 128
```

### Q: "Data loading is slow"
**A**: Ensure GPU optimization:
```python
config = ExperimentConfig(
    pin_memory=True,   # For GPU
    num_workers=None   # Auto-optimized (never reduce!)
)
```

### Q: "Results differ between runs"
**A**: Set seed:
```python
from ensemble_attacks.config import ReproducibilityConfig
config = ExperimentConfig(
    reproducibility=ReproducibilityConfig(seed=42)
)
```

## 📈 Understanding Results

### Clean Accuracy
Accuracy on unperturbed test images. Goal: as high as possible.

### Adversarial Accuracy
Accuracy after attack. Goal: understand model robustness.

### Attack Success Rate (ASR)
`ASR = 1 - Adversarial_Accuracy`
- FGSM: Baseline (fastest, weakest)
- BIM: Intermediate (iterative single-step)
- PGD: Strongest (optimal attack)
- MEA: Mean ensemble (average of three)
- WEA: Weighted ensemble (customizable weights)

### Per-Class ASR
Identifies which classes are most vulnerable. Important for understanding attack patterns.

### Avg L∞ Perturbation
Average maximum per-pixel change. Lower = more imperceptible attacks.

## 🎯 Production Tips

### Before Major Experiments
1. Always run validation first: `uv run python scripts/validate_setup.py`
2. Check GPU memory: Look at device diagnostics output
3. Review metadata in logs: Ensure right device/PyTorch version

### During Experiments
1. Monitor `outputs/logs/YYYYMMDD_HHMMSS/experiment.log` for errors
2. Check disk space (each experiment can generate large images)
3. Keep track of experiment IDs for later reference

### After Experiments
1. Archive `outputs/logs/YYYYMMDD_HHMMSS/` for reproducibility
2. Review CSV results for trends
3. Compare plots across multiple runs

## 📝 Quick Reference

### Key Functions
```python
from ensemble_attacks.utils import get_device, get_device_info, set_random_seed
from ensemble_attacks.logging_utils import ExperimentLogger
from ensemble_attacks.config import ExperimentConfig

# Device
device = get_device(prefer_cuda=True)
info = get_device_info()

# Logging
logger = ExperimentLogger()
logger.log_text("Message")
logger.log_metric({"metric": value})

# Config
config = ExperimentConfig()
config.batch_size = 256
```

### Command Reference
```bash
# Validate setup
uv run python scripts/validate_setup.py

# Quick 1-epoch test
uv run python scripts/quick_test.py

# Full training
uv run python scripts/train_resnet18.py

# Evaluate all attacks
uv run python scripts/eval_ensemble_attacks.py

# List experiments
ls outputs/logs/

# View experiment log
cat outputs/logs/20260518_HHMMSS/experiment.log

# View results
cat outputs/logs/20260518_HHMMSS/results/results.csv
```

## 🔗 Documentation
- **README.md**: Overview and features
- **PRODUCTION_GUIDE.md**: Detailed production guide
- **COMPLETION_SUMMARY.md**: Technical changes summary
- **GETTING_STARTED.md**: This file

## ✅ Success Criteria

You've successfully set up when:
- ✅ All 7 validation tests pass
- ✅ Training completes with >90% test accuracy
- ✅ Evaluation runs all 5 attacks successfully
- ✅ Results saved to `outputs/logs/` with CSV and plots
- ✅ Device info shows correct GPU/CPU

## 🚀 Ready to Launch!

You're now ready for production experiments on AutoDL or any cloud GPU environment.

**Next**: `uv run python scripts/validate_setup.py`

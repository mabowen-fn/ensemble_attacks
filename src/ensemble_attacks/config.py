from dataclasses import dataclass, field


@dataclass
class AttackConfig:
    """Configuration for adversarial attacks."""
    epsilon: float = 8 / 255
    alpha: float = 2 / 255
    iters: int = 10
    
    def __post_init__(self):
        """Validate attack parameters."""
        assert self.epsilon > 0, "epsilon must be positive"
        assert self.alpha > 0, "alpha must be positive"
        assert self.iters > 0, "iters must be positive"
        assert self.alpha <= self.epsilon, "alpha should not exceed epsilon"


@dataclass
class EnsembleConfig:
    """Configuration for ensemble attacks."""
    fgsm_weight: float = 0.4
    pgd_weight: float = 0.3
    bim_weight: float = 0.3
    
    def __post_init__(self):
        """Validate weights sum to 1.0."""
        total = self.fgsm_weight + self.pgd_weight + self.bim_weight
        assert abs(total - 1.0) < 1e-6, f"weights must sum to 1.0, got {total}"


@dataclass
class ReproducibilityConfig:
    """Configuration for reproducibility and determinism."""
    seed: int | None = 42
    deterministic: bool = False  # Enable deterministic algorithms (may reduce perf)
    benchmark: bool = True  # Enable cuDNN benchmarking for speed
    
    def __post_init__(self):
        """Validate reproducibility settings."""
        if self.deterministic and self.benchmark:
            import warnings
            warnings.warn(
                "deterministic=True may conflict with benchmark=True. "
                "Consider setting benchmark=False for full determinism."
            )


@dataclass
class ExperimentConfig:
    """Full experiment configuration combining all aspects."""
    attack: AttackConfig = field(default_factory=AttackConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    reproducibility: ReproducibilityConfig = field(default_factory=ReproducibilityConfig)
    
    # Training
    batch_size: int = 128
    num_epochs: int = 10
    learning_rate: float = 1e-3
    
    # Data loading
    num_workers: int | None = None  # Auto-determined based on device
    pin_memory: bool | None = None  # Auto-determined based on device
    data_dir: str = "./data"
    
    # Device
    prefer_cuda: bool = True
    device_warmup: bool = True
    device_warmup_iters: int = 3
    
    # Logging
    verbose: bool = True
    log_dir: str = "outputs/logs"
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.num_epochs > 0, "num_epochs must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"
        if self.num_workers is not None:
            assert self.num_workers >= 0, "num_workers must be non-negative"

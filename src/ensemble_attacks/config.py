from dataclasses import dataclass

@dataclass
class AttackConfig:
    epsilon: float = 8 / 255
    alpha: float = 2 / 255
    iters: int = 10

@dataclass
class EnsembleConfig:
    fgsm_weight: float = 0.4
    pgd_weight: float = 0.3
    bim_weight: float = 0.3

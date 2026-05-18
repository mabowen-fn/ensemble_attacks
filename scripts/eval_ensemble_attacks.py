import os
import torch

from ensemble_attacks.data import get_cifar10_loaders
from ensemble_attacks.models import get_resnet18
from ensemble_attacks.eval import evaluate_all_attacks
from ensemble_attacks.utils import get_device, get_device_info, warmup_device, set_random_seed
from ensemble_attacks.config import ExperimentConfig
from ensemble_attacks.logging_utils import ExperimentLogger, print_device_info, append_csv_row, ensure_dir, now_timestamp
from ensemble_attacks.report import plot_attack_summary, plot_per_class_asr


def main():
    # Initialize experiment config
    config = ExperimentConfig()
    
    # Setup device
    try:
        device = get_device(prefer_cuda=config.prefer_cuda)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return False
    
    device_info = get_device_info()
    print_device_info(device_info)
    
    # Warmup device if on GPU
    if config.device_warmup and device.type == "cuda":
        print("Warming up CUDA device...")
        warmup_device(device, iterations=config.device_warmup_iters)
    
    # Set random seeds for reproducibility
    set_random_seed(
        seed=config.reproducibility.seed,
        deterministic=config.reproducibility.deterministic
    )
    
    # Initialize experiment logger
    logger = ExperimentLogger(log_dir=config.log_dir)
    
    def log_fn(msg):
        logger.log_text(msg)
    
    log_fn(f"Experiment ID: {logger.exp_id}")
    log_fn(f"Device: {device}")
    log_fn(f"Config: {config}")
    
    # Load data
    log_fn(f"Loading CIFAR-10 with batch_size={config.batch_size}")
    _, test_loader = get_cifar10_loaders(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        data_dir=config.data_dir,
        device=device,
    )
    
    log_fn(f"Test batches: {len(test_loader)}")
    
    # Load model
    log_fn("Loading pre-trained ResNet18 model...")
    model = get_resnet18()
    state = torch.load("resnet18_cifar10.pt", map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    
    log_fn(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Attack parameters from config
    epsilon = config.attack.epsilon
    alpha = config.attack.alpha
    iters = config.attack.iters
    
    log_fn(f"Attack config: epsilon={epsilon:.6f}, alpha={alpha:.6f}, iters={iters}")
    log_fn(f"Ensemble weights: FGSM={config.ensemble.fgsm_weight}, PGD={config.ensemble.pgd_weight}, BIM={config.ensemble.bim_weight}")
    
    # Setup output directories
    exp_dir = logger.get_exp_dir()
    out_dir = os.path.join(exp_dir, "results")
    img_dir = os.path.join(out_dir, "images")
    ensure_dir(out_dir)
    ensure_dir(img_dir)
    
    log_fn(f"Evaluating attacks (this may take a while)...")
    
    # Evaluate all attacks
    results = evaluate_all_attacks(
        model, test_loader, device,
        epsilon=epsilon,
        alpha=alpha,
        iters=iters,
        w_fgsm=config.ensemble.fgsm_weight,
        w_pgd=config.ensemble.pgd_weight,
        w_bim=config.ensemble.bim_weight,
        save_dir=img_dir,
        log_fn=log_fn
    )
    
    # Print results table
    log_fn("\n" + "="*50)
    log_fn("ATTACK EVALUATION RESULTS")
    log_fn("="*50)
    log_fn(f"{'Attack':>6} | {'Accuracy':>10} | {'ASR':>8} | {'Avg L∞':>10}")
    log_fn("-" * 50)
    
    log_fn(f"{'clean':>6} | {results['clean']:>10.4f} | {'-':>8} | {'-':>10}")
    
    for attack_name in ["fgsm", "bim", "pgd", "mea", "wea"]:
        acc = results[attack_name]
        asr = 1.0 - acc
        linf = results[f"{attack_name}_linf"]
        log_fn(f"{attack_name:>6} | {acc:>10.4f} | {asr:>8.4f} | {linf:>10.6f}")
    
    log_fn("="*50 + "\n")
    
    # Generate plots
    log_fn("Generating plots...")
    report_dir = os.path.join(out_dir, "plots")
    plot_attack_summary(results, report_dir)
    plot_per_class_asr(results, report_dir)
    log_fn(f"Plots saved to: {report_dir}")
    
    # Log results to CSV with comprehensive metadata
    csv_path = os.path.join(out_dir, "results.csv")
    
    for attack_name in ["fgsm", "bim", "pgd", "mea", "wea"]:
        adv_acc = results[attack_name]
        avg_linf = results[f"{attack_name}_linf"]
        per_class_asr = results[f"{attack_name}_asr_pc"]
        
        row = {
            "timestamp": now_timestamp(),
            "exp_id": logger.exp_id,
            "model": "resnet18",
            "dataset": "cifar10",
            "attack": attack_name,
            "epsilon": epsilon,
            "alpha": alpha,
            "iters": iters,
            "w_fgsm": config.ensemble.fgsm_weight,
            "w_pgd": config.ensemble.pgd_weight,
            "w_bim": config.ensemble.bim_weight,
            "clean_acc": results["clean"],
            "adv_acc": adv_acc,
            "attack_success_rate": 1.0 - adv_acc,
            "avg_linf": avg_linf,
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "pytorch_version": torch.__version__,
        }
        
        for c in range(10):
            row[f"asr_class_{c}"] = per_class_asr[c]
        
        append_csv_row(csv_path, row)
    
    log_fn(f"Results logged to: {csv_path}")
    log_fn(f"Adversarial images saved to: {img_dir}")
    log_fn(f"✓ Evaluation complete! All results in: {exp_dir}")
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

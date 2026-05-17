import torch
import torch.optim as optim

from ensemble_attacks.data import get_cifar10_loaders
from ensemble_attacks.models import get_resnet18
from ensemble_attacks.train import train_one_epoch, evaluate
from ensemble_attacks.utils import get_device, get_device_info, warmup_device, set_random_seed
from ensemble_attacks.config import ExperimentConfig
from ensemble_attacks.logging_utils import ExperimentLogger, print_device_info


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
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        data_dir=config.data_dir,
        device=device,
    )
    
    log_fn(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    
    # Initialize model
    model = get_resnet18().to(device)
    log_fn(f"Model: ResNet18 with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    log_fn(f"Optimizer: Adam (lr={config.learning_rate})")
    
    # Training loop
    for epoch in range(1, config.num_epochs + 1):
        loss, acc = train_one_epoch(model, train_loader, optimizer, device, log_fn=log_fn)
        test_acc = evaluate(model, test_loader, device, log_fn=log_fn)
        
        msg = f"Epoch {epoch:02d} | Loss={loss:.4f} | TrainAcc={acc:.4f} | TestAcc={test_acc:.4f}"
        log_fn(msg)
        print(msg)
        
        # Log to CSV
        logger.log_metric({
            "epoch": epoch,
            "train_loss": loss,
            "train_acc": acc,
            "test_acc": test_acc,
        })
    
    # Save model
    model_path = "resnet18_cifar10.pt"
    torch.save(model.state_dict(), model_path)
    log_fn(f"Saved model to {model_path}")
    
    print(f"\n✓ Training complete! Experiment logs: {logger.exp_dir}")
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

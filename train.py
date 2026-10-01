"""Training loop for segmentation experiments (SGD, warmup, LR scheduling)."""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import copy
from tqdm import tqdm

from datasets import get_dataloaders, get_dataset_info
from utils import warmup_lr


class EarlyStopping:
    """Stop training when a monitored metric has stopped improving.

    Keeps a copy of the best model weights and restores them when stopping.
    """

    def __init__(self, patience, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.enabled = patience is not None and patience > 0
        self.best_score = None
        self.best_state = None
        self.best_epoch = 0
        self.counter = 0
        self.should_stop = False

    def step(self, val_miou, model, epoch):
        score = val_miou
        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            self.best_state = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.enabled and self.counter >= self.patience:
                self.should_stop = True

    def restore_best(self, model):
        """Load the best weights back into *model*."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def compute_confusion_matrix(logits, target, num_classes=21, ignore_index=255):
    pred = torch.argmax(logits, dim=1)
    valid = target != ignore_index

    pred = pred[valid]
    target = target[valid]

    if target.numel() == 0:
        return torch.zeros((num_classes, num_classes), dtype=torch.int64, device=logits.device)

    indices = target * num_classes + pred
    conf = torch.bincount(indices, minlength=num_classes * num_classes)
    return conf.view(num_classes, num_classes)


def compute_miou_and_pixel_acc(conf):
    conf = conf.float()
    tp = conf.diag()
    fp = conf.sum(dim=0) - tp
    fn = conf.sum(dim=1) - tp
    denom = tp + fp + fn
    valid = denom > 0

    iou = torch.zeros_like(tp)
    iou[valid] = tp[valid] / denom[valid]

    mean_iou = iou[valid].mean().item() if valid.any() else 0.0
    pixel_acc = (tp.sum() / conf.sum()).item() if conf.sum() > 0 else 0.0
    per_class_iou = iou.cpu().tolist()
    return mean_iou, pixel_acc, per_class_iou


def run_empirical_analysis(model, train_params, dataset_config, device):
    """Train a segmentation model and collect empirical metrics.

    Empirical evaluation methodology (adapted from no_free_lunch_architectures-main):
      Reference uses classification on CIFAR (accuracy, CE loss).
      This project adapts to semantic segmentation (mIoU, pixel accuracy, CE loss).

    Metrics collected and their correspondence to the reference:
      - expressivity_converged_train_loss:  final training loss at convergence.
        Analogous to reference's converged training loss/accuracy.
      - convergence_epochs_to_threshold:  epochs to reach 50% of initial loss.
        Measures optimisation speed (reference tracks training curves similarly).
      - generalization_gap:  val_loss - train_loss at the best epoch.
        Reference: test_loss - train_loss  (or test_acc - train_acc).
      - val_mIoU / val_pixel_acc:  segmentation-specific quality metrics
        (no direct analogue in reference's classification setup).
    """
    print("Starting empirical analysis (training for segmentation)...")

    # 1. Create dataloaders via dataset registry
    dataset_name = dataset_config.get("dataset_name", "VOC2012")
    dataset_root = dataset_config.get("path")
    num_classes, ignore_index = get_dataset_info(dataset_name)

    dataloader, val_loader = get_dataloaders(
        dataset_name,
        root=dataset_root,
        img_size=dataset_config["img_size"],
        batch_size=train_params["batch_size"],
        num_workers=dataset_config.get("num_workers", 4),
    )

    # 2. Initialize optimizer and multiclass loss
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    optimizer = optim.SGD(model.parameters(), lr=train_params['lr'],
                          momentum=train_params.get('momentum', 0),
                          weight_decay=train_params.get('weight_decay', 0))

    # LR scheduler (multi-step)
    decreasing_lr = train_params.get('decreasing_lr', None)
    scheduler = None
    if decreasing_lr:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)

    warmup_epochs = train_params.get('warmup', 0)
    early_stopping = EarlyStopping(patience=train_params['patience'])

    # 3. Training loop and metrics collection
    train_loss_history = []
    val_loss_history = []
    val_miou_history = []
    val_pixel_acc_history = []
    grad_norm_history = []            # per-epoch mean gradient norm
    epochs_to_convergence_threshold = -1  # Use -1 to indicate not reached
    loss_convergence_threshold = None

    epochs = train_params['epochs']
    pbar = tqdm(range(epochs), desc=f"Training Epochs")
    for epoch in pbar:
        model.train()
        epoch_loss = 0
        epoch_grad_norm = 0.0
        for i, (images, masks) in enumerate(dataloader):
            if epoch < warmup_epochs:
                warmup_lr(warmup_epochs, train_params['lr'], epoch, i + 1,
                          optimizer, one_epoch_step=len(dataloader))

            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            # Record gradient norm before the optimizer step (no clipping applied)
            batch_norm = sum(
                p.grad.detach().norm().item() ** 2
                for p in model.parameters() if p.grad is not None
            ) ** 0.5
            epoch_grad_norm += batch_norm
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(dataloader)
        avg_grad_norm = epoch_grad_norm / len(dataloader)
        train_loss_history.append(avg_train_loss)
        grad_norm_history.append(avg_grad_norm)

        # Set convergence threshold after epoch 3 (average of first 3 epochs
        # is more stable than epoch-0 alone, which can be noisy).
        if epoch == 2 and loss_convergence_threshold is None:
            loss_convergence_threshold = sum(train_loss_history[:3]) / 3 * 0.5

        # Check for convergence (only start checking from epoch 4 onwards)
        if (epoch >= 3
                and epochs_to_convergence_threshold == -1
                and loss_convergence_threshold is not None):
            if avg_train_loss < loss_convergence_threshold:
                epochs_to_convergence_threshold = epoch + 1

        # Validation step
        model.eval()
        val_epoch_loss = 0
        conf_total = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
        with torch.no_grad():
            for val_images, val_masks in val_loader:
                val_images, val_masks = val_images.to(device), val_masks.to(device)
                val_outputs = model(val_images)
                val_epoch_loss += criterion(val_outputs, val_masks).item()
                conf_total += compute_confusion_matrix(val_outputs, val_masks, num_classes=num_classes, ignore_index=ignore_index)
        
        avg_val_loss = val_epoch_loss / len(val_loader)
        avg_val_miou, avg_val_pixel_acc, _ = compute_miou_and_pixel_acc(conf_total)
        val_loss_history.append(avg_val_loss)
        val_miou_history.append(avg_val_miou)
        val_pixel_acc_history.append(avg_val_pixel_acc)
        
        pbar.set_postfix(
            {
                "Train Loss": f"{avg_train_loss:.4f}",
                "Val Loss": f"{avg_val_loss:.4f}",
                "Val mIoU": f"{avg_val_miou:.4f}",
                "Val Acc": f"{avg_val_pixel_acc:.4f}",
            }
        )

        if scheduler:
            scheduler.step()

        early_stopping.step(avg_val_miou, model, epoch + 1)
        if early_stopping.should_stop:
            print(f"Early stopping triggered at epoch {epoch + 1} "
                  f"(no improvement for {early_stopping.patience} epochs).")
            break

    early_stopping.restore_best(model)
    best_ep = early_stopping.best_epoch  # 1-indexed
    print(f"Training complete (best model from epoch {best_ep} restored).")

    # 4a. Re-evaluate best model to get per-class IoU
    model.eval()
    conf_best = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
    with torch.no_grad():
        for val_images, val_masks in val_loader:
            val_images, val_masks = val_images.to(device), val_masks.to(device)
            val_outputs = model(val_images)
            conf_best += compute_confusion_matrix(
                val_outputs, val_masks, num_classes=num_classes, ignore_index=ignore_index)
    _, _, best_per_class_iou = compute_miou_and_pixel_acc(conf_best)

    # 4b. Calculate final metrics (from best epoch, matching the restored model)
    best_idx = best_ep - 1
    final_train_loss = train_loss_history[best_idx]
    final_val_loss = val_loss_history[best_idx]

    # Expressivity: Converged training loss
    expressivity_metric = final_train_loss

    # Convergence: Epochs to reach 50% of initial training loss
    convergence_metric = epochs_to_convergence_threshold

    # Generalization: Gap between converged validation and training loss
    generalization_gap = final_val_loss - final_train_loss

    results = {
        "expressivity_converged_train_loss": expressivity_metric,
        "convergence_epochs_to_threshold_50_percent": convergence_metric,
        "generalization_gap": generalization_gap,
        "final_val_loss": final_val_loss,
        "final_val_miou": val_miou_history[best_idx],
        "final_val_pixel_acc": val_pixel_acc_history[best_idx],
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history,
        "val_miou_history": val_miou_history,
        "val_pixel_acc_history": val_pixel_acc_history,
        "grad_norm_history": grad_norm_history,
        "per_class_iou_best_epoch": best_per_class_iou,
        "meta_convergence_threshold_value": loss_convergence_threshold,
        "best_epoch": best_ep,
        "total_epochs_trained": len(train_loss_history),
    }
    
    return results

"""Utility helpers: seeding, model builder, warmup LR, config I/O."""

import torch
import os
import json
import random
import numpy as np


def save_results(data, directory, filename):
    """Saves dictionary data to a JSON file."""
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)

    def convert(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, torch.Tensor):
            return o.cpu().numpy().tolist()
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

    with open(filepath, "w") as f:
        json.dump(data, f, indent=4, default=convert)
    print(f"Results saved to {filepath}")


def setup_device(gpu_id=None):
    """Setup compute device. Respects CUDA_VISIBLE_DEVICES if set."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if gpu_id is not None:
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cuda")


def set_seed(seed=None):
    """Set random seed for reproducibility. Returns the seed used."""
    if seed is None:
        seed = random.randint(0, 9999)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def enable_cudnn_benchmark():
    """Enable cuDNN benchmark for faster training on fixed-size inputs."""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True


def warmup_lr(warmup_epoch, lr, epoch, step, optimizer, one_epoch_step):
    """Linear warmup: ramp LR from 0 to *lr* over *warmup_epoch* epochs."""
    overall_steps = warmup_epoch * one_epoch_step
    current_steps = epoch * one_epoch_step + step
    cur_lr = lr * current_steps / overall_steps
    cur_lr = min(cur_lr, lr)
    for p in optimizer.param_groups:
        p['lr'] = cur_lr


def count_parameters(model):
    """Return the total number of trainable parameters in *model*."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def detect_base_model_name(raw_name: str) -> str:
    """Strip suffixes like '_segmentation_test' to get the canonical model name."""
    if "UNet3Plus" in raw_name:
        return "UNet3Plus"
    if "UNetPlusPlus" in raw_name:
        return "UNetPlusPlus"
    return "UNet"


def build_model_from_config(config):
    """Build a UNet-family model from an experiment config dict."""
    from models.unet import UNet
    from models.unetplusplus import UNetPlusPlus
    from models.unet3plus import UNet3Plus

    _MODEL_REGISTRY = {
        'UNet': UNet,
        'UNetPlusPlus': UNetPlusPlus,
        'UNet3Plus': UNet3Plus,
    }
    model_name = detect_base_model_name(config.get("model_name", "UNet"))
    model_params = config.get("model_params", {})
    dag_spec = config.get("dag_spec", {})
    cls = _MODEL_REGISTRY[model_name]
    return cls(
        n_channels=model_params.get("n_channels", 3),
        n_classes=model_params.get("n_classes", 21),
        bn=config.get("bn", False),
        base_width=dag_spec.get("base_channels", 32),
        depth=dag_spec.get("depth", 5),
    )

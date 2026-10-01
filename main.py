"""
Main entry point for running experiments on DAG-encoded UNet architectures.

Modes:
  - Training (default): DAG structural analysis → theoretical analysis →
    empirical training + validation metrics.
  - Inference (--inference): loads a trained checkpoint, predicts on a
    validation sample, and saves a visualisation image.

Pipeline (training mode):
1.  DAG structural analysis (effective depth/width/edges).
2.  For each architecture: theoretical analysis on the DAG-consistent
    UNetDAG surrogate, plus empirical analysis on the standalone
    segmentation model.
"""

import torch
import os
import json
import time
import traceback
import argparse

from theoretical_metrics import run_theoretical_analysis
from train import run_empirical_analysis
from datasets import DATASET_REGISTRY, build_dataset, get_dataset_info
from models.unet import UNet
from models.unetplusplus import UNetPlusPlus
from models.unet3plus import UNet3Plus
from unet_dag import build_unet_dag, get_all_dag_specs
from dag_utils import dag_summary, dag_to_string
from utils import save_results, setup_device, set_seed, enable_cudnn_benchmark, build_model_from_config, count_parameters

# ──────────────────────────────────────────────────────────────────────
# Argument parser  (follows no_free_lunch_architectures-main/main.py)
# ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='DAG-encoded UNet Architectures')

##################################### Dataset #################################################
parser.add_argument('--dataset', type=str, required=True, choices=list(DATASET_REGISTRY.keys()), help='dataset')
parser.add_argument('--dataset_root', type=str, default=None,
                    help='override dataset root directory (default: from env var or dataset/...)')

##################################### Architecture ############################################
parser.add_argument('--arch', type=str, required=True, choices=['UNet', 'UNetPlusPlus', 'UNet3Plus', 'all'], help='model architecture')
parser.add_argument('--base_channels_unet', type=int, default=None, help='UNet base channel width (if omitted, resolved from --param_budget + --depth)')
parser.add_argument('--base_channels_unetpp', type=int, default=None, help='UNet++ base channel width (if omitted, resolved from --param_budget + --depth)')
parser.add_argument('--base_channels_unet3p', type=int, default=None, help='UNet3+ base channel width (if omitted, resolved from --param_budget + --depth)')
parser.add_argument('--param_budget', type=str, default='5M', choices=['2M', '5M', '10M', '15M'],
                    help='parameter budget; base_channels auto-filled from calibrated table unless --base_channels_* is passed')
parser.add_argument('--depth', type=int, default=5, help='number of encoder resolution levels (e.g. 3, 4, 5)')
parser.add_argument('--img_size', type=int, default=256, help='input image size (square)')
parser.add_argument('--bn', action="store_true", help="use BN")

##################################### General setting ############################################
parser.add_argument('--seeds', type=str, default='42',
                    help='comma-separated random seeds, e.g. "42" or "42,123,2024". '
                         'Each seed produces a subdir seed<N>/ under the experiment folder.')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers in dataloader')
parser.add_argument('--inference', action="store_true", help="testing")
parser.add_argument('--model_path', type=str, default=None, help='path to trained_model.pth or experiment dir (inference mode)')
#parser.add_argument('--save_dir', help='the directory used to save the trained models', default='./experiments', type=str)
parser.add_argument('--output_dir', type=str, default=None,
                    help='directly specify the output folder; if omitted, '
                         'auto-creates experiments/experiment_MMDD-HHMMSS')

##################################### Training setting #################################################
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0., type=float, help='momentum')
parser.add_argument('--weight_decay', default=0, type=float, help='weight decay')
parser.add_argument('--epochs', default=350, type=int, help='number of total epochs to run')
parser.add_argument('--warmup', default=0, type=int, help='warm up epochs')
parser.add_argument('--decreasing_lr', default=None, help='decreasing strategy')
parser.add_argument('--patience', default=30, type=int, help='early stopping patience (0=disabled)')

##################################### Theoretical analysis #################################################
parser.add_argument('--ntk_batch_size', type=int, default=128,
                    help='number of input samples for NTK/NNGP kernel matrices '
                         '(= number of eigenvalues).  Reference: 128')
parser.add_argument('--n_interp', type=int, default=128,
                    help='number of interpolation points on the SVD circle '
                         'for manifold complexity.  Reference: 128')
parser.add_argument('--complexity_fwd_batch_size', type=int, default=4,
                    help='forward-pass sub-batch for curve complexity '
                         '(keep small for large UNets to save GPU memory)')
parser.add_argument('--repeat', type=int, default=3,
                    help='number of random re-initialisations for theoretical '
                         'metrics (reduces variance).  Reference: 3')
parser.add_argument('--theoretical_only', action='store_true',
                help='run theoretical analysis only (skip empirical training)')


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
_MODEL_REGISTRY = {
    'UNet': UNet,
    'UNetPlusPlus': UNetPlusPlus,
    'UNet3Plus': UNet3Plus,
}

_BASE_CHANNELS_ARG = {
    'UNet': 'base_channels_unet',
    'UNetPlusPlus': 'base_channels_unetpp',
    'UNet3Plus': 'base_channels_unet3p',
}

# Pre-calibrated base_channels keeping parameter count close to the target budget.
# Key: budget_key -> {(depth, arch_name): base_channels}. Values verified by counting
# parameters of the actual standalone models (utils.count_parameters).
# Override via --base_channels_unet / --base_channels_unetpp / --base_channels_unet3p.
_PARAM_BUDGET_TABLE: dict[str, dict[tuple[int, str], int]] = {
    '2M': {
        (3, 'UNet'): 63,  (3, 'UNetPlusPlus'): 60,  (3, 'UNet3Plus'): 59,
        (4, 'UNet'): 31,  (4, 'UNetPlusPlus'): 29,  (4, 'UNet3Plus'): 32,
        (5, 'UNet'): 15,  (5, 'UNetPlusPlus'): 14,  (5, 'UNet3Plus'): 17,
    },
    '5M': {
        (3, 'UNet'): 100, (3, 'UNetPlusPlus'): 95,  (3, 'UNet3Plus'): 93,
        (4, 'UNet'): 49,  (4, 'UNetPlusPlus'): 45,  (4, 'UNet3Plus'): 50,
        (5, 'UNet'): 24,  (5, 'UNetPlusPlus'): 22,  (5, 'UNet3Plus'): 28,
    },
    '10M': {
        (3, 'UNet'): 141, (3, 'UNetPlusPlus'): 134, (3, 'UNet3Plus'): 132,
        (4, 'UNet'): 69,  (4, 'UNetPlusPlus'): 64,  (4, 'UNet3Plus'): 71,
        (5, 'UNet'): 34,  (5, 'UNetPlusPlus'): 32,  (5, 'UNet3Plus'): 39,
    },
    '15M': {
        (3, 'UNet'): 172, (3, 'UNetPlusPlus'): 164, (3, 'UNet3Plus'): 161,
        (4, 'UNet'): 85,  (4, 'UNetPlusPlus'): 79,  (4, 'UNet3Plus'): 87,
        (5, 'UNet'): 42,  (5, 'UNetPlusPlus'): 39,  (5, 'UNet3Plus'): 48,
    },
}


def _resolve_base_channels(args, arch_name: str) -> int:
    """Return base_channels for *arch_name*: CLI override if provided, else table lookup."""
    cli_val = getattr(args, _BASE_CHANNELS_ARG[arch_name])
    if cli_val is not None:
        return cli_val
    try:
        bc = _PARAM_BUDGET_TABLE[args.param_budget][(args.depth, arch_name)]
    except KeyError:
        raise ValueError(
            f"No calibrated base_channels for (budget={args.param_budget}, "
            f"depth={args.depth}, arch={arch_name}). Pass --base_channels_* explicitly."
        )
    print(f"  [budget] {arch_name}: base_channels={bc} @ depth={args.depth}, budget={args.param_budget}")
    return bc

def _build_empirical_model(config):
    """Instantiate the standalone UNet-family model used for training."""
    model_name = config.get("model_name", "UNet")
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


def _build_theoretical_model(config):
    """Instantiate the DAG-consistent model used for theoretical metrics."""
    model_name = config.get("model_name", "UNet")
    model_params = config.get("model_params", {})
    dag_spec = config.get("dag_spec", {})
    return build_unet_dag(
        model_name,
        n_channels=model_params.get("n_channels", 3),
        n_classes=model_params.get("n_classes", 21),
        base_channels=dag_spec.get("base_channels", 32),
        bn=config.get("bn", False),
        depth=dag_spec.get("depth", 5),
    )


def run_single_experiment(config, parent_dir, device, run_name=None, theoretical_only=False):
    """
    Runs a full theoretical and empirical analysis for a single model configuration.
    """
    # 1. Setup: Seed, Output Directory
    if run_name is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        run_name = f"{config['model_name']}_{timestamp}"
    output_dir = os.path.join(parent_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n----- Starting Experiment: {run_name} -----")
    print(f"Results will be saved in: {output_dir}")

    seed = set_seed(config.get('seed'))
    config['seed'] = seed
    config['theoretical_only'] = theoretical_only

    # 2. Instantiate Model
    model_for_theoretical = _build_theoretical_model(config)
    model_for_empirical = _build_empirical_model(config)
    n_params_theoretical = count_parameters(model_for_theoretical)
    n_params_empirical = count_parameters(model_for_empirical)
    print(f"  Theoretical model: UNetDAG  params={n_params_theoretical:,} ({n_params_theoretical/1e6:.3f}M)")
    print(f"  Empirical model:   standalone  params={n_params_empirical:,} ({n_params_empirical/1e6:.3f}M)")

    # 3. Run Theoretical Analysis
    try:
        theoretical_dataset_params = {
            'dataset_name': config['dataset_config']['dataset_name'],
            'path': config['dataset_config']['path'],
            'img_size': config['dataset_config']['img_size'],
            'num_workers': config['dataset_config'].get('num_workers', 0),
        }
        theoretical_results = run_theoretical_analysis(
            model_for_theoretical, theoretical_dataset_params, device,
            ntk_batch_size=config.get('ntk_batch_size', 128),
            n_interp=config.get('n_interp', 128),
            complexity_fwd_batch_size=config.get('complexity_fwd_batch_size', 4),
            repeat=config.get('repeat', 3),
        )
        theoretical_results.setdefault("meta", {}).update({
            "model_impl": config.get("theoretical_model_impl", "UNetDAG"),
            "param_count": n_params_theoretical,
            "dag_string": config.get("theoretical_dag", {}).get("dag_string"),
            "node_scales": config.get("theoretical_dag", {}).get("node_scales"),
        })
        save_results(theoretical_results, output_dir, "theoretical_metrics.json")
    except Exception as e:
        print(f"Error during theoretical analysis: {e}")
        with open(os.path.join(output_dir, "theoretical_error.txt"), 'w') as f:
            f.write(f"{str(e)}\n\n{traceback.format_exc()}")

    # 4. Run Empirical Analysis (skip in theoretical-only mode)
    if not theoretical_only:
        # Re-seed so empirical training is independent of theoretical analysis runtime.
        set_seed(config['seed'])
        try:
            model_for_empirical.to(device)
            empirical_results = run_empirical_analysis(model_for_empirical, config['train_params'], config['dataset_config'], device)

            save_results(empirical_results, output_dir, "empirical_metrics.json")

            model_path = os.path.join(output_dir, "trained_model.pth")
            torch.save(model_for_empirical.state_dict(), model_path)
            print(f"Trained model saved to {model_path}")
        except Exception as e:
            print(f"Error during empirical analysis: {e}")
            with open(os.path.join(output_dir, "empirical_error.txt"), 'w') as f:
                f.write(f"{str(e)}\n\n{traceback.format_exc()}")
    else:
        print("Skipping empirical analysis (theoretical-only mode).")

    # 5. Save config for this run
    save_results(config, output_dir, "config.json")
    print(f"----- Experiment {run_name} Finished -----")


# ──────────────────────────────────────────────────────────────────────
# Inference visualisation
# ──────────────────────────────────────────────────────────────────────
def run_inference(args, device):
    """Load a trained model and visualise one prediction."""
    import matplotlib.pyplot as plt
    if args.model_path is None:
        raise ValueError("--model_path is required for inference mode.")

    if os.path.isdir(args.model_path):
        exp_dir = args.model_path
        model_file = os.path.join(exp_dir, "trained_model.pth")
    else:
        model_file = args.model_path
        exp_dir = os.path.dirname(model_file)

    config_path = os.path.join(exp_dir, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"config.json not found in {exp_dir}")
    with open(config_path, "r") as f:
        config = json.load(f)

    model_name = config.get("model_name", "UNet")
    print(f"Model: {model_name}")

    print(f"Loading model from: {model_file}")
    model = build_model_from_config(config).to(device)
    model.load_state_dict(torch.load(model_file, map_location=device, weights_only=True))
    model.eval()
    model_params = config.get("model_params", {})

    dataset_config = config.get("dataset_config", {})
    dataset_name = dataset_config.get("dataset_name", "VOC2012")
    dataset_root = args.dataset_root or dataset_config.get("path", os.path.join("dataset", "VOC2012"))
    img_size = tuple(dataset_config.get("img_size", [256, 256]))

    val_dataset = build_dataset(dataset_name, root=dataset_root, split="val", img_size=img_size)
    image, mask_true = val_dataset[0]
    image_batch = image.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_batch)

    num_classes = model_params.get("n_classes", 21)
    if num_classes == 1:
        mask_pred = (torch.sigmoid(logits) > 0.5).squeeze(0).squeeze(0).cpu().numpy()
    else:
        mask_pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

    image_np = image.permute(1, 2, 0).numpy()
    mask_true_np = mask_true.numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image_np)
    axes[0].set_title("Input Image")
    axes[0].axis("off")
    axes[1].imshow(mask_true_np, cmap="tab20", interpolation="nearest")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")
    axes[2].imshow(mask_pred, cmap="tab20", interpolation="nearest")
    axes[2].set_title("Prediction")
    axes[2].axis("off")
    output_path = os.path.join(os.path.dirname(args.model_path), "segmentation_result.png")
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Result saved to {output_path}")


# ──────────────────────────────────────────────────────────────────────
# Multi-seed aggregation
# ──────────────────────────────────────────────────────────────────────
def _flatten_scalars(d, prefix=''):
    """Recurse into nested dicts, yield (dotted_key, value) for scalar numeric leaves."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_scalars(v, key))
        elif isinstance(v, bool):
            continue
        elif isinstance(v, (int, float)):
            out[key] = v
    return out


def aggregate_results(master_dir, seeds, arch_list):
    """Collect scalar metrics from seed<N>/<arch>/*.json, save mean/std per arch."""
    import statistics

    agg = {}
    metric_files = ['empirical_metrics.json', 'theoretical_metrics.json']

    for arch in arch_list:
        per_seed_scalars = []
        for seed in seeds:
            combined = {}
            for mf in metric_files:
                path = os.path.join(master_dir, f'seed{seed}', arch, mf)
                if os.path.isfile(path):
                    with open(path, 'r') as f:
                        combined.update(_flatten_scalars(json.load(f)))
            per_seed_scalars.append(combined)

        all_keys = set()
        for ps in per_seed_scalars:
            all_keys.update(ps.keys())

        arch_agg = {}
        for key in sorted(all_keys):
            vals = [ps[key] for ps in per_seed_scalars if key in ps]
            if not vals:
                continue
            arch_agg[key] = {
                'mean': statistics.mean(vals),
                'std': statistics.stdev(vals) if len(vals) > 1 else 0.0,
                'min': min(vals),
                'max': max(vals),
                'values': vals,
                'n_seeds': len(vals),
            }
        agg[arch] = arch_agg

    save_results(agg, master_dir, 'aggregated_metrics.json')
    print(f"Aggregated metrics saved to {os.path.join(master_dir, 'aggregated_metrics.json')}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    args = parser.parse_args()

    device = setup_device(args.gpu)

    # --- Inference mode ---
    if args.inference:
        run_inference(args, device)
        return

    # --- Training mode ---
    if args.output_dir:
        master_dir = args.output_dir
    else:
        timestamp = time.strftime("%m%d-%H%M%S")
        master_dir = os.path.join("experiments", f"experiment_{timestamp}")
    os.makedirs(master_dir, exist_ok=True)

    try:
        seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
    except ValueError:
        raise ValueError(f"--seeds must be comma-separated ints, got: {args.seeds!r}")
    if not seeds:
        raise ValueError("--seeds must contain at least one seed")
    enable_cudnn_benchmark()
    print(f"Device: {device}  Seeds: {seeds}  Results -> {master_dir}")

    num_classes, _ = get_dataset_info(args.dataset)
    dataset_root = args.dataset_root or DATASET_REGISTRY[args.dataset]["default_root"]

    decreasing_lr = list(map(int, args.decreasing_lr.split(','))) if args.decreasing_lr else None

    # Collect architecture specs (parameterised by --depth)
    all_specs = get_all_dag_specs(depth=args.depth)
    if args.arch != 'all':
        all_specs = {args.arch: all_specs[args.arch]}

    # ---- Phase 1: DAG structural analysis (seed-independent; save at master_dir top level) ----
    print("\n===== DAG Structural Analysis =====")
    for arch_name, (dag, node_scales) in all_specs.items():
        info = dag_summary(dag, node_scales)
        print(f"  {arch_name}:  nodes={info['num_nodes']}  "
              f"param_edges={info['num_param_edges']}  "
              f"skip_edges={info['num_skip_edges']}  "
              f"eff_depth={info['effective_depth']:.2f}  "
              f"param_paths={info['parameterized_end_to_end_paths']}  "
              f"paths={info['end_to_end_paths']}  "
              f"eff_width={info['effective_width']:.2f}")
        save_results(info, master_dir, f"dag_structural_analysis_{arch_name}.json")

    # ---- Phase 2: Per-seed, per-architecture experiment ----
    for seed in seeds:
        seed_dir = os.path.join(master_dir, f"seed{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        print(f"\n========== SEED {seed} ==========")
        for arch_name, (dag, node_scales) in all_specs.items():
            bc = _resolve_base_channels(args, arch_name)
            config = {
                "model_name": arch_name,
                "bn": args.bn,
                "dag_spec": {"base_channels": bc, "depth": args.depth},
                "theoretical_model_impl": "UNetDAG",
                "theoretical_dag": {
                    "dag_string": dag_to_string(dag),
                    "node_scales": node_scales,
                },
                "model_params": {"n_channels": 3, "n_classes": num_classes},
                "dataset_config": {
                    "dataset_name": args.dataset,
                    "path": dataset_root,
                    "img_size": (args.img_size, args.img_size),
                    "num_workers": args.num_workers,
                },
                "train_params": {
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "momentum": args.momentum,
                    "weight_decay": args.weight_decay,
                    "warmup": args.warmup,
                    "decreasing_lr": decreasing_lr,
                    "patience": args.patience,
                },
                "seed": seed,
                # Theoretical analysis params (reference: no_free_lunch_architectures-main)
                "ntk_batch_size": args.ntk_batch_size,
                "n_interp": args.n_interp,
                "complexity_fwd_batch_size": args.complexity_fwd_batch_size,
                "repeat": args.repeat,
            }
            run_single_experiment(config, seed_dir, device, run_name=arch_name, theoretical_only=args.theoretical_only)

    # ---- Phase 3: Aggregate across seeds (only when multi-seed) ----
    if len(seeds) > 1:
        aggregate_results(master_dir, seeds, list(all_specs.keys()))

    print(f"\nAll done. Results saved in: {master_dir}")


if __name__ == '__main__':
    main()

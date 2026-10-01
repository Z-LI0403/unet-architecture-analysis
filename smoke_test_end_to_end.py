"""Quick end-to-end smoke test: build, forward pass, and train 1 step."""

import torch
import torch.nn as nn

from theoretical_metrics import run_theoretical_analysis
from datasets import get_dataloaders
from unet_dag import build_unet_dag, get_all_dag_specs
from dag_utils import dag_summary
from utils import setup_device


def main():
    device = setup_device()
    print("device", device)

    train_loader, _ = get_dataloaders(
        "VOC2012", root="dataset/VOC2012", img_size=(64, 64), batch_size=2, num_workers=0
    )
    images, masks = next(iter(train_loader))
    print(
        "train_batch",
        tuple(images.shape),
        tuple(masks.shape),
        masks.dtype,
        int(masks.min()),
        int(masks.max()),
    )

    # ---- DAG structural analysis ----
    print("--- DAG structural analysis ---")
    for arch_name, (dag, node_scales) in get_all_dag_specs().items():
        info = dag_summary(dag, node_scales)
        print(
            arch_name,
            f"nodes={info['num_nodes']}",
            f"param_edges={info['num_param_edges']}",
            f"skip_edges={info['num_skip_edges']}",
            f"eff_depth={info['effective_depth']:.2f}",
            f"param_paths={info['parameterized_end_to_end_paths']}",
            f"paths={info['end_to_end_paths']}",
            f"eff_width={info['effective_width']:.2f}",
        )

    # ---- Build DAG models ----
    models = [
        ("UNet", build_unet_dag("UNet", n_channels=3, n_classes=21, base_channels=16)),
        ("UNetPlusPlus", build_unet_dag("UNetPlusPlus", n_channels=3, n_classes=21, base_channels=16)),
        ("UNet3Plus", build_unet_dag("UNet3Plus", n_channels=3, n_classes=21, base_channels=16)),
    ]

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    images = images.to(device)
    masks = masks.to(device)

    print("--- forward/loss smoke ---")
    for name, model in models:
        model = model.to(device).eval()
        with torch.no_grad():
            outputs = model(images)
            loss = criterion(outputs, masks)
        print(name, "out", tuple(outputs.shape), "loss", float(loss))

    print("--- theoretical smoke ---")
    for name, model in models:
        results = run_theoretical_analysis(
            model,
            {
                "dataset_name": "VOC2012",
                "path": "dataset/VOC2012",
                "img_size": (32, 32),
                "num_workers": 0,
            },
            device,
            ntk_batch_size=8, n_interp=16,
            complexity_fwd_batch_size=4, repeat=2,
        )
        n_nngp = len(results["nngp_eigenvalues"])
        n_ntk = len(results["ntk_eigenvalues"])
        n_rep = len(results["nngp_eigenvalues_per_repeat"])
        print(
            name,
            "ok",
            f"ntk={n_ntk}",
            f"nngp={n_nngp}",
            f"repeats={n_rep}",
            f"len={float(results['complexity_length']):.4f}",
            f"curv={float(results['complexity_curvature']):.4f}",
        )
        # Verify eigenvalue count matches ntk_batch_size
        assert n_nngp == 8, f"Expected 8 NNGP eigenvalues, got {n_nngp}"
        assert n_ntk == 8, f"Expected 8 NTK eigenvalues, got {n_ntk}"
        assert n_rep == 2, f"Expected 2 repeats, got {n_rep}"

    print("SMOKE_TEST_PASS")


if __name__ == "__main__":
    main()

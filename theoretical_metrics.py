"""Theoretical analysis: NTK/NNGP eigenvalues, manifold complexity.

Methodology follows 'No Free Lunch in Neural Architectures' (Chen et al.):

  - NTK/NNGP: Compute kernel matrices from network Jacobians / features on a
    real training batch. Eigenvalue spectra characterise the inductive bias
    of the architecture at initialisation.
    Reference: ntk.py, traversal_nngp_ntk.py  (no_free_lunch_architectures-main)

  - Curve length: Embed a 1-D circle (via SVD) in input space, push through
    the network, and integrate ||df/dtheta||.  Measures output sensitivity /
    manifold complexity.

  - Extrinsic curvature: 2nd-order derivative along the same parametric curve
    using the differential geometry formula
        kappa = sqrt(|v|^2 |a|^2 - (v . a)^2) / |v|^{3/2}
    Measures how much the output manifold bends.
    Reference: length.py, traversal_complexity.py  (no_free_lunch_architectures-main)

Key parameters (reference defaults in parentheses):
  - ntk_batch_size (128): number of input samples -> number of eigenvalues.
    Using only 4 (the former default) gives far too few for spectral analysis.
  - n_interp (128): number of points on the SVD parametric circle for
    manifold complexity.
  - repeat (3): number of random re-initialisations to average over,
    reducing variance from random weight init.
"""

import torch
import numpy as np
from torch import autograd

from datasets import get_dataloaders


# ── Helpers ────────────────────────────────────────────────────────────

def _maybe_gap(tensor):
    """Global-Average-Pool spatial dims for segmentation outputs (B, C, H, W).

    Reference models produce (B, C) classification logits directly.  UNet
    segmentation outputs are (B, C, H, W); GAP reduces the spatial dimensions
    to (B, C) so kernel / Jacobian computations remain tractable and
    comparable to the classification setting.
    """
    if tensor.dim() == 4:
        return tensor.mean(dim=(2, 3))
    return tensor.view(tensor.size(0), -1)


def _load_ntk_batch(dataset_params, ntk_batch_size):
    """Load one normalised train batch for NTK/NNGP."""
    img_h, img_w = dataset_params['img_size']
    dataset_name = dataset_params.get('dataset_name')
    dataset_root = dataset_params.get('path')
    num_workers = dataset_params.get('num_workers', 0)

    if dataset_name is None or dataset_root is None:
        raise ValueError(
            "run_theoretical_analysis now requires dataset_params to include "
            "'dataset_name' and 'path' so NTK/NNGP can use a real train batch."
        )

    train_loader, _ = get_dataloaders(
        dataset_name,
        root=dataset_root,
        img_size=(img_h, img_w),
        batch_size=ntk_batch_size,
        num_workers=num_workers,
    )
    images, _ = next(iter(train_loader))
    images /= torch.norm(images, dim=-1, keepdim=True)
    return images


# ── Parametric curve construction ──────────────────────────────────────

def get_curve_input(size_curve, device):
    """Create a 1-D parametric curve (circle) embedded in input space.

    Reference: length.py / get_curve_input  (no_free_lunch_architectures-main)

    The curve is parameterised by theta in [0, 2*pi] with *n_interp* evenly
    spaced points.  A random orthonormal 2-D basis in input space is obtained
    via the thin SVD of a random matrix, and the curve is the circle in this
    plane:

        x(theta) = U @ [cos theta, sin theta]^T        (U from thin SVD)

    Because theta has requires_grad=True, autograd can differentiate network
    outputs w.r.t. theta, enabling analytic curve-length and curvature
    computation.  This replaces the former approach of using two independent
    random tensors, which did not form a proper 1-D manifold.
    """
    n_interp = size_curve[0]
    CHW = size_curve[1:]

    theta = torch.linspace(0, 2 * np.pi, n_interp, device=device)
    theta.requires_grad_(True)

    # Thin SVD gives an orthonormal 2-D plane in R^(prod(CHW))
    rand_mat = torch.randn(int(np.prod(CHW)), 2, device=device)
    U = torch.linalg.svd(rand_mat, full_matrices=False)[0]  # (prod(CHW), 2)

    curve_input = torch.matmul(
        U, torch.stack([torch.cos(theta), torch.sin(theta)])
    ).T.reshape((n_interp, *CHW))
    curve_input.requires_grad_(True)
    return theta, curve_input


# ── Curve length (manifold complexity) ─────────────────────────────────

def curve_complexity_differentiable(network, curve_inputs, batch_size=4,
                                     train_mode=True, need_graph=True,
                                     reduction='mean'):
    """Curve length in output space along the input-space parametric circle.

    Reference: length.py / curve_complexity_differentiable

    For each output coordinate c, compute df_c/dtheta via autograd.  The
    arc-length element is ||df/dtheta||, and the total length is the sum
    (or mean) over theta.

    Adaptation for UNet: the raw segmentation output (B, C, H, W) is reduced
    via Global-Average-Pooling to (B, C) before the Jacobian loop, so we
    iterate over C = n_classes (e.g. 21) instead of C*H*W -- keeping
    computation tractable while preserving the channel-wise manifold
    structure.
    """
    theta, curve_input = curve_inputs
    device = theta.device
    network = network.to(device)
    if train_mode:
        network.train()
    else:
        network.eval()
    network.zero_grad()

    LE = torch.tensor(0.0, device=device)
    _idx = 0
    while _idx < len(curve_input):
        output = network(curve_input[_idx:_idx + batch_size])
        # Reduce segmentation output (B,C,H,W) -> (B,C) via GAP
        output = _maybe_gap(output)
        n, c = output.size()

        jacobs = []
        for coord in range(c):
            # df_coord / dtheta -- gradient of summed output coord w.r.t.
            # the curve parameter theta
            _grad = autograd.grad(
                outputs=output[:, coord].sum(),
                inputs=theta,
                only_inputs=True,
                retain_graph=need_graph,
                create_graph=need_graph,
            )
            jacobs.append(_grad[0].detach())

        jacobs = torch.stack(jacobs, 0).permute(1, 0)  # (n_interp, c)
        # ||df/dtheta|| for current batch's theta points
        gE = torch.einsum('nd,nd->n',
                          jacobs[_idx:_idx + batch_size],
                          jacobs[_idx:_idx + batch_size]).sqrt()
        LE += gE.sum()
        _idx += batch_size
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if reduction == 'mean':
        return LE / len(theta)
    return LE


# ── Extrinsic curvature ───────────────────────────────────────────────

def get_extrinsic_curvature(network, curve_inputs, batch_size=4,
                             train_mode=True):
    """Extrinsic curvature kappa of the output manifold along the parametric curve.

    Reference: length.py / get_extrinsic_curvature

    Uses the differential-geometry formula:

        kappa = sqrt(|v|^2 |a|^2 - (v . a)^2) / |v|^{3/2}

    where v = df/dtheta (velocity) and a = d^2f/dtheta^2 (acceleration).
    Requires create_graph=True so that second-order derivatives through
    theta are available.

    Adaptation for UNet: GAP is applied to reduce (B,C,H,W) -> (B,C), same
    as for curve length.
    """
    theta, curve_input = curve_inputs
    device = theta.device
    network = network.to(device)
    if train_mode:
        network.train()
    else:
        network.eval()
    network.zero_grad()

    kappa = 0
    _idx = 0
    while _idx < len(curve_input):
        output = network(curve_input[_idx:_idx + batch_size])
        output = _maybe_gap(output)
        n, c = output.size()

        v_s = []  # velocity  (1st derivative)
        a_s = []  # acceleration (2nd derivative)
        for coord in range(c):
            v = autograd.grad(output[:, coord].sum(), theta,
                              create_graph=True, retain_graph=True)[0]
            a = autograd.grad(v[_idx:_idx + batch_size].sum(), theta,
                              create_graph=True, retain_graph=True)[0]
            v_s.append(v[_idx:_idx + batch_size].detach().clone())
            a_s.append(a[_idx:_idx + batch_size].detach().clone())

        v_s = torch.stack(v_s, 0).permute(1, 0)  # (batch, c)
        a_s = torch.stack(a_s, 0).permute(1, 0)
        vv = torch.einsum('nd,nd->n', v_s, v_s)
        aa = torch.einsum('nd,nd->n', a_s, a_s)
        va = torch.einsum('nd,nd->n', v_s, a_s)
        # kappa = sqrt(|v|^2|a|^2 - (v.a)^2) / |v|^3  (clamp for numerical safety)
        kappa += (vv ** (-3 / 2) * (vv * aa - va ** 2).clamp(min=0).sqrt()).sum().item()
        _idx += batch_size
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Reference (length.py) returns the sum over all theta points (not averaged),
    # consistent with traversal_complexity.py usage.
    return kappa


# ── NTK / NNGP eigenvalues ────────────────────────────────────────────

def get_ntk_nngp_eig(Xs, network, device, train_mode=False, fwd_batch_size=4):
    """Compute eigenvalues of the NNGP and NTK kernel matrices.

    Reference: ntk.py  (no_free_lunch_architectures-main)

    The kernel matrix dimension equals len(Xs), which directly determines the
    number of eigenvalues.  The reference uses batch_size=128 (from the
    training dataloader), producing 128 eigenvalues per kernel.  Using only 4
    samples (the former default) gives just 4 eigenvalues -- far too few for
    reliable spectral analysis.

    Adaptation for UNet / segmentation:
      - Features and logits have spatial dimensions (B, C, H, W).  We apply
        Global-Average-Pooling before building kernels, reducing each sample
        to a (C,)-vector -- analogous to the (num_classes,) logits in the
        classification reference.
      - The NNGP forward pass is sub-batched with *fwd_batch_size* to limit
        GPU memory for large images (256x256 UNet vs 32x32 CIFAR).
      - NTK gradients are still collected one sample at a time (as in the
        reference), so memory scales linearly.
    """
    if train_mode:
        network.train()
    else:
        network.eval()

    n = len(Xs)
    inputs = Xs.to(device)

    # --- Phase 1: collect features for NNGP (no grad needed) ---
    # Reference: nngp = einsum('nc,mc->nm', features[-1], features[-1])
    all_features = []
    for i in range(0, n, fwd_batch_size):
        batch = inputs[i:i + fwd_batch_size]
        with torch.no_grad():
            features, _ = network(batch, return_all=True)
            feat = _maybe_gap(features[-1].detach())
            all_features.append(feat)

    flat_features = torch.cat(all_features, dim=0)
    nngp = torch.einsum('nc,mc->nm', flat_features, flat_features)
    eigenvalues_nngp = torch.linalg.eigh(nngp)[0]

    # --- Phase 2: per-sample gradients for NTK ---
    # Reference: backward through logits one sample at a time, collect
    # parameter gradients, build NTK gram matrix.
    grads = []
    for _idx in range(n):
        network.zero_grad()
        _, logits = network(inputs[_idx:_idx + 1], return_all=True)
        if isinstance(logits, tuple):
            logits = logits[1]
        # GAP for segmentation logits (B,C,H,W) -> (B,C)
        logits = _maybe_gap(logits)
        logits.backward(torch.ones_like(logits))
        grad = []
        for name, W in network.named_parameters():
            if 'weight' in name or 'bias' in name:
                if W.grad is not None:
                    grad.append(W.grad.view(-1).detach())
        grads.append(torch.cat(grad, -1))
        network.zero_grad()
        if torch.cuda.is_available() and device.type == 'cuda':
            torch.cuda.empty_cache()

    grads = torch.stack(grads, 0)
    ntk = torch.einsum('nc,mc->nm', grads, grads)
    eigenvalues_ntk = torch.linalg.eigh(ntk)[0]

    return eigenvalues_nngp.detach().cpu().numpy(), eigenvalues_ntk.detach().cpu().numpy()


# ── Re-initialisation helper ──────────────────────────────────────────

def _reinit_model(model):
    """Re-initialise learnable parameters of the model.

    If the model exposes a ``_init()`` method (UNet, UNet++, UNet3+), it is
    used directly.  Otherwise fall back to ``reset_parameters()`` on every
    sub-module that supports it.

    Reference: model._init() in traversal_nngp_ntk.py / traversal_complexity.py
    """
    if hasattr(model, '_init'):
        model._init()
    else:
        for m in model.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()


# ── Main theoretical analysis wrapper ──────────────────────────────────

def run_theoretical_analysis(model, dataset_params, device,
                              ntk_batch_size=128,
                              n_interp=128,
                              complexity_fwd_batch_size=4,
                              repeat=3):
    """Run NTK, NNGP, curve-length and curvature analysis.

    Reference workflow (no_free_lunch_architectures-main):
      1. traversal_nngp_ntk.py -- batch_size=128 for NTK/NNGP, yielding 128
         eigenvalues per kernel.  Repeated 3x with fresh random init.
      2. traversal_complexity.py -- n_interp=128 points along a parametric SVD
         circle.  Length and curvature each repeated 3x.
      The reference uses one normalised batch drawn from the training
      dataloader for NTK/NNGP inputs, which is what this implementation uses.

    Args:
        model:      UNet-family model with ``._init()`` for re-initialisation.
        ntk_batch_size:  number of input samples for kernel matrices
                         (= number of eigenvalues).  Default 128 per reference.
        n_interp:   number of interpolation points on the SVD circle for
                    manifold complexity.  Default 128 per reference.
        complexity_fwd_batch_size:  forward-pass sub-batch for curve
                    complexity (keep small for large UNets to save GPU memory).
        repeat:     number of random re-initialisations to average over.
                    Default 3 per reference.
    """
    print(f"Starting theoretical analysis "
          f"(ntk_batch={ntk_batch_size}, n_interp={n_interp}, repeat={repeat})...")

    img_h, img_w = dataset_params['img_size']
    n_ch = model.n_channels

    # --- Prepare fixed inputs (shared across repeats) ---
    # NTK/NNGP: use one normalised train batch, matching the Chen23
    # traversal_nngp_ntk.py workflow.
    ntk_input = _load_ntk_batch(
        {
            **dataset_params,
            'img_size': (img_h, img_w),
        },
        ntk_batch_size,
    )

    # Curve for complexity: SVD-based parametric circle  (reference: get_curve_input)
    curve_shape = (n_interp, n_ch, img_h, img_w)
    theta, curve_input = get_curve_input(curve_shape, device)

    # --- Collect metrics over multiple random initialisations ---
    # Reference: repeat=3 in traversal_nngp_ntk.py / traversal_complexity.py
    all_nngp = []
    all_ntk = []
    all_le = []
    all_kappa = []

    for r in range(repeat):
        print(f"  repeat {r + 1}/{repeat}")
        _reinit_model(model)
        model.to(device)

        # NTK / NNGP  (eval mode, per reference traversal_nngp_ntk.py)
        model.eval()
        nngp_eig, ntk_eig = get_ntk_nngp_eig(
            ntk_input, model, device,
            fwd_batch_size=complexity_fwd_batch_size)
        all_nngp.append(nngp_eig.tolist())
        all_ntk.append(ntk_eig.tolist())

        # Curve length & curvature  (train mode, per reference traversal_complexity.py)
        model.train()
        le = curve_complexity_differentiable(
            model, curve_inputs=(theta, curve_input),
            batch_size=complexity_fwd_batch_size,
            train_mode=True, need_graph=True, reduction='mean')
        kappa = get_extrinsic_curvature(
            model, curve_inputs=(theta, curve_input),
            batch_size=complexity_fwd_batch_size,
            train_mode=True)
        all_le.append(le.item() if isinstance(le, torch.Tensor) else le)
        all_kappa.append(kappa)

    # --- Average across repeats ---
    avg_nngp = np.mean(all_nngp, axis=0).tolist()
    avg_ntk = np.mean(all_ntk, axis=0).tolist()
    avg_le = float(np.mean(all_le))
    avg_kappa = float(np.mean(all_kappa))

    print("Theoretical analysis complete.")

    results = {
        "nngp_eigenvalues": avg_nngp,
        "ntk_eigenvalues": avg_ntk,
        "complexity_length": avg_le,
        "complexity_curvature": avg_kappa,
        # Per-repeat raw data for variance analysis
        "nngp_eigenvalues_per_repeat": all_nngp,
        "ntk_eigenvalues_per_repeat": all_ntk,
        "complexity_length_per_repeat": all_le,
        "complexity_curvature_per_repeat": all_kappa,
        "meta": {
            "ntk_batch_size": ntk_batch_size,
            "n_interp": n_interp,
            "repeat": repeat,
        },
    }

    return results


if __name__ == '__main__':
    from models.unet import UNet
    from utils import setup_device

    test_device = setup_device()
    print(f"Testing on device: {test_device}")

    test_model = UNet(n_channels=3, n_classes=21)
    test_dataset_params = {
        'dataset_name': 'VOC2012',
        'path': 'dataset/VOC2012',
        'img_size': (32, 32),
        'num_workers': 0,
    }

    # Use small values for quick smoke test
    theoretical_results = run_theoretical_analysis(
        test_model, test_dataset_params, test_device,
        ntk_batch_size=8, n_interp=16, complexity_fwd_batch_size=4, repeat=2)

    print("\n--- Theoretical Analysis Test Results ---")
    print(f"NNGP eigenvalues ({len(theoretical_results['nngp_eigenvalues'])} values, "
          f"first 5): {np.round(theoretical_results['nngp_eigenvalues'][:5], 4)}")
    print(f"NTK eigenvalues ({len(theoretical_results['ntk_eigenvalues'])} values, "
          f"first 5): {np.round(theoretical_results['ntk_eigenvalues'][:5], 4)}")
    print(f"Complexity Length: {theoretical_results['complexity_length']:.4f}")
    print(f"Complexity Curvature: {theoretical_results['complexity_curvature']:.4f}")

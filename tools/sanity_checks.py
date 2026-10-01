"""Dataset-free sanity checks against hand-solvable numerical cases."""
from pathlib import Path
import ast
import hashlib
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn
from dag_utils import dag2affinity, effective_depth_width
from models.unet import UNet
from models.unetplusplus import UNetPlusPlus
from models.unet3plus import UNet3Plus
from theoretical_metrics import (get_ntk_nngp_eig, curve_complexity_differentiable,
                                 get_extrinsic_curvature)
from train import compute_miou_and_pixel_acc
from unet_dag import build_unet_dag


def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    print('Sanity checks: CPU; synthetic tensors')
    print('Python:', sys.version.replace('\n', ' '))
    print('torch:', torch.__version__, 'numpy:', np.__version__)
    manifest = json.loads((ROOT / 'evidence/artifact_checksums.json').read_text())
    for entry in manifest['files']:
        raw = (ROOT / entry['path']).read_bytes()
        assert len(raw) == entry['bytes'], entry['path']
        assert hashlib.sha256(raw).hexdigest() == entry['sha256'], entry['path']
    print('PASS research artifact checksums:', len(manifest['files']))
    py_files = [ROOT / name for name in ['main.py', 'dag_utils.py', 'datasets.py',
                'theoretical_metrics.py', 'train.py', 'unet_dag.py', 'utils.py',
                'visualization.py', 'smoke_test_end_to_end.py']]
    py_files += list((ROOT / 'models').glob('*.py')) + list((ROOT / 'tools').glob('*.py'))
    for path in py_files:
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    json_files = list((ROOT / 'experiments').rglob('*.json'))
    for path in json_files:
        json.loads(path.read_text(encoding='utf-8'))
    print('PASS Python syntax:', len(py_files), 'experiment JSON:', len(json_files))

    # Expectations manually enumerated, without the project's path enumerator.
    cases = [([[2], [0, 2]], (2, 1, .5, 1, 2)),
             ([[2], [1, 2]], (1, 1, 1, 2, 2)),
             ([[0], [0, 2]], (0, 0, 0, 0, 0)),
             ([[1], [0, 1]], (0, 0, 0, 1, 0))]
    for dag, expected in cases:
        actual = effective_depth_width(dag2affinity(dag))
        assert np.allclose(actual, expected), (dag, actual, expected)
    print('PASS DAG: chain, skip branch, disconnected input, skip-only path')

    # Two samples x=e1,e2. Features=x: feature Gram=I_2.
    # f(x)=Wx with W in R^(2x2); grad_W sum_c f_c(x) repeats x twice:
    # scalarized gradient Gram=2 I_2, independent of W.
    class LinearToy(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(2, 2, bias=False, dtype=torch.float64)

        def forward(self, x, return_all=False):
            y = self.layer(x)
            return ([x], y) if return_all else y

    nngp, ntk = get_ntk_nngp_eig(torch.eye(2, dtype=torch.float64),
                                LinearToy(), torch.device('cpu'), fwd_batch_size=1)
    assert np.allclose(nngp, [1, 1])
    assert np.allclose(ntk, [2, 2])
    print('PASS finite feature/gradient Gram spectra: analytic linear case')

    # Identity image of a unit circle has speed=1 and curvature=1 at each
    # theta. Source returns mean speed and a sum over 9 curvature samples.
    theta = torch.linspace(0, 2 * math.pi, 9, dtype=torch.float64, requires_grad=True)
    circle = torch.stack([theta.cos(), theta.sin()], dim=1)
    speed = curve_complexity_differentiable(nn.Identity(), (theta, circle),
                                           batch_size=3, train_mode=False)
    curvature_sum = get_extrinsic_curvature(nn.Identity(), (theta, circle),
                                            batch_size=3, train_mode=False)
    assert math.isclose(float(speed), 1, abs_tol=1e-10)
    assert math.isclose(curvature_sum, 9, abs_tol=1e-10)
    print('PASS smooth unit-circle toy: mean speed=1; sampled curvature sum=9')

    miou, acc, _ = compute_miou_and_pixel_acc(torch.tensor([[3, 1], [2, 4]]))
    assert math.isclose(miou, 15 / 28, abs_tol=1e-6)
    assert math.isclose(acc, .7, abs_tol=1e-6)
    empty_miou, empty_acc, _ = compute_miou_and_pixel_acc(torch.zeros(2, 2))
    assert empty_miou == empty_acc == 0
    print('PASS segmentation metrics: manually calculated confusion matrix and empty case')

    x = torch.randn(1, 3, 32, 32)
    target = torch.randint(0, 3, (1, 32, 32))
    classes = {'UNet': UNet, 'UNetPlusPlus': UNetPlusPlus, 'UNet3Plus': UNet3Plus}
    for depth in [3, 4, 5]:
        for arch, cls in classes.items():
            for kind in ['standalone', 'DAG']:
                model = (cls(n_channels=3, n_classes=3, base_width=2, depth=depth)
                         if kind == 'standalone' else
                         build_unet_dag(arch, n_channels=3, n_classes=3,
                                        base_channels=2, depth=depth))
                output = model(x)
                assert output.shape == (1, 3, 32, 32), (arch, depth, kind, output.shape)
                assert torch.isfinite(output).all(), (arch, depth, kind)
                loss = nn.functional.cross_entropy(output, target)
                loss.backward()
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                assert grads and all(torch.isfinite(g).all() for g in grads)
                print(f'PASS {arch} depth={depth} {kind}: shape, finite output/loss/gradient')
    print('SANITY_CHECKS_PASS')
    print('Limits: full VOC smoke, training, stored-run provenance, GPU behavior and')
    print('general mathematical validity were not verified by these minimal checks.')


if __name__ == '__main__':
    main()

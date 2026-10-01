# Measurement definitions and limitations

This document describes the current finite-network implementation. It does not assert an infinite-width limit, prove a general architecture ranking, or establish a causal relation between numerical statistics and segmentation accuracy.

## Models and tensor domains

Inputs have shape `(B, 3, H, W)`; VOC segmentation logits have shape `(B, 21, H, W)`. Empirical training uses the standalone models in `models/`. Numerical analysis uses `UNetDAG`, constructed from edge types and node spatial scales. Equivalence of these implementations has not been proved. Exact counts must be checked separately when using nominal parameter budgets.

The DAG edge encoding is `0` for no edge, `1` for a resize/skip edge, and `2` for a parameterized convolution edge. Path statistics are definitions implemented by `dag_utils.py`; they do not by themselves imply optimization or generalization guarantees.

## Kernel spectra

For a finite batch of `N` inputs, spatial average pooling reduces features to `F` of shape `(N, C_f)` and logits to `(N, C_out)`. The quantity named NNGP is the finite feature Gram matrix `F F^T`, of shape `(N, N)`.

For each sample, the code differentiates the sum of pooled class logits with respect to the selected weight/bias parameters. Stacking these gradients gives `G` of shape `(N, P)` and a scalarized gradient Gram matrix `G G^T`, also of shape `(N, N)`. This quantity is named NTK in the source. It is not the full matrix-valued output NTK. Kernel evaluation runs in evaluation mode.

The source averages sorted eigenvalues over random reinitializations. Near-zero negative eigenvalues may reflect floating-point error; conditioning and non-finite input issues have not been resolved. For finite real matrices in exact arithmetic, a Gram matrix is positive semidefinite: for every real vector `z` of length `N`, `z^T F F^T z = ||F^T z||² >= 0` (and the same identity holds for `G`). This elementary identity does not validate every stored floating-point result.

## Curve statistics

A random orthonormal two-dimensional plane is embedded in input space, with `x(theta) = U [cos(theta), sin(theta)]^T` and sampled `theta` in `[0, 2*pi]`. Outputs are spatially pooled before differentiation. For the pooled output `f(theta)`, velocity is `v = df/dtheta` and acceleration is `a = d²f/dtheta²`.

- `complexity_length` returns the mean sampled speed `||v||`, rather than quadrature of the arc-length integral.
- `complexity_curvature` sums sampled values of `sqrt((v·v)(a·a) - (v·a)²) / (v·v)^(3/2)`; it is not an integrated or averaged curvature.

The curvature formula requires differentiability through second order and nonzero speed. Piecewise smooth activations and zero-speed points need additional analysis. The implementation has no zero-speed denominator guard. Curve measurements run in training mode; enabling BatchNorm introduces dependence among samples in a forward batch. Values depend on sampling conventions and need these qualifications when compared.

## Data and empirical metrics

VOC2012 uses 21 labels with ignore index 255. The code records train/validation losses, mIoU, pixel accuracy, gradient norms, and the selected best validation epoch. Current `final_val_*` fields refer to the best epoch selected by the training routine, not necessarily the last epoch. Historical source-to-artifact correspondence remains unverified.

The kernel input normalization divides along the last tensor dimension without an epsilon. All-zero rows can cause non-finite inputs. Three included theoretical analyses failed during eigendecomposition; their error types and diagnostic frames are retained, with private absolute paths replaced by filenames.

## Evidence and claim status

The recorded runs contain only seed 42 and are not a completed multi-seed study. All nine supplied configs set `theoretical_only=true` while empirical artifacts coexist; this metadata contradiction is retained. Complete original execution logs, historical source revisions, and environment metadata were not supplied.

Small analytic-case checks verify selected implementation behaviors under their stated toy assumptions. They do not independently reproduce the full experiments or prove the broader research claims. See [the research log](RESEARCH_LOG.md) for explicit claim status and [the evidence index](../evidence/README.md) for recorded results.

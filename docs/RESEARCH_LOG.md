# Research status log

## Recorded evidence

The repository records UNet-family segmentation experiments at depths 3, 4 and 5, with seed 42. Numerical arrays and histories are retained without changing their values. Three theoretical analyses failed; failure messages remain part of the evidence. The implementation definitions and assumptions are documented in [methodology.md](methodology.md).

| Claim or object | Status | Basis and limitations |
| --- | --- | --- |
| Graph metrics and finite-network kernel/curve measurements | DEFINITION | Defined by the implementation; mathematical scope is described in methodology.md |
| Selected toy-case implementation behavior | NUMERICALLY_VERIFIED | Dataset-free checks against manually solvable graphs, linear kernels, a smooth circle and confusion matrices |
| Independent reproduction of stored segmentation performance | UNRESOLVED | No full training rerun is recorded |
| Complete theoretical results for all nine runs | CONTRADICTED | Three eigendecomposition failures and missing corresponding theoretical result files |
| Equivalence of DAG surrogate and standalone training architectures | UNRESOLVED | Distinct implementations; no equivalence proof supplied |
| Causal mechanism relating numerical statistics to segmentation performance | UNRESOLVED | Recorded comparisons and plots are observational |
| Infinite-width conclusions or general architecture rankings | UNRESOLVED | No corresponding proof or multi-seed evidence supplied |

## Unresolved issues

1. Configuration provenance: all nine configs set `theoretical_only=true`, despite coexisting empirical results. Original logs or independently traced reruns are needed to establish their relationship.
2. Model comparison: nominal budget tables do not prove exact parameter matching or correspondence between surrogate and empirical architectures.
3. Metric interpretation: the NNGP quantity is a finite pooled-feature Gram matrix; the NTK quantity uses summed pooled class logits. Curve statistics use mean speed and summed curvature samples.
4. Numerical robustness: input normalization has no epsilon; curvature has no zero-speed denominator guard. Eigenvalue conditioning, differentiability assumptions and BatchNorm sample coupling require review.
5. Historical reproducibility: full original environment metadata and a source revision linked to each run are unavailable. Current artifact checksums establish file integrity, not historical provenance.

## 2026-10-02 — evidence review

The public evidence was reviewed without changing model implementations or scientific result values. Error-stack paths were replaced with filenames while preserving exception messages, source line references and call frames. Artifact checksums were regenerated after sanitization. Internal publication documents were removed; research limitations and negative results remain explicit.

No scientific claim was upgraded through this documentation review. The small numerical sanity checks cover finite CPU toy cases; full VOC training and GPU behavior remain unverified. No new scientific assumptions or experiments were introduced. The authoritative research plan is not present in the supplied files, so plan-level consistency remains unresolved.

Next dependency-valid research action: independent review of the intended plan, metric definitions and recorded-run provenance before broader interpretation or expanded experiments.

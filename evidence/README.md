# Recorded research evidence

The table below is transcribed from the included empirical result files. Values are recorded validation mIoU, not independently reproduced results. `PRESENT` reports file availability; it is not a verification of scientific correctness.

| Depth | Architecture | Seed | Theoretical artifact | Recorded val mIoU | Recorded epochs |
| --- | --- | --- | --- | --- | --- |
| 3 | UNet | 42 | FAILED | 0.118300 | 350 |
| 3 | UNet3Plus | 42 | PRESENT | 0.093800 | 230 |
| 3 | UNetPlusPlus | 42 | PRESENT | 0.118000 | 300 |
| 4 | UNet | 42 | FAILED | 0.129485 | 350 |
| 4 | UNet3Plus | 42 | PRESENT | 0.109126 | 225 |
| 4 | UNetPlusPlus | 42 | PRESENT | 0.126590 | 303 |
| 5 | UNet | 42 | PRESENT | 0.134269 | 350 |
| 5 | UNet3Plus | 42 | PRESENT | 0.128996 | 190 |
| 5 | UNetPlusPlus | 42 | FAILED | 0.124165 | 303 |


## Artifact layout

- `../experiments/depth=*/`: graph structural summaries.
- `../experiments/depth=*/seed42/<architecture>/`: configuration, empirical metrics and either theoretical metrics or a failure report.
- `../figures/`: plots and ranking tables derived from recorded results.
- `run_summary.csv`: machine-readable table of result availability and metrics.
- `artifact_audit.json`: inventory and metadata contradictions.
- `artifact_checksums.json`: SHA-256 and byte sizes for the distributed experiment/figure files.

Run `python tools/sanity_checks.py` to verify checksums and the dataset-free toy cases.

## Interpretation and provenance

Only seed 42 is present. Three theoretical stages failed: UNet at depths 3 and 4, and UNet++ at depth 5. Their failure messages remain in the repository; private absolute paths in tracebacks were replaced by source filenames. All numerical result files are retained unchanged.

All nine configs set `theoretical_only=true` while empirical metrics coexist. Original execution logs or traced reruns are required before treating each directory as one completely documented execution. Historical source revisions and full original runtime metadata are unavailable.

Plots and rankings are derivative, observational evidence. They do not prove a general architecture ranking, a causal mechanism, or an infinite-width result. See [measurement definitions](../docs/methodology.md) and [claim status](../docs/RESEARCH_LOG.md).

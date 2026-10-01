# Experiment commands

These commands invoke the current pipeline. They do not establish retrospective provenance for the included runs. Obtain VOC2012 and install dependencies as described in the root README. Write new results to separate directories.

## Compare architectures across depths

```bash
uv run python main.py --dataset VOC2012 --arch all --depth 3 --param_budget 5M --seeds 42,123,2024 --output_dir experiments/reproduction_depth3
uv run python main.py --dataset VOC2012 --arch all --depth 4 --param_budget 5M --seeds 42,123,2024 --output_dir experiments/reproduction_depth4
uv run python main.py --dataset VOC2012 --arch all --depth 5 --param_budget 5M --seeds 42,123,2024 --output_dir experiments/reproduction_depth5
```

Defaults include 350 maximum epochs, batch size 16, learning rate 0.01, momentum 0, weight decay 0, no warmup, BatchNorm disabled, and early-stopping patience 30. See `--help` for overrides. Only seed 42 is present in the included evidence.

## Theory-only analysis

This mode still loads a real VOC training batch for the kernel measurements.

```bash
uv run python main.py --dataset VOC2012 --arch all --depth 5 --param_budget 2M --seeds 42 --theoretical_only --output_dir experiments/theory_reproduction_2M
```

## Explicit widths

Width overrides bypass the parameter-budget lookup; they do not guarantee equal model sizes.

```bash
uv run python main.py --dataset VOC2012 --arch all --depth 5 --seeds 42 --base_channels_unet 24 --base_channels_unetpp 22 --base_channels_unet3p 28 --output_dir experiments/explicit_width_reproduction
```

## Inference

Inference requires a locally trained `trained_model.pth` and its adjacent `config.json`. Checkpoints are not included in the public repository. Supply the dataset and architecture arguments required by the parser:

```bash
uv run python main.py --dataset VOC2012 --arch UNet --inference --model_path experiments/reproduction_depth5/seed42/UNet/trained_model.pth
```

## Plot recorded results

```bash
uv run python visualization.py --experiment depth=5 --output_dir reproduced_figures
uv run python visualization.py --compare_depths --model UNet3Plus --output_dir reproduced_figures
```

Plotting may omit unavailable theoretical measurements. Consult per-run failure reports and the evidence index before interpreting rankings or correlations.

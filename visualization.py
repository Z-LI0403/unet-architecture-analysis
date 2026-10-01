"""
Visualization module for UNet architecture analysis.

Generates publication-quality plots comparing theoretical and empirical metrics
across UNet, UNet++, and UNet3Plus architectures.
"""

import os
import json
import argparse
from collections.abc import Iterable
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.stats as stats

from utils import detect_base_model_name


# ---------------------------------------------------------------------------
# Matplotlib style (follows reference notebook)
# ---------------------------------------------------------------------------
STYLE = {
    "font.size": 14,
    "axes.labelsize": 18,
    "axes.titlesize": 20,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.figsize": (6, 4),
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
}
plt.rcParams.update(STYLE)

# Per-architecture colours / markers
MODEL_STYLE = {
    "UNet":       {"color": "#1f77b4", "marker": "o", "label": "UNet"},
    "UNetPlusPlus": {"color": "#ff7f0e", "marker": "s", "label": "UNet++"},
    "UNet3Plus":  {"color": "#2ca02c", "marker": "D", "label": "UNet 3+"},
}

DEPTH_STYLE_CYCLE = [
    {"color": "#1f77b4", "marker": "o"},
    {"color": "#ff7f0e", "marker": "s"},
    {"color": "#2ca02c", "marker": "D"},
    {"color": "#d62728", "marker": "^"},
    {"color": "#9467bd", "marker": "v"},
    {"color": "#8c564b", "marker": "P"},
    {"color": "#e377c2", "marker": "X"},
    {"color": "#7f7f7f", "marker": "*"},
]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _detect_model_name(config: dict) -> str:
    return detect_base_model_name(config.get("model_name", ""))


def normalize_model_name(raw_name: str) -> str:
    """Normalize model selectors such as 'UNet++' to canonical family names."""
    compact = raw_name.replace(" ", "").lower()
    if "unet3plus" in compact or "unet3+" in compact:
        return "UNet3Plus"
    if "unetplusplus" in compact or "unet++" in compact:
        return "UNetPlusPlus"
    if "unet" in compact:
        return "UNet"
    return raw_name


def _is_complete_record(record: dict) -> bool:
    return record["theoretical"] is not None and record["empirical"] is not None


def _prefer_record(previous: dict, candidate: dict) -> dict:
    """Choose the better record between *previous* and *candidate*."""
    previous_complete = _is_complete_record(previous)
    candidate_complete = _is_complete_record(candidate)
    if candidate_complete and not previous_complete:
        return candidate
    if candidate_complete == previous_complete and candidate["config_mtime"] > previous["config_mtime"]:
        return candidate
    return previous


def _depth_label(depth: int | str) -> str:
    return f"depth={depth}"


def _depth_sort_key(depth_key: int | str) -> tuple[int, object]:
    if isinstance(depth_key, str) and depth_key.startswith("depth="):
        value = depth_key.split("=", 1)[1]
    else:
        value = depth_key

    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, str(value))


def build_depth_style_map(depth_keys: Iterable[str]) -> dict[str, dict]:
    """Build a deterministic plotting style map for cross-depth comparisons."""
    style_map = {}
    for index, depth_key in enumerate(sorted(depth_keys, key=_depth_sort_key)):
        style = DEPTH_STYLE_CYCLE[index % len(DEPTH_STYLE_CYCLE)].copy()
        style["label"] = str(depth_key)
        style_map[str(depth_key)] = style
    return style_map


def _get_style(name: str, style_map: dict[str, dict] | None = None) -> dict:
    if style_map and name in style_map:
        return style_map[name]
    return MODEL_STYLE.get(name, {"color": "gray", "marker": "x", "label": name})


def _is_within(parent_dir: str, child_dir: str) -> bool:
    """Return True when *child_dir* is inside *parent_dir*."""
    try:
        return os.path.commonpath([os.path.abspath(parent_dir), os.path.abspath(child_dir)]) == os.path.abspath(parent_dir)
    except ValueError:
        return False


def resolve_experiment_target(experiments_dir: str, experiment: str | None) -> str:
    """Resolve an experiment selector to an existing directory."""
    if experiment is None:
        return os.path.abspath(experiments_dir)

    candidate = experiment
    if not os.path.isabs(candidate):
        candidate = os.path.join(experiments_dir, candidate)
    candidate = os.path.abspath(candidate)

    if not os.path.isdir(candidate):
        raise FileNotFoundError(f"Experiment target not found: {experiment}")
    return candidate


def describe_target(target_dir: str, experiments_dir: str) -> str:
    """Return a display/output label for *target_dir*."""
    target_dir = os.path.abspath(target_dir)
    experiments_dir = os.path.abspath(experiments_dir)

    if _is_within(experiments_dir, target_dir):
        rel = os.path.relpath(target_dir, experiments_dir)
        if rel in (".", ""):
            return "all_experiments"
        return rel.replace("\\", "/")

    return os.path.basename(target_dir.rstrip(os.sep)) or "experiment"


def build_output_dir_for_target(output_dir: str, target_dir: str, experiments_dir: str) -> str:
    """Create and return the figure directory corresponding to *target_dir*."""
    label = describe_target(target_dir, experiments_dir)
    parts = [p for p in label.split("/") if p not in ("", ".")]
    resolved = os.path.join(output_dir, *parts) if parts else output_dir
    os.makedirs(resolved, exist_ok=True)
    return resolved


def _contains_results(search_dir: str) -> bool:
    """Return True when *search_dir* contains at least one visualizable run."""
    for root, _, files in os.walk(search_dir):
        if "config.json" in files and (
            "theoretical_metrics.json" in files or "empirical_metrics.json" in files
        ):
            return True
    return False


def default_experiment_targets(experiments_dir: str) -> list[str]:
    """Return top-level experiment folders that contain run results."""
    if not os.path.isdir(experiments_dir):
        return []

    targets = []
    for name in sorted(os.listdir(experiments_dir)):
        candidate = os.path.join(experiments_dir, name)
        if os.path.isdir(candidate) and _contains_results(candidate):
            targets.append(os.path.abspath(candidate))

    if targets:
        return targets

    if _contains_results(experiments_dir):
        return [os.path.abspath(experiments_dir)]

    return []


def load_experiments(search_dir: str, experiments_root: str | None = None) -> list[dict]:
    """
    Recursively walk *search_dir* and collect every run that has valid results.

    Supports layouts such as:
        experiments/<experiment_dir>/seed42/<ModelName>/config.json
        experiments/<experiment_dir>/<ModelName>/config.json
        <any custom subtree>/<ModelName>/config.json

    Returns a list of dicts, each containing:
        model_name, run_dir, experiment_dir, config, theoretical, empirical
    """
    records = []
    search_dir = os.path.abspath(search_dir)
    experiments_root = os.path.abspath(experiments_root or search_dir)

    if not os.path.isdir(search_dir):
        print(f"[warn] experiments directory not found: {search_dir}")
        return records

    for run_dir, subdirs, files in os.walk(search_dir):
        if "config.json" not in files:
            continue
        if "theoretical_metrics.json" not in files and "empirical_metrics.json" not in files:
            continue

        config_path = os.path.join(run_dir, "config.json")
        theo_path = os.path.join(run_dir, "theoretical_metrics.json")
        emp_path = os.path.join(run_dir, "empirical_metrics.json")

        with open(config_path) as f:
            config = json.load(f)

        theoretical = None
        if os.path.isfile(theo_path):
            with open(theo_path) as f:
                theoretical = json.load(f)

        empirical = None
        if os.path.isfile(emp_path):
            with open(emp_path) as f:
                empirical = json.load(f)

        relative_run_dir = (
            os.path.relpath(run_dir, experiments_root)
            if _is_within(experiments_root, run_dir)
            else os.path.basename(run_dir)
        )
        relative_parts = [p for p in relative_run_dir.split(os.sep) if p not in ("", ".")]
        experiment_dir = relative_parts[0] if relative_parts else os.path.basename(run_dir)

        records.append({
            "model_name": _detect_model_name(config),
            "run_dir": run_dir,
            "experiment_dir": experiment_dir,
            "run_name": os.path.basename(run_dir),
            "relative_run_dir": relative_run_dir.replace("\\", "/"),
            "config": config,
            "config_mtime": os.path.getmtime(config_path),
            "theoretical": theoretical,
            "empirical": empirical,
        })

        # A run directory should not contain nested run directories.
        subdirs[:] = []

    return records


def _latest_per_model(records: list[dict]) -> dict[str, dict]:
    """Keep only the latest run per model family that has both metrics.

    Uses the actual filesystem modification time of the config.json file
    to determine recency, so it works across naming conventions
    (experiment_MMDD-..., DAG_RUN_YYYYMMDD-..., custom names, etc.).
    """
    best = {}
    for r in records:
        name = r["model_name"]
        prev = best.get(name)
        if prev is None:
            best[name] = r
        else:
            best[name] = _prefer_record(prev, r)
    return best


def select_latest_per_depth(records: list[dict]) -> dict[str, dict]:
    """Keep the latest run per depth for one model family."""
    best = {}
    for record in records:
        depth = record["config"].get("dag_spec", {}).get("depth")
        if depth is None:
            continue
        label = _depth_label(depth)
        previous = best.get(label)
        if previous is None:
            best[label] = record
        else:
            best[label] = _prefer_record(previous, record)

    return dict(sorted(best.items(), key=lambda item: _depth_sort_key(item[0])))


def build_depth_comparison_output_dir(output_dir: str, model_name: str) -> str:
    """Create and return the output directory for one model's cross-depth comparison."""
    resolved = os.path.join(output_dir, "depth_comparison", model_name)
    os.makedirs(resolved, exist_ok=True)
    return resolved


def list_available_model_names(records: list[dict]) -> list[str]:
    """Return sorted canonical model names present in *records*."""
    return sorted({record["model_name"] for record in records})


# ---------------------------------------------------------------------------
# Legacy DAG hook
# ---------------------------------------------------------------------------

def plot_dag(output_dir: str):
    """Deprecated no-op retained for notebook/backward compatibility."""
    print("[skip] DAG topology generation has been removed; no dag_topology.pdf will be created.")


# ---------------------------------------------------------------------------
# 2.  Training Curves
# ---------------------------------------------------------------------------

def plot_training_curves(
    records: dict[str, dict],
    output_dir: str,
    style_map: dict[str, dict] | None = None,
    show: bool = False,
):
    """Plot train/val loss and val mIoU / pixel-acc curves per model."""
    models_with_emp = {k: v for k, v in records.items() if v["empirical"] is not None}
    if not models_with_emp:
        print("[skip] No empirical data - training curves skipped.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax_train, ax_val, ax_metric = axes

    metric_label = "Val Dice"  # default; overwritten per model if miou present
    for name, rec in models_with_emp.items():
        emp = rec["empirical"]
        sty = _get_style(name, style_map)
        train_history = emp.get("train_loss_history")
        val_history = emp.get("val_loss_history")
        if not train_history or not val_history:
            print(f"[warn] {name} has incomplete loss history - skipping training curves for this model.")
            continue

        epochs = list(range(1, len(train_history) + 1))

        ax_train.plot(epochs, train_history,
                      color=sty["color"], marker=sty["marker"], markersize=5,
                      label=sty["label"])
        ax_val.plot(epochs, val_history,
                    color=sty["color"], marker=sty["marker"], markersize=5,
                    label=sty["label"])

        # mIoU if present, else val_dice
        metric_key = "val_miou_history" if "val_miou_history" in emp else "val_dice_history"
        metric_label = "Val mIoU" if "val_miou_history" in emp else "Val Dice"
        if metric_key in emp:
            ax_metric.plot(epochs, emp[metric_key],
                           color=sty["color"], marker=sty["marker"], markersize=5,
                           label=sty["label"])

    ax_train.set_xlabel("Epoch")
    ax_train.set_ylabel("Training Loss")
    ax_train.set_title("Training Loss")
    ax_train.legend()

    ax_val.set_xlabel("Epoch")
    ax_val.set_ylabel("Validation Loss")
    ax_val.set_title("Validation Loss")
    ax_val.legend()

    ax_metric.set_xlabel("Epoch")
    ax_metric.set_ylabel(metric_label)
    ax_metric.set_title(metric_label)
    ax_metric.legend()

    fig.tight_layout()
    path = os.path.join(output_dir, "training_curves.png")
    fig.savefig(path)
    if show:
        plt.show()
    plt.close(fig)
    print(f"  Training curves saved -> {path}")


# ---------------------------------------------------------------------------
# 3.  Theoretical Metrics Comparison (bar charts)
# ---------------------------------------------------------------------------

def plot_theoretical_bars(
    records: dict[str, dict],
    output_dir: str,
    style_map: dict[str, dict] | None = None,
    show: bool = False,
):
    """Bar charts comparing NNGP/NTK eigenvalues and complexity across models."""
    models_with_theo = {k: v for k, v in records.items() if v["theoretical"] is not None}
    if not models_with_theo:
        print("[skip] No theoretical data - bar charts skipped.")
        return

    names = list(models_with_theo.keys())
    colors = [_get_style(n, style_map).get("color", "gray") for n in names]
    labels = [_get_style(n, style_map).get("label", n) for n in names]

    # Gather values
    nngp_min = [models_with_theo[n]["theoretical"]["nngp_eigenvalues"][0] for n in names]
    nngp_max = [models_with_theo[n]["theoretical"]["nngp_eigenvalues"][-1] for n in names]
    ntk_min = [models_with_theo[n]["theoretical"]["ntk_eigenvalues"][0] for n in names]
    lengths = [models_with_theo[n]["theoretical"]["complexity_length"] for n in names]
    curvatures = [models_with_theo[n]["theoretical"]["complexity_curvature"] for n in names]

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    panels = [
        (axes[0, 0], nngp_min, r"$\lambda_{\min}$ NNGP", "NNGP Min Eigenvalue"),
        (axes[0, 1], nngp_max, r"$\lambda_{\max}$ NNGP", "NNGP Max Eigenvalue"),
        (axes[0, 2], [mx / mn if mn > 0 else 0 for mn, mx in zip(nngp_min, nngp_max)],
         r"$\lambda_{\max}/\lambda_{\min}$ NNGP", "NNGP Condition Ratio"),
        (axes[1, 0], ntk_min, r"$\lambda_{\min}$ NTK", "NTK Min Eigenvalue"),
        (axes[1, 1], lengths, "Curve Length", "Complexity (Length)"),
        (axes[1, 2], curvatures, "Extrinsic Curvature", "Complexity (Curvature)"),
    ]

    x = np.arange(len(names))
    for ax, values, ylabel, title in panels:
        bars = ax.bar(x, values, color=colors, edgecolor="black", width=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=13)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.set_title(title, fontsize=17)
        # Add value labels on bars
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.2e}" if abs(v) > 1e3 else f"{v:.4f}",
                    ha="center", va="bottom", fontsize=11)

    fig.tight_layout()
    path = os.path.join(output_dir, "theoretical_comparison.png")
    fig.savefig(path)
    if show:
        plt.show()
    plt.close(fig)
    print(f"  Theoretical comparison saved -> {path}")


# ---------------------------------------------------------------------------
# 4.  Correlation Scatter Plots (Theoretical <-> Empirical)
# ---------------------------------------------------------------------------

def _scatter_with_corr(ax, x_vals, y_vals, names, xlabel, ylabel, title, style_map: dict[str, dict] | None = None):
    """Helper: scatter with per-point labels and Kendall tau annotation."""
    for xv, yv, n in zip(x_vals, y_vals, names):
        sty = _get_style(n, style_map)
        ax.scatter(xv, yv, c=sty["color"], marker=sty["marker"], s=120,
                   edgecolors="black", zorder=5, label=sty["label"])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=10)

    if len(x_vals) >= 3:
        corr, p = stats.kendalltau(x_vals, y_vals)
        ax.annotate(f"Kendall $\\tau$={corr:.2f}\n$p$={p:.3f}",
                    xy=(0.05, 0.95), xycoords="axes fraction",
                    va="top", fontsize=13,
                    bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.7))

    # De-duplicate legend entries
    handles, lbls = ax.get_legend_handles_labels()
    by_label = dict(zip(lbls, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="best")


def plot_correlation_scatter(
    records: dict[str, dict],
    output_dir: str,
    style_map: dict[str, dict] | None = None,
    show: bool = False,
):
    """
    Scatter plots relating theoretical and empirical metrics.

    Adapted from reference notebook visualizations:
    - Convergence vs. NNGP lambda_min
    - Expressivity vs. Curvature
    - Generalization vs. NTK lambda_min
    - Complexity (length) vs. Convergence
    """
    complete = {k: v for k, v in records.items()
                if v["theoretical"] is not None and v["empirical"] is not None}
    if len(complete) < 2:
        print("[skip] Need >=2 models with both metrics for correlation plots.")
        return

    names = list(complete.keys())

    def _theo(n, key):
        return complete[n]["theoretical"][key]

    def _emp(n, key):
        return complete[n]["empirical"][key]

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    # --- Row 1: Convergence correlations ---
    # 1a  NNGP lambda_min vs Convergence
    _scatter_with_corr(
        axes[0, 0],
        [_theo(n, "nngp_eigenvalues")[0] for n in names],
        [_emp(n, "convergence_epochs_to_threshold_50_percent") for n in names],
        names,
        r"$\lambda_{\min}$ of NNGP",
        "Epochs to 50% Loss",
        "Convergence vs. NNGP",
        style_map=style_map,
    )

    # 1b  Complexity length vs Convergence
    _scatter_with_corr(
        axes[0, 1],
        [_theo(n, "complexity_length") for n in names],
        [_emp(n, "convergence_epochs_to_threshold_50_percent") for n in names],
        names,
        "Curve Length",
        "Epochs to 50% Loss",
        "Convergence vs. Length",
        style_map=style_map,
    )

    # 1c  Curvature vs Expressivity (converged training loss)
    _scatter_with_corr(
        axes[0, 2],
        [_theo(n, "complexity_curvature") for n in names],
        [_emp(n, "expressivity_converged_train_loss") for n in names],
        names,
        "Extrinsic Curvature",
        "Training Loss (converged)",
        "Expressivity vs. Curvature",
        style_map=style_map,
    )

    # --- Row 2: Generalization correlations ---
    # 2a  NTK lambda_min vs Generalization gap
    _scatter_with_corr(
        axes[1, 0],
        [_theo(n, "ntk_eigenvalues")[0] for n in names],
        [_emp(n, "generalization_gap") for n in names],
        names,
        r"$\lambda_{\min}$ of NTK",
        "Generalization Gap",
        "Generalization vs. NTK",
        style_map=style_map,
    )

    # 2b  NNGP condition number vs final val loss
    _scatter_with_corr(
        axes[1, 1],
        [_theo(n, "nngp_eigenvalues")[-1] / max(_theo(n, "nngp_eigenvalues")[0], 1e-10)
         for n in names],
        [_emp(n, "final_val_loss") for n in names],
        names,
        r"$\lambda_{\max}/\lambda_{\min}$ NNGP",
        "Final Val Loss",
        "Val Loss vs. NNGP Condition",
        style_map=style_map,
    )

    # 2c  Complexity length vs Generalization gap
    _scatter_with_corr(
        axes[1, 2],
        [_theo(n, "complexity_length") for n in names],
        [_emp(n, "generalization_gap") for n in names],
        names,
        "Curve Length",
        "Generalization Gap",
        "Generalization vs. Length",
        style_map=style_map,
    )

    fig.tight_layout()
    path = os.path.join(output_dir, "correlation_scatter.png")
    fig.savefig(path)
    if show:
        plt.show()
    plt.close(fig)
    print(f"  Correlation scatter saved -> {path}")


# ---------------------------------------------------------------------------
# 5.  Eigenvalue Spectrum
# ---------------------------------------------------------------------------

def _sanitize_for_log_plot(values: list[float], floor: float = 1e-12) -> tuple[list[float], int]:
    """Clip non-positive values so they can be shown on a log-scale plot."""
    clipped = []
    replaced = 0
    for value in values:
        if value <= 0:
            clipped.append(floor)
            replaced += 1
        else:
            clipped.append(value)
    return clipped, replaced

def plot_eigenvalue_spectrum(
    records: dict[str, dict],
    output_dir: str,
    style_map: dict[str, dict] | None = None,
    show: bool = False,
):
    """Plot NTK and NNGP eigenvalue distributions side by side."""
    models_with_theo = {k: v for k, v in records.items() if v["theoretical"] is not None}
    if not models_with_theo:
        print("[skip] No theoretical data - eigenvalue spectrum skipped.")
        return

    fig, (ax_nngp, ax_ntk) = plt.subplots(1, 2, figsize=(14, 5))

    for name, rec in models_with_theo.items():
        theo = rec["theoretical"]
        sty = _get_style(name, style_map)

        nngp_eigs, nngp_replaced = _sanitize_for_log_plot(sorted(theo["nngp_eigenvalues"]))
        ntk_eigs, ntk_replaced = _sanitize_for_log_plot(sorted(theo["ntk_eigenvalues"]))
        idx_nngp = list(range(1, len(nngp_eigs) + 1))
        idx_ntk = list(range(1, len(ntk_eigs) + 1))

        if nngp_replaced:
            print(f"[warn] {name} has {nngp_replaced} non-positive NNGP eigenvalue(s); clipped for log-scale plotting.")
        if ntk_replaced:
            print(f"[warn] {name} has {ntk_replaced} non-positive NTK eigenvalue(s); clipped for log-scale plotting.")

        ax_nngp.semilogy(idx_nngp, nngp_eigs, marker=sty["marker"], color=sty["color"],
                         label=sty["label"], linewidth=2, markersize=8)
        ax_ntk.semilogy(idx_ntk, ntk_eigs, marker=sty["marker"], color=sty["color"],
                        label=sty["label"], linewidth=2, markersize=8)

    ax_nngp.set_xlabel("Eigenvalue Index")
    ax_nngp.set_ylabel("Eigenvalue (log scale)")
    ax_nngp.set_title("NNGP Eigenvalue Spectrum")
    ax_nngp.legend()

    ax_ntk.set_xlabel("Eigenvalue Index")
    ax_ntk.set_ylabel("Eigenvalue (log scale)")
    ax_ntk.set_title("NTK Eigenvalue Spectrum")
    ax_ntk.legend()

    fig.tight_layout()
    path = os.path.join(output_dir, "eigenvalue_spectrum.png")
    fig.savefig(path)
    if show:
        plt.show()
    plt.close(fig)
    print(f"  Eigenvalue spectrum saved -> {path}")


# ---------------------------------------------------------------------------
# 6.  Multi-Objective Ranking Table
# ---------------------------------------------------------------------------

def _rank(values, ascending=True):
    """Return 1-based ranks for a list of values."""
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=not ascending)
    ranks = [0] * len(values)
    for rank, idx in enumerate(order, start=1):
        ranks[idx] = rank
    return ranks


def print_ranking_table(records: dict[str, dict], style_map: dict[str, dict] | None = None):
    """
    Print a multi-objective ranking table (Convergence, Expressivity, Generalization)
    following the reference notebook's ranking comparison methodology.
    """
    complete = {k: v for k, v in records.items()
                if v["theoretical"] is not None and v["empirical"] is not None}
    if not complete:
        print("[skip] No complete results for ranking.")
        return

    names = list(complete.keys())
    labels = [_get_style(n, style_map).get("label", n) for n in names]

    # Metrics (lower is better for all)
    convergence = [complete[n]["empirical"]["convergence_epochs_to_threshold_50_percent"] for n in names]
    expressivity = [complete[n]["empirical"]["expressivity_converged_train_loss"] for n in names]
    generalization = [complete[n]["empirical"]["generalization_gap"] for n in names]

    r_conv = _rank(convergence, ascending=True)
    r_expr = _rank(expressivity, ascending=True)
    r_gen = _rank(generalization, ascending=True)

    header = f"{'Model':<14} {'Conv.Eps':>10} {'Rank':>5} {'Train Loss':>11} {'Rank':>5} {'Gen. Gap':>10} {'Rank':>5} {'Sum':>5}"
    sep = "-" * len(header)
    print("\n" + sep)
    print("  Multi-Objective Ranking (lower rank = better)")
    print(sep)
    print(header)
    print(sep)

    for i, (lbl, n) in enumerate(zip(labels, names)):
        total = r_conv[i] + r_expr[i] + r_gen[i]
        print(f"{lbl:<14} {convergence[i]:>10} {r_conv[i]:>5} "
              f"{expressivity[i]:>11.6f} {r_expr[i]:>5} "
              f"{generalization[i]:>10.6f} {r_gen[i]:>5} {total:>5}")

    print(sep)

    # Also include theoretical metrics summary
    print("\n  Theoretical Metrics Summary")
    print(sep)
    theo_header = (f"{'Model':<14} {'NNGP lmin':>12} {'NNGP lmax':>12} "
                   f"{'NTK lmin':>12} {'Length':>12} {'Curvature':>10}")
    print(theo_header)
    print(sep)
    for lbl, n in zip(labels, names):
        t = complete[n]["theoretical"]
        print(f"{lbl:<14} {t['nngp_eigenvalues'][0]:>12.4f} {t['nngp_eigenvalues'][-1]:>12.4f} "
              f"{t['ntk_eigenvalues'][0]:>12.1f} {t['complexity_length']:>12.1f} "
              f"{t['complexity_curvature']:>10.6f}")
    print(sep + "\n")


def save_ranking_table(
    records: dict[str, dict],
    output_dir: str,
    style_map: dict[str, dict] | None = None,
):
    """Save ranking table as a text file."""
    import io
    import sys
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        print_ranking_table(records, style_map=style_map)
    finally:
        sys.stdout = old_stdout
    text = buf.getvalue()
    if text.strip():
        path = os.path.join(output_dir, "ranking_table.txt")
        with open(path, "w") as f:
            f.write(text)
        print(f"  Ranking table saved -> {path}")
    # Also print to console
    print(text)


# ---------------------------------------------------------------------------
# 7.  Empirical Summary Bar Chart
# ---------------------------------------------------------------------------

def plot_empirical_summary(
    records: dict[str, dict],
    output_dir: str,
    style_map: dict[str, dict] | None = None,
    show: bool = False,
):
    """Bar chart of final empirical metrics per model."""
    models_with_emp = {k: v for k, v in records.items() if v["empirical"] is not None}
    if not models_with_emp:
        print("[skip] No empirical data - summary skipped.")
        return

    names = list(models_with_emp.keys())
    colors = [_get_style(n, style_map).get("color", "gray") for n in names]
    labels = [_get_style(n, style_map).get("label", n) for n in names]

    # Collect metrics
    val_loss = [models_with_emp[n]["empirical"]["final_val_loss"] for n in names]
    gen_gap = [models_with_emp[n]["empirical"]["generalization_gap"] for n in names]

    # mIoU or dice
    metric_key = "final_val_miou" if "final_val_miou" in models_with_emp[names[0]]["empirical"] else "final_val_dice"
    metric_label = "Val mIoU" if metric_key == "final_val_miou" else "Val Dice"
    perf = [models_with_emp[n]["empirical"].get(metric_key, 0) for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(names))

    for ax, values, ylabel, title in [
        (axes[0], perf, metric_label, f"Final {metric_label}"),
        (axes[1], val_loss, "Val Loss", "Final Validation Loss"),
        (axes[2], gen_gap, "Gap", "Generalization Gap (Val-Train)"),
    ]:
        bars = ax.bar(x, values, color=colors, edgecolor="black", width=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=13)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.set_title(title, fontsize=17)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.4f}", ha="center", va="bottom", fontsize=12)

    fig.tight_layout()
    path = os.path.join(output_dir, "empirical_summary.png")
    fig.savefig(path)
    if show:
        plt.show()
    plt.close(fig)
    print(f"  Empirical summary saved -> {path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_for_target(target_dir: str, experiments_dir: str = "experiments", output_dir: str = "visualization") -> bool:
    """Generate every visualization for one selected experiment target."""
    target_dir = os.path.abspath(target_dir)
    target_label = describe_target(target_dir, experiments_dir)
    target_output_dir = build_output_dir_for_target(output_dir, target_dir, experiments_dir)

    all_records = load_experiments(target_dir, experiments_root=experiments_dir)
    if not all_records:
        print(f"[skip] No experiment results found in {target_dir}")
        return False

    records = _latest_per_model(all_records)
    print(f"\n=== Visualizing {target_label} ===")
    print(f"Loaded {len(records)} model(s): {', '.join(records.keys())}")
    print(f"Source directory: {target_dir}")
    print(f"Output directory: {target_output_dir}\n")

    # 1. Training curves
    plot_training_curves(records, target_output_dir)

    # 2. Eigenvalue spectrum
    plot_eigenvalue_spectrum(records, target_output_dir)

    # 3. Theoretical bar charts
    plot_theoretical_bars(records, target_output_dir)

    # 4. Correlation scatter
    plot_correlation_scatter(records, target_output_dir)

    # 5. Empirical summary
    plot_empirical_summary(records, target_output_dir)

    # 6. Ranking table
    save_ranking_table(records, target_output_dir)

    print(f"Finished: {target_label}")
    return True


def generate_depth_comparison_for_model(
    model_name: str,
    records: list[dict],
    output_dir: str = "visualization",
) -> bool:
    """Generate cross-depth comparison figures for one architecture family."""
    canonical_name = normalize_model_name(model_name)
    model_records = [record for record in records if record["model_name"] == canonical_name]
    if not model_records:
        print(f"[skip] No experiment results found for model: {canonical_name}")
        return False

    records_by_depth = select_latest_per_depth(model_records)
    if not records_by_depth:
        print(f"[skip] No depth-tagged experiment results found for model: {canonical_name}")
        return False

    style_map = build_depth_style_map(records_by_depth.keys())
    target_output_dir = build_depth_comparison_output_dir(output_dir, canonical_name)

    print(f"\n=== Depth Comparison: {canonical_name} ===")
    print(f"Loaded depths: {', '.join(records_by_depth.keys())}")
    print(f"Output directory: {target_output_dir}\n")

    plot_training_curves(records_by_depth, target_output_dir, style_map=style_map)
    plot_eigenvalue_spectrum(records_by_depth, target_output_dir, style_map=style_map)
    plot_theoretical_bars(records_by_depth, target_output_dir, style_map=style_map)
    plot_correlation_scatter(records_by_depth, target_output_dir, style_map=style_map)
    plot_empirical_summary(records_by_depth, target_output_dir, style_map=style_map)
    save_ranking_table(records_by_depth, target_output_dir, style_map=style_map)

    print(f"Finished: depth comparison for {canonical_name}")
    return True


def generate_depth_comparisons(
    experiments_dir: str = "experiments",
    output_dir: str = "visualization",
    models: list[str] | None = None,
):
    """Generate cross-depth comparisons for one or more architecture families."""
    os.makedirs(output_dir, exist_ok=True)
    all_records = load_experiments(experiments_dir, experiments_root=experiments_dir)
    if not all_records:
        print(f"No experiment results found in {experiments_dir}/")
        return

    available_models = list_available_model_names(all_records)
    if models:
        seen = set()
        selected_models = []
        for model in models:
            canonical = normalize_model_name(model)
            if canonical not in seen:
                seen.add(canonical)
                selected_models.append(canonical)
    else:
        selected_models = available_models

    generated = 0
    for model_name in selected_models:
        generated += int(generate_depth_comparison_for_model(model_name, all_records, output_dir))

    if generated:
        print(f"\nGenerated depth-comparison visualizations for {generated} architecture(s).")
    else:
        print("\nNo depth-comparison figures were generated.")


def resolve_requested_targets(experiments_dir: str, experiments: list[str] | None = None) -> list[str]:
    """Resolve requested experiment selectors, or fall back to every top-level experiment."""
    if experiments:
        seen = set()
        targets = []
        for experiment in experiments:
            resolved = resolve_experiment_target(experiments_dir, experiment)
            if resolved not in seen:
                seen.add(resolved)
                targets.append(resolved)
        return targets
    return default_experiment_targets(experiments_dir)


def generate_all(
    experiments_dir: str = "experiments",
    output_dir: str = "visualization",
    experiments: list[str] | None = None,
):
    """Generate visualizations for one or more experiment targets."""
    os.makedirs(output_dir, exist_ok=True)
    targets = resolve_requested_targets(experiments_dir, experiments)
    if not targets:
        print(f"No experiment results found in {experiments_dir}/")
        return

    generated = 0
    for target_dir in targets:
        generated += int(generate_for_target(target_dir, experiments_dir, output_dir))

    if generated:
        print(f"\nGenerated visualizations for {generated} experiment target(s).")
    else:
        print("\nNo figures were generated.")


def main():
    parser = argparse.ArgumentParser(description="Generate UNet analysis visualizations.")
    parser.add_argument("--experiments_dir", type=str, default="experiments",
                        help="Directory containing experiment results.")
    parser.add_argument("--output_dir", type=str, default="visualization",
                        help="Directory to save generated figures.")
    parser.add_argument("--compare_depths", action="store_true",
                        help="Compare the same architecture across depths. Saves to <output_dir>/depth_comparison/<ModelName>.")
    parser.add_argument("--experiment", action="append", default=None,
                        help="Experiment folder to visualize. Can be relative to --experiments_dir or absolute. Repeat for multiple targets.")
    parser.add_argument("--model", action="append", default=None,
                        help="Model family to visualize when using --compare_depths. Repeat for multiple models.")
    parser.add_argument("--list_experiments", action="store_true",
                        help="List top-level experiment folders detected under --experiments_dir.")
    parser.add_argument("--list_models", action="store_true",
                        help="List model families detected under --experiments_dir.")
    args = parser.parse_args()

    if args.list_models:
        all_records = load_experiments(args.experiments_dir, experiments_root=args.experiments_dir)
        model_names = list_available_model_names(all_records)
        if not model_names:
            print(f"No experiment results found in {args.experiments_dir}/")
            return
        print("Available model families:")
        for model_name in model_names:
            print(f"  - {model_name}")
        return

    if args.list_experiments:
        targets = default_experiment_targets(args.experiments_dir)
        if not targets:
            print(f"No experiment results found in {args.experiments_dir}/")
            return
        print("Available experiment targets:")
        for target in targets:
            print(f"  - {describe_target(target, args.experiments_dir)}")
        return

    if args.compare_depths:
        generate_depth_comparisons(args.experiments_dir, args.output_dir, args.model)
        return

    generate_all(args.experiments_dir, args.output_dir, args.experiment)


if __name__ == "__main__":
    main()

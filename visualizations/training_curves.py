"""Utilities for plotting training metrics from CSV files or Weights & Biases."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd

# Type alias for optional series transform functions.
SeriesTransform = Optional[Callable[[pd.Series], pd.Series]]


@dataclass
class SeriesSpec:
    """Configuration for a single plotted series."""

    column: str
    label: str
    color: str = "blue"
    linestyle: str = "-"
    linewidth: float = 3.0
    marker: Optional[str] = None
    transform: SeriesTransform = None


@dataclass
class PanelConfig:
    """Configuration for a subplot panel."""

    series: Sequence[SeriesSpec]
    ylabel: Optional[str] = None
    xlabel: Optional[str] = None
    ylim: Optional[Tuple[float, float]] = None
    yticks: Optional[Sequence[float]] = None
    xscale: Optional[str] = "log"
    xlim: Optional[Tuple[float, float]] = None
    grid: bool = False
    legend: bool = True
    legend_loc: str = "upper right"
    legend_fontsize: Optional[int] = None
    hide_x_ticks: bool = False
    tick_labelsize: Optional[int] = None
    xticks: Optional[Sequence[float]] = None
    xticklabels: Optional[Sequence[str]] = None
    twin_series: Sequence[SeriesSpec] = ()
    twin_ylabel: Optional[str] = None
    twin_ylim: Optional[Tuple[float, float]] = None
    twin_yticks: Optional[Sequence[float]] = None
    twin_tick_labelsize: Optional[int] = None
    combine_legends: bool = True
    twin_legend: bool = False
    twin_legend_loc: str = "upper right"


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_training_metrics_from_csv(csv_path: str | Path, *, sort_by: Optional[str] = "epoch") -> pd.DataFrame:
    """Load a training metrics CSV file sorted by the desired column."""

    df = pd.read_csv(csv_path)
    if sort_by and sort_by in df.columns:
        df = df.sort_values(sort_by)
    return df


def load_training_metrics_from_wandb(
    entity: str,
    project: str,
    run_name: str,
    *,
    metrics: Optional[Sequence[str]] = None,
    sort_by: Optional[str] = "epoch",
) -> pd.DataFrame:
    """Fetch training metrics recorded on Weights & Biases."""

    try:
        import wandb  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("wandb must be installed to download metrics") from exc

    if metrics is None:
        metrics = (
            "epoch",
            "test_error",
            "train_error_total",
            "train_error_noisy",
            "train_error_clean",
            "train_loss",
            "test_loss",
        )

    api = wandb.Api()
    run_id = None
    for run in api.runs(f"{entity}/{project}"):
        if run.name == run_name:
            run_id = run.id
            break

    if run_id is None:
        raise ValueError(f"Run name '{run_name}' was not found under {entity}/{project}.")

    run = api.run(f"{entity}/{project}/{run_id}")
    history = run.history(keys=list(metrics))
    df = pd.DataFrame(history)

    if sort_by and sort_by in df.columns:
        df = df.dropna(subset=[sort_by]).sort_values(sort_by)

    return df


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def collect_metrics_by_width(
    root_dir: str | Path,
    target_epochs: Sequence[int],
    *,
    metrics: Sequence[str],
    pattern: str = r"width(\d+)",
    csv_name: str = "training_metrics.csv",
) -> pd.DataFrame:
    """Collect metrics at the first matching epoch for each width directory."""

    root = Path(root_dir)
    results: List[dict[str, float | int]] = []

    for entry in root.iterdir():
        if not entry.is_dir():
            continue

        match = re.search(pattern, entry.name)
        if not match:
            continue
        csv_file = entry / "csv" / csv_name
        if not csv_file.exists():
            continue

        df = pd.read_csv(csv_file)
        selected_row = None
        selected_epoch = None
        for epoch in target_epochs:
            rows = df[df["epoch"] == epoch]
            if not rows.empty:
                selected_row = rows.iloc[0]
                selected_epoch = int(epoch)
                break
        if selected_row is None:
            continue

        width_value = int(match.group(1))
        record: dict[str, float | int] = {"width": width_value, "epoch": selected_epoch}
        for metric_name in metrics:
            if metric_name not in selected_row:
                raise KeyError(f"Metric '{metric_name}' not present in {csv_file}")
            record[metric_name] = float(selected_row[metric_name])
        results.append(record)

    if not results:
        return pd.DataFrame(columns=["width", "epoch", *metrics])

    df_result = pd.DataFrame(results).sort_values("width")
    return df_result


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _apply_series(ax, df: pd.DataFrame, spec: SeriesSpec, x: str) -> None:
    """Plot a single series on the given axis."""

    y = df[spec.column]
    if spec.transform is not None:
        y = spec.transform(y)
    ax.plot(
        df[x],
        y,
        label=spec.label,
        color=spec.color,
        linestyle=spec.linestyle,
        linewidth=spec.linewidth,
        marker=spec.marker,
    )


def plot_panels(
    df: pd.DataFrame,
    panels: Sequence[PanelConfig],
    output_base: str | Path,
    *,
    x: str = "epoch",
    vertical_lines: Sequence[float] | None = None,
    figure_size: Tuple[float, float] | None = None,
    font_family: Optional[str] = None,
    font_size: Optional[int] = None,
    output_formats: Sequence[str] = ("svg", "pdf"),
) -> List[str]:
    """Plot one or more panels and save to the requested formats."""

    rc_updates = {}
    if font_family is not None:
        rc_updates["font.family"] = font_family
    if font_size is not None:
        rc_updates["font.size"] = font_size

    if figure_size is None:
        figure_size = (11, 13)

    saved_paths: List[str] = []
    with plt.rc_context(rc_updates):
        fig, axes = plt.subplots(len(panels), 1, figsize=figure_size, constrained_layout=True)
        if len(panels) == 1:
            axes = [axes]  # type: ignore[list-as-list]

        for ax, panel in zip(axes, panels):
            for spec in panel.series:
                _apply_series(ax, df, spec, x)

            if panel.xscale:
                ax.set_xscale(panel.xscale)
            if panel.xlim is not None:
                ax.set_xlim(panel.xlim)
            if panel.ylim is not None:
                ax.set_ylim(panel.ylim)
            if panel.ylabel is not None:
                ax.set_ylabel(panel.ylabel)
            if panel.yticks is not None:
                ax.set_yticks(panel.yticks)
            if panel.grid:
                ax.grid(True, which="both", linestyle="--", linewidth=0.3)
            if panel.xticks is not None:
                ax.set_xticks(panel.xticks)
                if panel.xticklabels is not None:
                    ax.set_xticklabels(panel.xticklabels)
            tick_kwargs = {}
            if panel.tick_labelsize is not None:
                tick_kwargs["labelsize"] = panel.tick_labelsize
            if panel.hide_x_ticks:
                tick_kwargs["labelbottom"] = False
            if tick_kwargs:
                ax.tick_params(axis="both", **tick_kwargs)
            elif panel.hide_x_ticks:
                ax.tick_params(axis="x", labelbottom=False)
            if panel.xlabel is not None:
                ax.set_xlabel(panel.xlabel)

            twin_handles: List = []
            twin_labels: List[str] = []
            if panel.twin_series:
                twin_ax = ax.twinx()
                for spec in panel.twin_series:
                    _apply_series(twin_ax, df, spec, x)
                if panel.twin_ylabel is not None:
                    twin_ax.set_ylabel(panel.twin_ylabel)
                if panel.twin_ylim is not None:
                    twin_ax.set_ylim(panel.twin_ylim)
                if panel.twin_yticks is not None:
                    twin_ax.set_yticks(panel.twin_yticks)
                if panel.twin_tick_labelsize is not None:
                    twin_ax.tick_params(axis="y", labelsize=panel.twin_tick_labelsize)
                if panel.combine_legends:
                    twin_handles, twin_labels = twin_ax.get_legend_handles_labels()
                elif panel.twin_legend:
                    twin_ax.legend(loc=panel.twin_legend_loc, fontsize=panel.legend_fontsize)

            if vertical_lines:
                for value in vertical_lines:
                    ax.axvline(x=value, color="black", linestyle="-", linewidth=0.9, zorder=0)

            handles, labels = ax.get_legend_handles_labels()
            if panel.legend:
                if panel.combine_legends and twin_handles:
                    handles = handles + twin_handles
                    labels = labels + twin_labels
                ax.legend(handles, labels, loc=panel.legend_loc, fontsize=panel.legend_fontsize)

        output_base = str(output_base)
        for fmt in output_formats:
            path = f"{output_base}.{fmt}"
            fig.savefig(path, format=fmt)
            saved_paths.append(path)
        plt.close(fig)

    return saved_paths


def plot_metric_vs_width(
    df: pd.DataFrame,
    metric: str,
    output_path: str | Path,
    *,
    x_col: str = "width",
    xlabel: str = "Model Width",
    ylabel: Optional[str] = None,
    color: str = "blue",
    marker: Optional[str] = "o",
    grid: bool = True,
    xscale: Optional[str] = None,
    figure_size: Tuple[float, float] = (8, 5),
) -> str:
    """Plot a single metric against model width."""

    if df.empty:
        raise ValueError("No data available to plot.")

    if ylabel is None:
        ylabel = metric.replace("_", " ").title()

    with plt.rc_context({}):
        fig, ax = plt.subplots(figsize=figure_size)
        ax.plot(df[x_col], df[metric], color=color, marker=marker)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if xscale:
            ax.set_xscale(xscale)
        if grid:
            ax.grid(True)
        fig.tight_layout()
        output_path = str(output_path)
        fig.savefig(output_path)
        plt.close(fig)
    return output_path


def plot_loss_curves_by_width(
    root_dir: str | Path,
    output_dir: str | Path,
    *,
    metrics: Sequence[str] = ("avg_loss_noisy", "avg_loss_clean"),
    pattern: str = r"width(\d+)",
    csv_name: str = "training_metrics.csv",
    x_column: str = "epoch",
    xscale: Optional[str] = "log",
    xlabel: str = "Epoch",
    ylabel: str = "Loss",
    ylim: Optional[Tuple[float, float]] = None,
) -> List[str]:
    """Plot loss curves for every width directory under ``root_dir``."""

    root = Path(root_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []

    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        match = re.search(pattern, entry.name)
        if not match:
            continue
        csv_file = entry / "csv" / csv_name
        if not csv_file.exists():
            continue

        df = pd.read_csv(csv_file)
        with plt.rc_context({}):
            fig, ax = plt.subplots()
            for metric_name in metrics:
                if metric_name not in df.columns:
                    raise KeyError(f"Column '{metric_name}' not found in {csv_file}")
                ax.plot(df[x_column], df[metric_name], label=metric_name)
            if xscale:
                ax.set_xscale(xscale)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            if ylim is not None:
                ax.set_ylim(ylim)
            ax.set_title(f"Loss curves (width={match.group(1)})")
            ax.legend()
            ax.grid(True, which="both", ls="--")
            fig.tight_layout()
            output_path = output / f"avg_loss_width{match.group(1)}.png"
            fig.savefig(output_path)
            plt.close(fig)
            saved.append(str(output_path))

    return saved


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

_DEFAULT_FIGSIZE = {
    "triple": (11, 19),
    "double": (13, 10),
    "double-twin": (16, 16),
    "single-twin": (10, 8),
}


def _build_default_layout(
    df: pd.DataFrame,
    layout: str,
    scale_errors: Optional[float],
) -> Tuple[pd.DataFrame, Sequence[PanelConfig]]:
    """Create default panel configurations for the CLI commands."""

    df_plot = df.copy()
    error_columns = (
        "test_error",
        "train_error_total",
        "train_error_noisy",
        "train_error_clean",
    )

    def scaled(column: str) -> str:
        if scale_errors is None:
            return column
        scaled_column = f"{column}_scaled"
        df_plot[scaled_column] = df_plot[column] / scale_errors
        return scaled_column

    test_error = scaled("test_error")
    train_error = scaled("train_error_total")
    train_noisy = scaled("train_error_noisy")
    train_clean = scaled("train_error_clean")

    if layout == "triple":
        panels = [
            PanelConfig(
                series=(
                    SeriesSpec(test_error, "test", color="red"),
                    SeriesSpec(train_error, "train", color="blue"),
                ),
                ylabel="Error",
                hide_x_ticks=True,
                xscale="log",
            ),
            PanelConfig(
                series=(SeriesSpec(train_noisy, "train noisy", color="purple", linestyle="--"),),
                twin_series=(SeriesSpec(train_clean, "train clean", color="green"),),
                ylabel="Train noisy error",
                twin_ylabel="Train clean error",
                hide_x_ticks=True,
                xscale="log",
            ),
            PanelConfig(
                series=(
                    SeriesSpec("test_loss", "test loss", color="red"),
                    SeriesSpec("train_loss", "train loss", color="blue"),
                ),
                ylabel="Loss",
                xlabel="epoch",
                xscale="log",
            ),
        ]
    elif layout == "double":
        panels = [
            PanelConfig(
                series=(
                    SeriesSpec(test_error, "test", color="red"),
                    SeriesSpec(train_error, "train", color="blue"),
                ),
                ylabel="Error",
                hide_x_ticks=True,
                xscale="log",
            ),
            PanelConfig(
                series=(
                    SeriesSpec(train_noisy, "train noisy", color="darkblue"),
                    SeriesSpec(train_clean, "train clean", color="cyan"),
                ),
                ylabel="Train error",
                xlabel="epoch",
                xscale="log",
            ),
        ]
    elif layout == "double-twin":
        panels = [
            PanelConfig(
                series=(
                    SeriesSpec(test_error, "test", color="red"),
                    SeriesSpec(train_error, "train", color="blue"),
                ),
                ylabel="Error",
                hide_x_ticks=True,
                xscale="log",
            ),
            PanelConfig(
                series=(SeriesSpec(train_noisy, "train noisy", color="blue", linestyle="--"),),
                twin_series=(SeriesSpec(train_clean, "train clean", color="blue"),),
                ylabel="Train noisy error",
                twin_ylabel="Train clean error",
                xlabel="epoch",
                xscale="log",
            ),
        ]
    elif layout == "single-twin":
        panels = [
            PanelConfig(
                series=(
                    SeriesSpec(test_error, "test", color="red"),
                    SeriesSpec(train_error, "train", color="blue"),
                    SeriesSpec(train_noisy, "train noisy", color="blue", linestyle="--"),
                ),
                twin_series=(SeriesSpec(train_clean, "train clean", color="green"),),
                ylabel="Error",
                twin_ylabel="Train clean error",
                xlabel="epoch",
                xscale="log",
            )
        ]
    else:  # pragma: no cover - validated by argparse choices
        raise ValueError(f"Unsupported layout: {layout}")

    return df_plot, panels


def _common_plot_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-base", required=True, help="Base path (without extension) for saved figures")
    parser.add_argument("--layout", choices=list(_DEFAULT_FIGSIZE.keys()), default="triple")
    parser.add_argument("--vertical-lines", type=float, nargs="*", default=[], help="Vertical guide lines")
    parser.add_argument("--font-family", default=None, help="Matplotlib font family")
    parser.add_argument("--font-size", type=int, default=None, help="Base font size")
    parser.add_argument("--scale-errors", type=float, default=100.0, help="Scale factor for error columns (set to 1 to disable)")
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        help="Override figure size",
    )
    parser.add_argument(
        "--output-formats",
        nargs="+",
        default=["svg", "pdf"],
        help="File formats to export (e.g., svg pdf png)",
    )
    parser.add_argument("--x-column", default="epoch", help="Column to use for the x-axis")


def _handle_plot_csv(args: argparse.Namespace) -> None:
    df = load_training_metrics_from_csv(args.csv)
    scale = None if args.scale_errors == 1 else args.scale_errors
    df_plot, panels = _build_default_layout(df, args.layout, scale)
    figure_size = tuple(args.figsize) if args.figsize else _DEFAULT_FIGSIZE[args.layout]
    saved = plot_panels(
        df_plot,
        panels,
        args.output_base,
        x=args.x_column,
        vertical_lines=args.vertical_lines,
        figure_size=figure_size,
        font_family=args.font_family,
        font_size=args.font_size,
        output_formats=args.output_formats,
    )
    print("Saved:")
    for path in saved:
        print(f"  {path}")


def _handle_plot_wandb(args: argparse.Namespace) -> None:
    metrics = args.metrics if args.metrics else None
    df = load_training_metrics_from_wandb(
        args.entity,
        args.project,
        args.run_name,
        metrics=metrics,
    )
    scale = None if args.scale_errors == 1 else args.scale_errors
    df_plot, panels = _build_default_layout(df, args.layout, scale)
    figure_size = tuple(args.figsize) if args.figsize else _DEFAULT_FIGSIZE[args.layout]
    saved = plot_panels(
        df_plot,
        panels,
        args.output_base,
        x=args.x_column,
        vertical_lines=args.vertical_lines,
        figure_size=figure_size,
        font_family=args.font_family,
        font_size=args.font_size,
        output_formats=args.output_formats,
    )
    print("Saved:")
    for path in saved:
        print(f"  {path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Training curve visualization utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    csv_parser = subparsers.add_parser("plot-csv", help="Plot metrics from a local CSV file")
    csv_parser.add_argument("--csv", required=True, help="Path to training_metrics.csv")
    _common_plot_args(csv_parser)
    csv_parser.set_defaults(func=_handle_plot_csv)

    wandb_parser = subparsers.add_parser("plot-wandb", help="Plot metrics fetched from Weights & Biases")
    wandb_parser.add_argument("--entity", required=True)
    wandb_parser.add_argument("--project", required=True)
    wandb_parser.add_argument("--run-name", required=True)
    wandb_parser.add_argument("--metrics", nargs="*", help="Additional metric keys to fetch")
    _common_plot_args(wandb_parser)
    wandb_parser.set_defaults(func=_handle_plot_wandb)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

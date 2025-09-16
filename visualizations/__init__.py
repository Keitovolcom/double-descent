"""Visualization utilities consolidating training-curve and dataset helpers."""

from .training_curves import (
    PanelConfig,
    SeriesSpec,
    collect_metrics_by_width,
    load_training_metrics_from_csv,
    load_training_metrics_from_wandb,
    plot_loss_curves_by_width,
    plot_metric_vs_width,
    plot_panels,
)

from .colored_emnist import (
    ColoredEmnistDataset,
    display_colored_sample,
    load_colored_emnist_data,
)

__all__ = [
    "PanelConfig",
    "SeriesSpec",
    "collect_metrics_by_width",
    "load_training_metrics_from_csv",
    "load_training_metrics_from_wandb",
    "plot_loss_curves_by_width",
    "plot_metric_vs_width",
    "plot_panels",
    "ColoredEmnistDataset",
    "display_colored_sample",
    "load_colored_emnist_data",
]

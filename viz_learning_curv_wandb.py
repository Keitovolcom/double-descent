"""Plot learning curves by fetching metrics from Weights & Biases."""

from __future__ import annotations

from visualizations.training_curves import (
    PanelConfig,
    SeriesSpec,
    load_training_metrics_from_wandb,
    plot_panels,
)

# ===== 設定 =====
ENTITY = "dsml-kernel24"
PROJECT = "kobayashi_emnist"
RUN_NAME = (
    "save_model/emnist_digits/noise_0.2/use_mixup_True_alpha8.0_test_seed_42width64_cnn_5layers_cus_"
    "emnist_digits_variance0_combined_lr0.01_batch128_epoch1000_LabelNoiseRate0.2_Optimsgd_Momentum0.0/csv/"
    "training_metrics.csv"
)
OUTPUT_BASE = "/workspace/vizualize/ACML/emnist_learning_curve"


def main() -> None:
    df = load_training_metrics_from_wandb(
        ENTITY,
        PROJECT,
        RUN_NAME,
        metrics=(
            "epoch",
            "test_error",
            "train_error_total",
            "train_error_clean",
            "train_error_noisy",
        ),
    )

    df["test_error_rate"] = df["test_error"] / 100
    df["train_error_rate"] = df["train_error_total"] / 100
    df["train_error_noisy_rate"] = df["train_error_noisy"] / 100
    df["train_error_clean_rate"] = df["train_error_clean"] / 100

    panels = [
        PanelConfig(
            series=(
                SeriesSpec("test_error_rate", "Test", color="red", linewidth=3),
                SeriesSpec("train_error_rate", "Train", color="blue", linewidth=3),
            ),
            ylabel="Error (%)",
            ylim=(-0.1, 103),
            xscale="log",
            xlim=(0.9, 2000),
            hide_x_ticks=True,
            grid=True,
            tick_labelsize=30,
            legend_fontsize=30,
        ),
        PanelConfig(
            series=(
                SeriesSpec("train_error_noisy_rate", "Noisy", color="darkblue", linewidth=3),
                SeriesSpec("train_error_clean_rate", "Clean", color="cyan", linewidth=3),
            ),
            ylabel="Train Error (%)",
            ylim=(-0.01, 1.0),
            xscale="log",
            xlim=(0.9, 1000),
            xlabel="Epoch",
            grid=True,
            tick_labelsize=30,
            legend_fontsize=30,
        ),
    ]

    saved = plot_panels(
        df,
        panels,
        OUTPUT_BASE,
        figure_size=(13, 10),
        font_family="DejaVu Sans",
        font_size=20,
    )

    print(f"[\u2713] Saved: {', '.join(saved)}")


if __name__ == "__main__":
    main()

"""ACML figure configuration for EMNIST learning curves fetched from W&B."""

from __future__ import annotations

from visualizations.training_curves import (
    PanelConfig,
    SeriesSpec,
    load_training_metrics_from_wandb,
    plot_panels,
)

ENTITY = "dsml-kernel24"
PROJECT = "kobayashi_emnist"
RUN_NAME = "cnn_5layers_width8_emnist_digits_lr0.01_batch_size128_epoch1000_LabelNoiseRate0.2_Optimsgd_momentum0.0"
OUTPUT_BASE = "/workspace/vizualize/ACML/emnist_learning_curve2"
VERTICAL_EPOCHS = [1, 27, 55, 120]


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
                SeriesSpec("test_error_rate", "test", color="red", linewidth=5),
                SeriesSpec("train_error_rate", "train", color="blue", linewidth=5),
            ),
            ylabel="error",
            yticks=[0.0, 0.1, 0.2, 0.3],
            ylim=(-0.02, 0.32),
            xscale="log",
            xlim=(0.9, 1000),
            hide_x_ticks=True,
            tick_labelsize=45,
            legend_fontsize=45,
        ),
        PanelConfig(
            series=(
                SeriesSpec(
                    "train_error_noisy_rate",
                    "train noisy",
                    color="blue",
                    linewidth=5,
                    linestyle="--",
                ),
            ),
            twin_series=(
                SeriesSpec("train_error_clean_rate", "train clean", color="blue", linewidth=5),
            ),
            ylabel="train noisy error",
            yticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            ylim=(-0.02, 1.02),
            twin_ylabel="train clean error",
            twin_ylim=(-0.00235, 0.11),
            twin_yticks=[0.0, 0.02, 0.04, 0.06, 0.08, 0.1],
            xscale="log",
            xlim=(0.9, 1000),
            xlabel="epoch",
            tick_labelsize=45,
            twin_tick_labelsize=45,
            legend_fontsize=45,
        ),
    ]

    saved = plot_panels(
        df,
        panels,
        OUTPUT_BASE,
        vertical_lines=VERTICAL_EPOCHS,
        figure_size=(16, 16),
        font_family="Times New Roman",
        font_size=30,
    )

    print(f"[\u2713] Saved: {', '.join(saved)}")


if __name__ == "__main__":
    main()

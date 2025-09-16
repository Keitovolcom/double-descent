"""Plot EMNIST learning curves from a saved CSV file."""

from __future__ import annotations

from pathlib import Path

from visualizations.training_curves import (
    PanelConfig,
    SeriesSpec,
    load_training_metrics_from_csv,
    plot_panels,
)

# !!!注意!!! 以下のファイルパスを、環境に合わせて修正してください
FILE_PATH = Path(
    "save_model/emnist_digits/noise_0.2/"
    "7_14_use_mixup_False_alpha0.0_test_seed_42width64_cnn_5layers_cus_"
    "emnist_digits_variance0_combined_lr0.01_batch128_epoch1000_"
    "LabelNoiseRate0.2_Optimsgd_Momentum0.0/csv/training_metrics.csv"
)
OUTPUT_BASE = Path("vizualize/7_19/clean_noise_02_emnist_learning_curve")
VERTICAL_EPOCHS = [30, 53, 140]


def main() -> None:
    df = load_training_metrics_from_csv(FILE_PATH)

    df["test_error_rate"] = df["test_error"] / 100
    df["train_error_rate"] = df["train_error_total"] / 100
    df["train_error_noisy_rate"] = df["train_error_noisy"] / 100
    df["train_error_clean_rate"] = df["train_error_clean"] / 100

    panels = [
        PanelConfig(
            series=(
                SeriesSpec("test_error_rate", "test", color="red", linewidth=3),
                SeriesSpec("train_error_rate", "train", color="blue", linewidth=3),
            ),
            ylabel="error",
            ylim=(-0.01, 0.41),
            xscale="log",
            xlim=(1, 1000),
            hide_x_ticks=True,
            tick_labelsize=30,
            legend_fontsize=30,
        ),
        PanelConfig(
            series=(
                SeriesSpec(
                    "train_error_noisy_rate",
                    "noisy",
                    color="purple",
                    linestyle="--",
                    linewidth=3,
                ),
            ),
            twin_series=(
                SeriesSpec("train_error_clean_rate", "clean", color="green", linewidth=3),
            ),
            ylabel="train noisy error",
            yticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            ylim=(-0.05, 1.01),
            twin_ylabel="train clean error",
            twin_ylim=(-0.01, 0.151),
            xscale="log",
            xlim=(1, 1000),
            hide_x_ticks=True,
            tick_labelsize=30,
            twin_tick_labelsize=30,
            legend_fontsize=30,
        ),
        PanelConfig(
            series=(
                SeriesSpec("test_loss", "noisy loss", color="red", linewidth=3),
                SeriesSpec("train_loss", "clean loss", color="blue", linewidth=3),
            ),
            ylabel="loss",
            xlabel="epoch",
            xscale="log",
            xlim=(1, 1000),
            tick_labelsize=30,
            legend_fontsize=30,
        ),
    ]

    saved = plot_panels(
        df,
        panels,
        OUTPUT_BASE,
        vertical_lines=VERTICAL_EPOCHS,
        figure_size=(11, 19),
        font_family="DejaVu Sans",
        font_size=20,
    )

    print(f"[\u2713] Saved: {', '.join(saved)}")


if __name__ == "__main__":
    main()

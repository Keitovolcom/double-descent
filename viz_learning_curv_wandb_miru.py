"""Single-panel MIRU visualization of EMNIST learning curves via W&B."""

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
OUTPUT_BASE = "/workspace/vizualize/miru_oral/ratio_single"
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
            "train_loss",
            "test_loss",
        ),
    )

    df["test_error_rate"] = df["test_error"] / 100
    df["train_error_rate"] = df["train_error_total"] / 100
    df["train_error_noisy_rate"] = df["train_error_noisy"] / 100
    df["train_error_clean_rate"] = df["train_error_clean"] / 100

    panel = PanelConfig(
        series=(
            SeriesSpec("test_error_rate", "test", color="red", linewidth=3),
            SeriesSpec("train_error_rate", "train", color="blue", linewidth=3),
            SeriesSpec(
                "train_error_noisy_rate",
                "train noisy",
                color="blue",
                linewidth=3,
                linestyle="--",
            ),
        ),
        twin_series=(
            SeriesSpec("train_error_clean_rate", "train clean", color="green", linewidth=3),
        ),
        ylabel="error",
        twin_ylabel="train clean error",
        ylim=(-0.02, 1.02),
        twin_ylim=(-0.00235, 0.11),
        xscale="log",
        xlim=(0.9, 1000),
        xlabel="epoch",
        tick_labelsize=30,
        twin_tick_labelsize=30,
        legend_fontsize=30,
    )

    saved = plot_panels(
        df,
        [panel],
        OUTPUT_BASE,
        vertical_lines=VERTICAL_EPOCHS,
        figure_size=(10, 8),
        font_family="Times New Roman",
        font_size=20,
    )

    print(f"[\u2713] Saved: {', '.join(saved)}")


if __name__ == "__main__":
    main()

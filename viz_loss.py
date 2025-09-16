"""Generate per-width loss curves for saved CIFAR-10 models."""

from __future__ import annotations

from pathlib import Path

from visualizations.training_curves import plot_loss_curves_by_width

ROOT_DIR = Path("save_model/cifar10/noise_0.2/")
OUTPUT_DIR = Path("./fig_loss")
PATTERN = r"seed_42width(\d+)_resnet18k_cifar10"


def main() -> None:
    plot_loss_curves_by_width(
        ROOT_DIR,
        OUTPUT_DIR,
        metrics=("avg_loss_noisy", "avg_loss_clean"),
        pattern=PATTERN,
        xlabel="Epoch (log scale)",
        ylabel="Average Loss",
        ylim=(-0.01, 0.14),
    )
    print("保存完了しました。")


if __name__ == "__main__":
    main()

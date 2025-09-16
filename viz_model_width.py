"""Plot CIFAR-10 evaluation error across model widths."""

from __future__ import annotations

from pathlib import Path

from visualizations.training_curves import collect_metrics_by_width, plot_metric_vs_width

ROOT_DIR = Path("/workspace/save_model/cifar10/noise_0.2")
TARGET_EPOCHS = [4000, 2000]
OUTPUT_PATH = Path("train_error_vs_width.png")


def main() -> None:
    metrics = ["test_error"]
    df = collect_metrics_by_width(ROOT_DIR, TARGET_EPOCHS, metrics=metrics)
    if df.empty:
        print("No matching training metrics were found.")
        return

    plot_metric_vs_width(
        df,
        metric="test_error",
        output_path=OUTPUT_PATH,
        xlabel="Model Width",
        ylabel="Test Error",
        figure_size=(8, 5),
    )
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()

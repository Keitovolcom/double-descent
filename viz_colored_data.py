"""Display a Colored EMNIST sample using the shared visualization utilities."""

from __future__ import annotations

from visualizations.colored_emnist import display_colored_sample, load_colored_emnist_data


def main() -> None:
    seed = 42
    variance = 1000
    correlation = 0.5
    dataset = load_colored_emnist_data(seed=seed, variance=variance, correlation=correlation)

    display_index = 2745
    save_path = "your_path.png"
    display_colored_sample(dataset, display_index, output_path=save_path)


if __name__ == "__main__":
    main()

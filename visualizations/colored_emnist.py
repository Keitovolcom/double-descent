"""Helpers for working with Colored EMNIST visualization utilities."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class ColoredEmnistDataset:
    """Container holding the Colored EMNIST training split."""

    images: np.ndarray
    digit_labels: np.ndarray
    color_labels: np.ndarray
    combined_labels: np.ndarray

    def __len__(self) -> int:  # pragma: no cover - trivial
        return int(self.images.shape[0])

    def sample(self, index: int) -> Tuple[np.ndarray, int, int, int]:
        """Return the ``(image, digit, color, combined)`` tuple at ``index``."""

        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} is out of bounds for dataset of size {len(self)}")
        image = self.images[index]
        digit_label = int(self.digit_labels[index])
        color_label = int(self.color_labels[index])
        combined_label = int(self.combined_labels[index])
        return image, digit_label, color_label, combined_label


def _resolve_base_path(
    *,
    base_path: Optional[str | Path],
    base_dir: str | Path,
    seed: Optional[int],
    variance: Optional[int | str],
    correlation: Optional[float | str],
) -> Path:
    if base_path is not None:
        return Path(base_path)
    if seed is None or variance is None or correlation is None:
        raise ValueError("Provide base_path or the trio of seed, variance, and correlation.")
    return Path(base_dir) / f"distribution_colored_EMNIST_Seed{seed}_Var{variance}_Corr{correlation}"


def load_colored_emnist_data(
    *,
    seed: Optional[int] = None,
    variance: Optional[int | str] = None,
    correlation: Optional[float | str] = None,
    base_dir: str | Path = "data",
    base_path: Optional[str | Path] = None,
) -> ColoredEmnistDataset:
    """Load Colored EMNIST arrays from disk."""

    resolved = _resolve_base_path(
        base_path=base_path,
        base_dir=base_dir,
        seed=seed,
        variance=variance,
        correlation=correlation,
    )

    arrays = {}
    for name in ("x_train_colored", "y_train_digits", "y_train_colors", "y_train_combined"):
        path = resolved / f"{name}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Expected file not found: {path}")
        arrays[name] = np.load(path)

    return ColoredEmnistDataset(
        images=arrays["x_train_colored"],
        digit_labels=arrays["y_train_digits"],
        color_labels=arrays["y_train_colors"],
        combined_labels=arrays["y_train_combined"],
    )


def display_colored_sample(
    dataset: ColoredEmnistDataset,
    index: int,
    *,
    output_path: Optional[str | Path] = None,
    show: bool = True,
    dpi: int = 300,
    caption_fontsize: int = 12,
) -> Optional[str]:
    """Display a Colored EMNIST sample and optionally save it to disk."""

    image, digit_label, color_label, combined_label = dataset.sample(index)

    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.axis("off")

    caption = (
        f"Digit Label: {digit_label}, Color Label: {color_label}, "
        f"Combined Label: {combined_label}"
    )
    fig.figtext(0.5, 0.01, caption, wrap=True, ha="center", fontsize=caption_fontsize)

    saved_path: Optional[str] = None
    if output_path is not None:
        output_path = str(output_path)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        saved_path = output_path

    if show:
        plt.show()
    plt.close(fig)
    return saved_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Colored EMNIST visualization utilities")
    parser.add_argument("--seed", type=int, help="Dataset seed")
    parser.add_argument("--variance", help="Dataset variance identifier")
    parser.add_argument("--correlation", help="Dataset correlation identifier")
    parser.add_argument("--base-dir", default="data", help="Root directory containing dataset folders")
    parser.add_argument(
        "--base-path",
        help="Direct path to the dataset directory (overrides seed/variance/correlation)",
    )
    parser.add_argument("--index", type=int, default=0, help="Index of the sample to display")
    parser.add_argument("--output", help="Optional file path to save the rendered sample")
    parser.add_argument("--no-show", action="store_true", help="Do not open a window; only save the file")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.base_path is None and (args.seed is None or args.variance is None or args.correlation is None):
        parser.error("Either --base-path or all of --seed/--variance/--correlation must be provided.")

    dataset = load_colored_emnist_data(
        seed=args.seed,
        variance=args.variance,
        correlation=args.correlation,
        base_dir=args.base_dir,
        base_path=args.base_path,
    )
    display_colored_sample(dataset, args.index, output_path=args.output, show=not args.no_show)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

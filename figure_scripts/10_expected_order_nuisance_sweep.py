#!/usr/bin/env python3
"""Generate figures/10_expected_order_nuisance_sweep.png."""

from __future__ import annotations

from collections import defaultdict

from common import METHOD_COLORS, REPO_ROOT, SCENE_COLORS, build_plot_parser, plt, read_csv_rows, save_figure, trim_figure_whitespace


def make_figure(output_path) -> None:
    scene_paths = {
        "open_aisle": REPO_ROOT / "results" / "submission_expected_order" / "open_aisle" / "data" / "nuisance_gain_offset.csv",
        "intersection": REPO_ROOT / "results" / "submission_expected_order" / "intersection" / "data" / "nuisance_gain_offset.csv",
        "rack_aisle": REPO_ROOT / "results" / "submission_expected_order" / "rack_aisle" / "data" / "nuisance_gain_offset.csv",
    }
    scene_labels = {"open_aisle": "Open aisle", "intersection": "Intersection", "rack_aisle": "Rack aisle"}

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.9), sharey=True)
    for ax, scene in zip(axes, ("open_aisle", "intersection", "rack_aisle"), strict=True):
        rows = read_csv_rows(scene_paths[scene])
        if not rows:
            raise SystemExit(f"Missing nuisance sweep CSV: {scene_paths[scene]}")
        grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for row in rows:
            grouped[row["method"]].append(
                (float(row["parameter_numeric_value"]), float(row["joint_resolution_probability"]))
            )
        for method in ("fft_masked", "music_masked"):
            series = sorted(grouped[method], key=lambda item: item[0])
            ax.plot(
                [item[0] for item in series],
                [item[1] for item in series],
                marker="o" if method == "fft_masked" else "s",
                linewidth=2.2,
                color=METHOD_COLORS[method],
                label="FFT" if method == "fft_masked" else "Expected-order MUSIC",
            )
        ax.set_title(scene_labels[scene], fontsize=12, fontweight="bold", color=SCENE_COLORS[scene])
        ax.set_xlabel("Uniform nuisance gain offset (dB)")
        ax.grid(True, alpha=0.22)
    axes[0].set_ylabel("Joint-resolution probability")
    axes[0].set_ylim(-0.03, 1.03)
    axes[1].legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=2)
    fig.suptitle("Expected-order nuisance-strength sweep", fontsize=14, fontweight="bold", x=0.08, ha="left")
    fig.tight_layout()
    save_figure(fig, output_path, bbox_inches="tight")
    trim_figure_whitespace(output_path)


def main() -> None:
    parser = build_plot_parser(
        "Generate the expected-order nuisance sweep figure.",
        "10_expected_order_nuisance_sweep.png",
    )
    args = parser.parse_args()
    make_figure(args.output)


if __name__ == "__main__":
    main()

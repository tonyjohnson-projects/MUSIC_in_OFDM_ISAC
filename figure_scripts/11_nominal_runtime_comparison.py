#!/usr/bin/env python3
"""Generate figures/11_nominal_runtime_comparison.png."""

from __future__ import annotations

from collections import defaultdict

from common import METHOD_COLORS, REPO_ROOT, build_plot_parser, np, plt, read_csv_rows, resolve_data_dir, save_figure, trim_figure_whitespace


def make_figure(data_dir, output_path) -> None:
    rows = read_csv_rows(data_dir / "runtime_summary.csv")
    if not rows:
        raise SystemExit(f"Missing or empty runtime summary: {data_dir / 'runtime_summary.csv'}")

    scene_order = ("intersection", "open_aisle", "rack_aisle")
    scene_labels = {"intersection": "Intersection", "open_aisle": "Open aisle", "rack_aisle": "Rack aisle"}
    by_scene: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        by_scene[row["scene_class"]][row["method"]] = float(row["total_runtime_s"])

    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    y_positions = np.arange(len(scene_order), dtype=float)
    offset = 0.17
    fft_values = [by_scene[scene]["fft_masked"] for scene in scene_order]
    music_values = [by_scene[scene]["music_masked"] for scene in scene_order]
    ax.barh(y_positions - offset, fft_values, height=0.28, color=METHOD_COLORS["fft_masked"], label="FFT")
    ax.barh(y_positions + offset, music_values, height=0.28, color=METHOD_COLORS["music_masked"], label="MUSIC")
    for index, _scene in enumerate(scene_order):
        ax.text(fft_values[index] + 0.02, y_positions[index] - offset, f"{fft_values[index]:.3f} s", va="center", fontsize=9)
        ax.text(music_values[index] + 0.02, y_positions[index] + offset, f"{music_values[index]:.3f} s", va="center", fontsize=9)
    ax.set_yticks(y_positions, [scene_labels[scene] for scene in scene_order])
    ax.set_xlim(0, max(max(fft_values), max(music_values)) * 1.15)
    ax.set_xlabel("Total runtime per nominal point")
    ax.set_title("Nominal runtime comparison", loc="left", fontsize=14, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.2)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    ax.invert_yaxis()
    fig.tight_layout()
    save_figure(fig, output_path)
    trim_figure_whitespace(output_path)


def main() -> None:
    parser = build_plot_parser(
        "Generate the nominal runtime comparison figure.",
        "11_nominal_runtime_comparison.png",
        default_input_root=REPO_ROOT / "results" / "submission",
    )
    args = parser.parse_args()
    make_figure(resolve_data_dir(args.input_root), args.output)


if __name__ == "__main__":
    main()

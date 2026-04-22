#!/usr/bin/env python3
"""Generate figures/09_model_order_nominal_comparison.png."""

from __future__ import annotations

from collections import defaultdict

from common import METHOD_COLORS, REPO_ROOT, SCENE_COLORS, build_plot_parser, plt, read_csv_rows, save_figure, trim_figure_whitespace, np


def make_figure(output_path) -> None:
    mdl_rows = read_csv_rows(REPO_ROOT / "results" / "analysis" / "model_order_nominal_64trials.csv")
    eigengap_rows = read_csv_rows(REPO_ROOT / "results" / "analysis" / "model_order_nominal_64trials_eigengap.csv")
    if not mdl_rows or not eigengap_rows:
        raise SystemExit("Missing model-order comparison CSVs in results/analysis")

    records: dict[str, dict[str, float]] = defaultdict(dict)
    for rows in (mdl_rows, eigengap_rows):
        for row in rows:
            scene = row["scene"]
            mode = row["music_model_order_mode"]
            records[scene][mode] = float(row["music_joint_pres"])
            records[scene]["fft"] = float(row["fft_joint_pres"])

    scene_order = ("intersection", "open_aisle", "rack_aisle")
    scene_labels = {"intersection": "Intersection", "open_aisle": "Open aisle", "rack_aisle": "Rack aisle"}
    method_order = ("fft", "mdl", "eigengap", "expected")
    method_labels = {"fft": "FFT", "mdl": "MDL MUSIC", "eigengap": "Eigengap MUSIC", "expected": "Expected-order MUSIC"}
    method_colors = {"fft": METHOD_COLORS["fft_masked"], "mdl": "#7F7F7F", "eigengap": "#009E73", "expected": METHOD_COLORS["music_masked"]}
    method_markers = {"fft": "o", "mdl": "D", "eigengap": "^", "expected": "s"}

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    y_positions = np.arange(len(scene_order), dtype=float)
    offsets = np.linspace(-0.24, 0.24, len(method_order))
    for scene_index, scene in enumerate(scene_order):
        values = [records[scene][method] for method in method_order]
        ax.hlines(y_positions[scene_index], min(values), max(values), color="#D7D7D7", linewidth=2.5)
        for offset, method in zip(offsets, method_order, strict=True):
            value = records[scene][method]
            ax.scatter(
                value,
                y_positions[scene_index] + offset,
                s=90,
                color=method_colors[method],
                marker=method_markers[method],
                edgecolors="white",
                linewidths=1.1,
                label=method_labels[method] if scene_index == 0 else None,
                zorder=3,
            )
            ax.text(value + 0.015, y_positions[scene_index] + offset, f"{value:.3f}", va="center", fontsize=9)
        ax.text(-0.02, y_positions[scene_index], scene_labels[scene], ha="right", va="center", fontsize=11, fontweight="bold", color=SCENE_COLORS[scene])

    ax.set_xlim(0.0, 1.08)
    ax.set_ylim(-0.6, len(scene_order) - 0.4)
    ax.set_yticks([])
    ax.set_xlabel("Nominal joint-resolution probability")
    ax.set_title("Model-order diagnosis at the nominal point", loc="left", fontsize=14, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.2)
    ax.legend(frameon=False, loc="lower right", ncol=2)
    ax.invert_yaxis()
    fig.tight_layout()
    save_figure(fig, output_path)
    trim_figure_whitespace(output_path)


def main() -> None:
    parser = build_plot_parser(
        "Generate the model-order comparison figure.",
        "09_model_order_nominal_comparison.png",
    )
    args = parser.parse_args()
    make_figure(args.output)


if __name__ == "__main__":
    main()

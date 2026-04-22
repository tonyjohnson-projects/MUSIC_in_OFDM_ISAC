#!/usr/bin/env python3
"""Generate figures/02_nominal_scene_verdict.png."""

from __future__ import annotations

from common import (
    METHOD_COLORS,
    METHOD_LABELS,
    SCENE_COLORS,
    REPO_ROOT,
    build_plot_parser,
    nominal_joint_deltas,
    np,
    plt,
    read_csv_rows,
    resolve_data_dir,
    save_figure,
    scene_key,
    scene_label_from_rows,
)


def make_figure(data_dir, output_path) -> None:
    rows = read_csv_rows(data_dir / "nominal_summary.csv")
    if not rows:
        raise SystemExit(f"Missing or empty nominal summary: {data_dir / 'nominal_summary.csv'}")

    scene_classes = sorted({row["scene_class"] for row in rows}, key=scene_key)
    scene_deltas = nominal_joint_deltas(rows)
    scene_classes.sort(
        key=lambda scene_class: float(
            next(
                row["joint_resolution_probability"]
                for row in rows
                if row["scene_class"] == scene_class and row["method"] == "music_masked"
            )
        )
        - float(
            next(
                row["joint_resolution_probability"]
                for row in rows
                if row["scene_class"] == scene_class and row["method"] == "fft_masked"
            )
        ),
        reverse=True,
    )

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    y_positions = np.arange(len(scene_classes), dtype=float)
    for scene_index, scene_class in enumerate(scene_classes):
        fft_row = next(
            row
            for row in rows
            if row["scene_class"] == scene_class and row["method"] == "fft_masked"
        )
        music_row = next(
            row
            for row in rows
            if row["scene_class"] == scene_class and row["method"] == "music_masked"
        )
        fft_value = float(fft_row["joint_resolution_probability"])
        music_value = float(music_row["joint_resolution_probability"])
        fft_ci = (
            float(fft_row["joint_resolution_probability_ci95_lower"]),
            float(fft_row["joint_resolution_probability_ci95_upper"]),
        )
        music_ci = (
            float(music_row["joint_resolution_probability_ci95_lower"]),
            float(music_row["joint_resolution_probability_ci95_upper"]),
        )
        y_value = y_positions[scene_index]
        ax.hlines(y_value, min(fft_value, music_value), max(fft_value, music_value), color="#B8B8B8", linewidth=3.0)
        ax.errorbar(
            fft_value,
            y_value,
            xerr=[[fft_value - fft_ci[0]], [fft_ci[1] - fft_value]],
            fmt="o",
            markersize=9,
            color=METHOD_COLORS["fft_masked"],
            # FIX: use markerfacecolor/markeredgecolor kwargs instead of mfc/mec
            # so matplotlib doesn't add a Line2D background patch in the legend
            markerfacecolor=METHOD_COLORS["fft_masked"],
            markeredgecolor="white",
            markeredgewidth=1.2,
            ecolor=METHOD_COLORS["fft_masked"],
            capsize=4,
            linewidth=2.2,
            label=METHOD_LABELS["fft_masked"] if scene_index == 0 else None,
        )
        ax.errorbar(
            music_value,
            y_value,
            xerr=[[music_value - music_ci[0]], [music_ci[1] - music_value]],
            fmt="s",
            markersize=9,
            color=METHOD_COLORS["music_masked"],
            markerfacecolor=METHOD_COLORS["music_masked"],
            markeredgecolor="white",
            markeredgewidth=1.2,
            ecolor=METHOD_COLORS["music_masked"],
            capsize=4,
            linewidth=2.2,
            label=METHOD_LABELS["music_masked"] if scene_index == 0 else None,
        )

        delta = music_value - fft_value
        if delta >= 0.05:
            delta_label = f"MUSIC +{delta:.2f}"
            delta_color = METHOD_COLORS["music_masked"]
        elif delta <= -0.05:
            delta_label = f"FFT +{abs(delta):.2f}"
            delta_color = METHOD_COLORS["fft_masked"]
        else:
            delta_label = f"Tie ({delta:+.2f})"
            delta_color = "#444444"

        # FIX: anchor delta labels on the right edge of the axes (x=1.0 in axes
        # coords) so they never overflow the figure boundary.
        ax.text(
            0.99,
            y_value,
            delta_label,
            va="center",
            ha="right",
            fontsize=10,
            color=delta_color,
            fontweight="bold",
            transform=ax.get_yaxis_transform(),  # x in axes coords, y in data coords
        )
        ax.text(
            -0.02,
            y_value,
            scene_label_from_rows(rows, scene_class),
            va="center",
            ha="right",
            fontsize=11,
            color=SCENE_COLORS.get(scene_class, "#333333"),
            fontweight="bold",
        )

    ax.axvline(0.5, color="#D8D8D8", linestyle="--", linewidth=1.0)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.7, len(scene_classes) - 0.3)
    ax.set_yticks([])
    ax.set_xlabel("Nominal Joint-Resolution Probability")

    # FIX: simplified title — single concise line
    ax.set_title("Nominal scene verdicts (64-trial FR1, 95% Wilson intervals)", loc="left", fontsize=14, fontweight="bold")

    ax.grid(True, axis="x", alpha=0.20)

    # FIX: suppress the error-bar line handles so only the marker glyphs appear
    # in the legend, eliminating the background-line artifact.
    handles, labels = ax.get_legend_handles_labels()
    clean_handles = [h[0] if hasattr(h, "__getitem__") else h for h in handles]
    ax.legend(clean_handles, labels, frameon=False, loc="lower right")

    ax.invert_yaxis()
    fig.tight_layout()
    save_figure(fig, output_path)


def main() -> None:
    parser = build_plot_parser(
        "Generate the nominal scene verdict figure.",
        "02_nominal_scene_verdict.png",
        default_input_root=REPO_ROOT / "results" / "submission",
    )
    args = parser.parse_args()
    make_figure(resolve_data_dir(args.input_root), args.output)


if __name__ == "__main__":
    main()
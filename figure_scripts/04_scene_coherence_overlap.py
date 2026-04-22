#!/usr/bin/env python3
"""Generate figures/04_scene_coherence_overlap.png."""

from __future__ import annotations

from common import (
    SCENE_COLORS,
    REPO_ROOT,
    build_plot_parser,
    nominal_headline_rows,
    np,
    plt,
    read_csv_rows,
    resolve_data_dir,
    save_figure,
    scene_key,
    scene_label_from_rows,
)


def make_figure(data_dir, output_path) -> None:
    rows = read_csv_rows(data_dir / "trial_level_results.csv")
    nominal_rows = nominal_headline_rows(rows, method_name="fft_masked")
    if not nominal_rows:
        raise SystemExit(f"Missing nominal trial rows: {data_dir / 'trial_level_results.csv'}")

    scene_classes = sorted({row["scene_class"] for row in nominal_rows}, key=scene_key)
    empirical_by_scene = [
        np.asarray(
            [
                float(row["empirical_target_coherence"])
                for row in nominal_rows
                if row["scene_class"] == scene_class
            ],
            dtype=float,
        )
        for scene_class in scene_classes
    ]
    configured_by_scene = {
        scene_class: float(
            next(
                row["configured_target_coherence"]
                for row in nominal_rows
                if row["scene_class"] == scene_class
            )
        )
        for scene_class in scene_classes
    }

    positions = np.arange(len(scene_classes), dtype=float)
    fig, ax = plt.subplots(figsize=(10.6, 5.0))
    box = ax.boxplot(
        empirical_by_scene,
        positions=positions,
        widths=0.52,
        patch_artist=True,
        showfliers=False,
    )
    for patch, scene_class in zip(box["boxes"], scene_classes, strict=True):
        patch.set_facecolor(SCENE_COLORS.get(scene_class, "#777777"))
        patch.set_alpha(0.26)
        patch.set_edgecolor(SCENE_COLORS.get(scene_class, "#777777"))
    for median in box["medians"]:
        median.set_color("#222222")
        median.set_linewidth(1.6)

    rng = np.random.default_rng(20260406)
    for position, scene_class, values in zip(positions, scene_classes, empirical_by_scene, strict=True):
        jitter = rng.uniform(-0.14, 0.14, size=values.size)
        ax.scatter(
            np.full(values.size, position) + jitter,
            values,
            s=20,
            alpha=0.55,
            color=SCENE_COLORS.get(scene_class, "#777777"),
            edgecolors="none",
        )
        configured_value = configured_by_scene[scene_class]
        ax.scatter(position, configured_value, marker="D", s=54, color="#111111", zorder=4)
        ax.text(position, 1.04, f"mean {np.mean(values):.2f}", ha="center", va="top", fontsize=9, fontweight="bold")

    ax.set_xticks(positions)
    ax.set_xticklabels([scene_label_from_rows(nominal_rows, scene_class) for scene_class in scene_classes])
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Target Coherence")
    ax.set_title(
        "Empirical nominal coherence overlaps strongly across scenes\nConfigured coherence alone does not isolate the operating regimes",
        loc="left",
        fontsize=14,
        fontweight="bold",
    )
    ax.text(
        0.01,
        0.91,
        "Circles = empirical finite-snapshot coherence per nominal trial; diamond = configured scene coherence",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )
    ax.grid(True, axis="y", alpha=0.20)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.12, top=0.84)
    save_figure(fig, output_path)


def main() -> None:
    parser = build_plot_parser(
        "Generate the scene coherence-overlap figure.",
        "04_scene_coherence_overlap.png",
        default_input_root=REPO_ROOT / "results" / "submission",
    )
    args = parser.parse_args()
    make_figure(resolve_data_dir(args.input_root), args.output)


if __name__ == "__main__":
    main()

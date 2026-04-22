#!/usr/bin/env python3
"""Generate figures/03_nominal_trial_delta.png."""

from __future__ import annotations

from collections import defaultdict

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
    nominal_rows = nominal_headline_rows(rows)
    if not nominal_rows:
        raise SystemExit(f"Missing nominal trial rows: {data_dir / 'trial_level_results.csv'}")

    paired_by_scene: dict[str, list[float]] = defaultdict(list)
    for scene_class in {row["scene_class"] for row in nominal_rows}:
        fft_by_trial = {
            row["trial_index"]: float(row["unconditional_joint_assignment_rmse"])
            for row in nominal_rows
            if row["scene_class"] == scene_class and row["method"] == "fft_masked"
        }
        music_by_trial = {
            row["trial_index"]: float(row["unconditional_joint_assignment_rmse"])
            for row in nominal_rows
            if row["scene_class"] == scene_class and row["method"] == "music_masked"
        }
        common_trials = sorted(set(fft_by_trial) & set(music_by_trial), key=int)
        paired_by_scene[scene_class] = [
            fft_by_trial[trial_index] - music_by_trial[trial_index]
            for trial_index in common_trials
        ]

    scene_classes = sorted(paired_by_scene, key=scene_key)
    fig, ax = plt.subplots(figsize=(10.0, 5.4))
    ax.axhspan(0.0, max(max(values) for values in paired_by_scene.values()) + 1.0e-9, color="#EAF2FB", alpha=0.65)
    ax.axhspan(min(min(values) for values in paired_by_scene.values()) - 1.0e-9, 0.0, color="#FBECEC", alpha=0.65)
    ax.axhline(0.0, color="#333333", linewidth=1.2)

    positions = np.arange(len(scene_classes), dtype=float)
    box = ax.boxplot(
        [paired_by_scene[scene_class] for scene_class in scene_classes],
        positions=positions,
        widths=0.52,
        patch_artist=True,
        showfliers=False,
    )
    for patch, scene_class in zip(box["boxes"], scene_classes, strict=True):
        patch.set_facecolor(SCENE_COLORS.get(scene_class, "#777777"))
        patch.set_alpha(0.28)
        patch.set_edgecolor(SCENE_COLORS.get(scene_class, "#777777"))
    for median in box["medians"]:
        median.set_color("#222222")
        median.set_linewidth(1.6)

    rng = np.random.default_rng(20260404)
    for position, scene_class in zip(positions, scene_classes, strict=True):
        values = np.asarray(paired_by_scene[scene_class], dtype=float)
        jitter = rng.uniform(-0.14, 0.14, size=values.size)
        ax.scatter(
            np.full(values.size, position) + jitter,
            values,
            s=22,
            alpha=0.60,
            color=SCENE_COLORS.get(scene_class, "#777777"),
            edgecolors="none",
        )
        win_fraction = float(np.mean(values > 0.0))
        ax.text(position, np.percentile(values, 85) + 0.02, f"{win_fraction:.0%} trials favor MUSIC", ha="center", fontsize=9)

    ax.set_xticks(positions)
    ax.set_xticklabels([scene_label_from_rows(nominal_rows, scene_class) for scene_class in scene_classes])
    ax.set_ylabel("FFT RMSE - MUSIC RMSE\n(positive means MUSIC is better)")
    win_fractions = {
        scene_class: float(np.mean(np.asarray(paired_by_scene[scene_class], dtype=float) > 0.0))
        for scene_class in scene_classes
    }
    if {"intersection", "open_aisle"}.issubset(scene_classes):
        title = "Intersection gains are broad, while open-aisle losses are systematic\nPaired nominal trial deltas at the 64-trial FR1 nominal point"
    elif len(scene_classes) == 1:
        scene_class = scene_classes[0]
        if win_fractions[scene_class] > 0.55:
            verdict = "mostly favor MUSIC"
        elif win_fractions[scene_class] < 0.45:
            verdict = "mostly favor FFT"
        else:
            verdict = "split almost evenly"
        title = (
            f"{scene_label_from_rows(nominal_rows, scene_class)} trial deltas {verdict}\n"
            "Paired nominal trials at the saved FR1 point"
        )
    else:
        title = "Paired nominal trial deltas by scene\nPositive values mean MUSIC reduces assignment RMSE"
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.20)
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.11, top=0.87)
    save_figure(fig, output_path)


def main() -> None:
    parser = build_plot_parser(
        "Generate the paired nominal trial-delta figure.",
        "03_nominal_trial_delta.png",
        default_input_root=REPO_ROOT / "results" / "submission",
    )
    args = parser.parse_args()
    make_figure(resolve_data_dir(args.input_root), args.output)


if __name__ == "__main__":
    main()

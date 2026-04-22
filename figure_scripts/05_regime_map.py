#!/usr/bin/env python3
"""Generate figures/05_regime_map.png."""

from __future__ import annotations

from common import (
    REPO_ROOT,
    SEPARATION_SWEEP_ORDER,
    STRONG_WIN_THRESHOLD,
    SUPPORT_SWEEP_ORDER,
    SWEEP_LABELS,
    TwoSlopeNorm,
    build_plot_parser,
    contrast_text_effects,
    np,
    plt,
    read_csv_rows,
    resolve_data_dir,
    save_figure,
    scene_key,
    scene_label_from_rows,
)


def make_figure(data_dir, output_path) -> None:
    rows = read_csv_rows(data_dir / "usefulness_windows.csv")
    if not rows:
        raise SystemExit(f"Missing or empty usefulness windows: {data_dir / 'usefulness_windows.csv'}")

    scene_classes = sorted({row["scene_class"] for row in rows}, key=scene_key)
    panel_specs = (
        ("Support-Limited Families", SUPPORT_SWEEP_ORDER, "Joint-resolution metric"),
        ("Axis-Isolated Separation Families", SEPARATION_SWEEP_ORDER, "Axis-specific metric"),
    )

    panel_data: list[tuple[np.ndarray, list[list[str]]]] = []
    max_abs_delta = 0.0
    for _title, sweep_names, _subtitle in panel_specs:
        matrix = np.zeros((len(scene_classes), len(sweep_names)), dtype=float)
        labels = [["" for _ in sweep_names] for _ in scene_classes]
        for row_index, scene_class in enumerate(scene_classes):
            for col_index, sweep_name in enumerate(sweep_names):
                sweep_rows = [
                    row
                    for row in rows
                    if row["scene_class"] == scene_class and row["sweep_name"] == sweep_name
                ]
                if not sweep_rows:
                    continue
                deltas = np.asarray([float(row["music_minus_fft"]) for row in sweep_rows], dtype=float)
                mean_delta = float(np.mean(deltas))
                strong_wins = int(np.sum(deltas >= STRONG_WIN_THRESHOLD))
                matrix[row_index, col_index] = mean_delta
                labels[row_index][col_index] = f"{mean_delta:+.2f}\n{strong_wins}/{len(deltas)}"
                max_abs_delta = max(max_abs_delta, float(np.max(np.abs(deltas))))
        panel_data.append((matrix, labels))

    max_abs_delta = max(max_abs_delta, 0.01)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-max_abs_delta, vmax=max_abs_delta)

    fig = plt.figure(figsize=(14.2, 6.2))
    grid = fig.add_gridspec(
        nrows=3,
        ncols=3,
        height_ratios=(0.9, 5.0, 0.7),
        width_ratios=(len(SUPPORT_SWEEP_ORDER), len(SEPARATION_SWEEP_ORDER) + 0.6, 0.28),
        hspace=0.18,
        wspace=0.28,
    )
    title_ax = fig.add_subplot(grid[0, :])
    ax_left = fig.add_subplot(grid[1, 0])
    ax_right = fig.add_subplot(grid[1, 1])
    colorbar_ax = fig.add_subplot(grid[1, 2])
    footer_ax = fig.add_subplot(grid[2, :])

    title_ax.axis("off")
    footer_ax.axis("off")
    title_ax.text(
        0.0,
        1.0,
        "MUSIC gains cluster in a few sweep families, not across the whole study\nCell color = mean headline-metric delta; text = strong-win count (>= +0.10)",
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
        transform=title_ax.transAxes,
    )
    footer_ax.text(
        0.0,
        0.35,
        "Support-limited and separation sweeps are split because they use different headline metrics.",
        ha="left",
        va="center",
        fontsize=9,
        color="#444444",
        transform=footer_ax.transAxes,
    )

    axes = (ax_left, ax_right)
    image = None
    for axis_index, (ax, panel_spec, panel_values) in enumerate(zip(axes, panel_specs, panel_data, strict=True)):
        panel_title, sweep_names, panel_subtitle = panel_spec
        matrix, labels = panel_values
        image = ax.imshow(matrix, aspect="auto", cmap="RdYlBu", norm=norm)
        ax.set_xticks(np.arange(len(sweep_names)))
        ax.set_xticklabels(
            [SWEEP_LABELS[name] for name in sweep_names],
            rotation=20,
            ha="right",
            rotation_mode="anchor",
            fontsize=8,
        )
        ax.set_yticks(np.arange(len(scene_classes)))
        if axis_index == 0:
            ax.set_yticklabels([scene_label_from_rows(rows, scene_class) for scene_class in scene_classes], fontsize=8)
        else:
            ax.set_yticklabels([])
        ax.set_title(f"{panel_title}\n{panel_subtitle}", fontsize=10.5, fontweight="bold", pad=10)
        ax.set_xticks(np.arange(-0.5, len(sweep_names), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(scene_classes), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.0)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.tick_params(length=0)

        for row_index in range(len(scene_classes)):
            for col_index in range(len(sweep_names)):
                if not labels[row_index][col_index]:
                    continue
                value = matrix[row_index, col_index]
                text_color = "white" if abs(value) >= 0.18 else "#222222"
                text = ax.text(
                    col_index,
                    row_index,
                    labels[row_index][col_index],
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=text_color,
                    fontweight="bold",
                )
                text.set_path_effects(contrast_text_effects(text_color))

    if image is not None:
        cbar = fig.colorbar(image, cax=colorbar_ax)
        cbar.set_label("Mean headline-metric delta", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    save_figure(fig, output_path, bbox_inches="tight", pad_inches=0.2)


def main() -> None:
    parser = build_plot_parser(
        "Generate the regime-map figure.",
        "05_regime_map.png",
        default_input_root=REPO_ROOT / "results" / "submission",
    )
    args = parser.parse_args()
    make_figure(resolve_data_dir(args.input_root), args.output)


if __name__ == "__main__":
    main()

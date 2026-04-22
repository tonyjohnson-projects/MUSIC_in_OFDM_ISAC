#!/usr/bin/env python3
"""Generate figures/05_regime_map.png."""

from __future__ import annotations

from collections import defaultdict

from common import METHOD_COLORS, REPO_ROOT, SWEEP_LABELS, build_plot_parser, np, plt, read_csv_rows, resolve_data_dir, save_figure


AXIS_METRICS = (
    "range_resolution_probability",
    "velocity_resolution_probability",
    "angle_resolution_probability",
)
EXCLUDED_SWEEPS = {"nominal", "nuisance_gain_offset"}


def make_figure(data_dir, output_path) -> None:
    rows = read_csv_rows(data_dir / "all_sweep_results.csv")
    if not rows:
        raise SystemExit(f"Missing or empty sweep results: {data_dir / 'all_sweep_results.csv'}")

    paired_rows: dict[tuple[str, str, str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        if row["knowledge_mode"] != "known_symbols" or row["sweep_name"] in EXCLUDED_SWEEPS:
            continue
        if row["method"] not in {"fft_masked", "music_masked"}:
            continue
        pair_key = (
            row["scene_class"],
            row["sweep_name"],
            row["parameter_value"],
            row["parameter_numeric_value"],
        )
        paired_rows[pair_key][row["method"]] = row

    sweep_slot_deltas: dict[str, list[float]] = defaultdict(list)
    for (_scene_class, sweep_name, _parameter_value, _parameter_numeric_value), methods in paired_rows.items():
        if "fft_masked" not in methods or "music_masked" not in methods:
            continue
        slot_delta = float(
            np.mean(
                [
                    float(methods["music_masked"][metric_name]) - float(methods["fft_masked"][metric_name])
                    for metric_name in AXIS_METRICS
                ]
            )
        )
        sweep_slot_deltas[sweep_name].append(slot_delta)

    if not sweep_slot_deltas:
        raise SystemExit("No paired FFT/MUSIC sweep rows were available for the sweep-family bar figure.")

    mean_deltas = {
        sweep_name: float(np.mean(slot_deltas))
        for sweep_name, slot_deltas in sweep_slot_deltas.items()
    }
    ordered_sweeps = sorted(mean_deltas, key=lambda sweep_name: abs(mean_deltas[sweep_name]), reverse=True)
    values = [mean_deltas[sweep_name] for sweep_name in ordered_sweeps]
    labels = [SWEEP_LABELS.get(sweep_name, sweep_name.replace("_", " ").title()) for sweep_name in ordered_sweeps]
    colors = [
        METHOD_COLORS["music_masked"] if value >= 0.0 else METHOD_COLORS["fft_masked"]
        for value in values
    ]

    max_abs_value = max(max(abs(value) for value in values), 0.05)
    y_limit = 1.25 * max_abs_value

    fig, ax = plt.subplots(figsize=(11.8, 4.9))
    x_positions = np.arange(len(ordered_sweeps), dtype=float)
    bars = ax.bar(x_positions, values, color=colors, width=0.72)

    ax.axhline(0.0, color="#444444", linewidth=1.1)
    ax.set_ylim(-y_limit, y_limit)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=24, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Mean MUSIC - FFT delta")
    ax.grid(True, axis="y", alpha=0.18)
    ax.set_title(
        "Average FFT/MUSIC gap by sweep family",
        loc="left",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.125,
        0.93,
        "Each bar averages the range, velocity, and angle resolution deltas over all scenes and sweep values in that family. Positive bars favor MUSIC.",
        ha="left",
        va="top",
        fontsize=9,
        color="#444444",
    )

    label_offset = 0.04 * y_limit
    for bar, value in zip(bars, values, strict=True):
        x_value = bar.get_x() + bar.get_width() / 2.0
        y_value = value + label_offset if value >= 0.0 else value - label_offset
        ax.text(
            x_value,
            y_value,
            f"{value:+.2f}",
            ha="center",
            va="bottom" if value >= 0.0 else "top",
            fontsize=9,
            fontweight="bold",
            color=bar.get_facecolor(),
        )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    save_figure(fig, output_path, bbox_inches="tight", pad_inches=0.16)


def main() -> None:
    parser = build_plot_parser(
        "Generate the sweep-family delta bar chart.",
        "05_regime_map.png",
        default_input_root=REPO_ROOT / "results" / "submission",
    )
    args = parser.parse_args()
    make_figure(resolve_data_dir(args.input_root), args.output)


if __name__ == "__main__":
    main()

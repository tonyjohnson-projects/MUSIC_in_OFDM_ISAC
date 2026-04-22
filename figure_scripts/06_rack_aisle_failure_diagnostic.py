#!/usr/bin/env python3
"""Generate figures/06_rack_aisle_failure_diagnostic.png."""

from __future__ import annotations

from matplotlib.patches import Patch

from common import METHOD_COLORS, REPO_ROOT, build_plot_parser, np, plt, read_csv_rows, resolve_data_dir, save_figure


def _cluster_centers(values: list[float], max_gap_deg: float = 1.5) -> list[float]:
    rounded = np.round(np.asarray(values, dtype=float), 3)
    unique, counts = np.unique(rounded, return_counts=True)
    centers: list[float] = []
    cluster_values = [float(unique[0])]
    cluster_weights = [int(counts[0])]
    for value, weight in zip(unique[1:], counts[1:], strict=True):
        value_f = float(value)
        if value_f - cluster_values[-1] <= max_gap_deg:
            cluster_values.append(value_f)
            cluster_weights.append(int(weight))
            continue
        centers.append(float(np.average(cluster_values, weights=cluster_weights)))
        cluster_values = [value_f]
        cluster_weights = [int(weight)]
    centers.append(float(np.average(cluster_values, weights=cluster_weights)))
    return centers


def _trial_presence_count(per_trial_values: list[list[float]], center_deg: float, tol_deg: float = 2.0) -> int:
    return sum(any(abs(value - center_deg) <= tol_deg for value in values) for values in per_trial_values)


def _classify_branch(center_deg: float, truth_mean_0: float, truth_mean_1: float) -> tuple[str, str, str]:
    if abs(center_deg - truth_mean_0) <= 2.5:
        return ("truth", "T0", "Truth T0")
    if abs(center_deg - truth_mean_1) <= 2.5:
        return ("truth", "T1 / endcap", "T1 / endcap sector")
    if abs(center_deg - (-24.0)) <= 4.0:
        return ("clutter", "Left-rack", "Left-rack clutter")
    if abs(center_deg - 23.0) <= 4.0:
        return ("clutter", "Right-rack", "Right-rack clutter")
    if abs(center_deg - (-11.0)) <= 3.0 or abs(center_deg - 10.0) <= 3.0:
        return ("multipath", "Wall bounce", "Wall-bounce multipath")
    return ("alias", "Alias", "Secondary alias")


def _annotation_offset(kind: str, center_deg: float) -> tuple[float, float]:
    if kind == "truth":
        return ((-14.0, 10.0) if center_deg < 0.0 else (14.0, 10.0))
    if kind == "clutter":
        return ((0.0, 5.5) if center_deg < 0.0 else (0.0, 4.8))
    if kind == "multipath":
        return (0.0, 14.0)
    if abs(center_deg) > 50.0:
        return (0.0, 10.0)
    return (0.0, 12.0)


def make_figure(data_dir, output_path) -> None:
    diag_rows = read_csv_rows(data_dir / "stage_diagnostics.csv")
    rack_rows = [row for row in diag_rows if row.get("scene_class") == "rack_aisle"]
    if not rack_rows:
        raise SystemExit("No rack_aisle rows found in stage_diagnostics.csv")

    per_trial_candidates: list[list[float]] = []
    truth_az_0: list[float] = []
    truth_az_1: list[float] = []
    per_trial_detections: list[list[float]] = []
    for row in rack_rows:
        trial_candidates = [float(candidate) for candidate in row["music_stage_azimuth_candidates_deg"].split("|") if candidate]
        trial_detections = [float(entry.split(":")[3]) for entry in row["detections"].split("|") if entry]
        per_trial_candidates.append(trial_candidates)
        per_trial_detections.append(trial_detections)
        for entry in row["truth_targets"].split("|"):
            parts = entry.split(":")
            azimuth = float(parts[5])
            if int(parts[0]) == 0:
                truth_az_0.append(azimuth)
            else:
                truth_az_1.append(azimuth)

    all_candidates = [value for trial_values in per_trial_candidates for value in trial_values]
    if not all_candidates:
        raise SystemExit("No candidate azimuths found in stage_diagnostics.csv")

    truth_mean_0 = float(np.mean(truth_az_0))
    truth_mean_1 = float(np.mean(truth_az_1))
    branch_centers = _cluster_centers(all_candidates)
    branch_rows: list[dict[str, object]] = []
    for center_deg in branch_centers:
        kind, short_label, long_label = _classify_branch(center_deg, truth_mean_0, truth_mean_1)
        branch_rows.append(
            {
                "center": center_deg,
                "kind": kind,
                "short_label": short_label,
                "long_label": long_label,
                "trial_count": _trial_presence_count(per_trial_candidates, center_deg, tol_deg=2.0),
            }
        )

    key_branches = (
        ("T0", truth_mean_0, METHOD_COLORS["music_masked"]),
        ("T1 / endcap", truth_mean_1, METHOD_COLORS["music_masked"]),
        ("Left-rack", -21.712, METHOD_COLORS["fft_masked"]),
        ("Right-rack", 25.232, METHOD_COLORS["fft_masked"]),
    )
    candidate_stage_counts = [_trial_presence_count(per_trial_candidates, center_deg, tol_deg=2.5) for _, center_deg, _ in key_branches]
    final_pair_counts = [_trial_presence_count(per_trial_detections, center_deg, tol_deg=3.0) for _, center_deg, _ in key_branches]

    fig, (ax_branch, ax_stage) = plt.subplots(1, 2, figsize=(12.5, 5.0), sharey=False)

    color_by_kind = {
        "truth": METHOD_COLORS["music_masked"],
        "clutter": METHOD_COLORS["fft_masked"],
        "multipath": "#5D6673",
        "alias": "#B4BAC2",
    }
    label_bbox = {
        "boxstyle": "round,pad=0.22,rounding_size=0.18",
        "facecolor": "white",
        "edgecolor": "#FFFFFF00",
        "alpha": 0.90,
    }

    for reference_count, reference_label in ((64, "100% of trials"), (32, "50% of trials")):
        ax_branch.axhline(reference_count, color="#CCD3DB", linewidth=1.0, linestyle=(0, (4, 3)), zorder=0)
        ax_branch.text(74.0, reference_count + 0.6, reference_label, ha="right", va="bottom", fontsize=8.5, color="#6E7781")

    for row in branch_rows:
        center_deg = float(row["center"])
        trial_count = int(row["trial_count"])
        kind = str(row["kind"])
        color = color_by_kind[kind]
        ax_branch.vlines(center_deg, 0, trial_count, color=color, linewidth=2.6, alpha=0.95, zorder=2)
        ax_branch.scatter(center_deg, trial_count, s=84, color=color, edgecolor="#3F4650", linewidth=0.6, zorder=3)
        dx, dy = _annotation_offset(kind, center_deg)
        if kind != "alias":
            ax_branch.annotate(
                f"{row['short_label']}\n{trial_count}/64",
                xy=(center_deg, trial_count),
                xytext=(center_deg + dx, trial_count + dy),
                ha="center",
                va="bottom",
                fontsize=8.6,
                color=color,
                bbox=label_bbox,
                arrowprops={"arrowstyle": "-", "color": color, "lw": 0.8, "alpha": 0.85},
            )

    ax_branch.text(
        -70.0,
        67.8,
        "4 secondary aliases\n61-64/64 trials",
        ha="left",
        va="center",
        fontsize=8.5,
        color="#5C6570",
        bbox=label_bbox,
    )

    ax_branch.set_xlim(-75, 75)
    ax_branch.set_ylim(0, 72)
    ax_branch.set_xlabel("Azimuth (°)")
    ax_branch.set_ylabel("Trials containing branch")
    ax_branch.set_title("Azimuth-stage branches that survive across trials", loc="left", fontsize=12, fontweight="bold")
    ax_branch.grid(True, axis="y", alpha=0.20)

    x = np.arange(len(key_branches))
    width = 0.34
    base_colors = [color for _, _, color in key_branches]
    bars_candidate = ax_stage.bar(
        x - width / 2,
        candidate_stage_counts,
        width,
        color=base_colors,
        alpha=0.28,
        edgecolor="#4A4A4A",
        linewidth=0.6,
    )
    bars_final = ax_stage.bar(
        x + width / 2,
        final_pair_counts,
        width,
        color=base_colors,
        alpha=0.90,
        edgecolor="#4A4A4A",
        linewidth=0.6,
    )
    for bars in (bars_candidate, bars_final):
        for bar in bars:
            count = int(bar.get_height())
            ax_stage.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.1,
                str(count),
                ha="center",
                va="bottom",
                fontsize=9.6,
                fontweight="bold",
            )
    ax_stage.set_xticks(x)
    ax_stage.set_xticklabels(("T0", "T1 / endcap", "Left-rack", "Right-rack"))
    ax_stage.set_ylim(0, 72)
    ax_stage.set_ylabel("Trials containing branch (of 64)")
    ax_stage.set_title("Key branches from azimuth stage to final pair", loc="left", fontsize=12, fontweight="bold")
    ax_stage.grid(True, axis="y", alpha=0.20)
    ax_stage.legend(
        handles=(
            Patch(facecolor="#697481", edgecolor="#4A4A4A", alpha=0.28, label="Azimuth-stage candidate set"),
            Patch(facecolor="#697481", edgecolor="#4A4A4A", alpha=0.90, label="Final output pair"),
        ),
        fontsize=8.5,
        loc="upper right",
        frameon=False,
    )

    fig.suptitle("Rack aisle: nuisance branches outlast the true pair\n64-trial FR1 nominal point, MDL model order", fontsize=14, fontweight="bold", y=0.985)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.14, top=0.82, wspace=0.27)
    save_figure(fig, output_path, bbox_inches="tight")


def main() -> None:
    parser = build_plot_parser(
        "Generate the rack-aisle failure-diagnostic figure.",
        "06_rack_aisle_failure_diagnostic.png",
        default_input_root=REPO_ROOT / "results" / "submission",
    )
    args = parser.parse_args()
    make_figure(resolve_data_dir(args.input_root), args.output)


if __name__ == "__main__":
    main()

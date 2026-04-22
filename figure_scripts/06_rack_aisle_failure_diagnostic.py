#!/usr/bin/env python3
"""Generate figures/06_rack_aisle_failure_diagnostic.png."""

from __future__ import annotations

from common import METHOD_COLORS, SCENE_COLORS, REPO_ROOT, build_plot_parser, np, plt, read_csv_rows, resolve_data_dir, save_figure


def make_figure(data_dir, output_path) -> None:
    diag_rows = read_csv_rows(data_dir / "stage_diagnostics.csv")
    rack_rows = [row for row in diag_rows if row.get("scene_class") == "rack_aisle"]
    if not rack_rows:
        raise SystemExit("No rack_aisle rows found in stage_diagnostics.csv")

    all_candidates: list[float] = []
    truth_az_0: list[float] = []
    truth_az_1: list[float] = []
    det_azimuths: list[float] = []
    for row in rack_rows:
        candidates = row["music_stage_azimuth_candidates_deg"]
        if candidates:
            all_candidates.extend(float(candidate) for candidate in candidates.split("|"))
        for entry in row["truth_targets"].split("|"):
            parts = entry.split(":")
            azimuth = float(parts[5])
            if int(parts[0]) == 0:
                truth_az_0.append(azimuth)
            else:
                truth_az_1.append(azimuth)
        for entry in row["detections"].split("|"):
            parts = entry.split(":")
            det_azimuths.append(float(parts[3]))

    if not all_candidates:
        raise SystemExit("No candidate azimuths found in stage_diagnostics.csv")

    truth_mean_0 = float(np.mean(truth_az_0))
    truth_mean_1 = float(np.mean(truth_az_1))
    clutter_azimuths = {"left_rack": -24.0, "right_rack": 23.0, "far_endcap": 3.0}
    multipath_azimuths = {"left_wall": -11.0, "right_wall": 10.0}

    fig, (ax_cand, ax_det) = plt.subplots(1, 2, figsize=(12.5, 5.0), sharey=False)

    bins = np.arange(-75, 80, 2.5)
    ax_cand.hist(all_candidates, bins=bins, color=SCENE_COLORS["rack_aisle"], alpha=0.55, edgecolor="#444444", linewidth=0.4)
    ax_cand.axvline(
        truth_mean_0,
        color=METHOD_COLORS["music_masked"],
        linewidth=2.6,
        linestyle=(0, (10, 4)),
        label=f"Truth T0 ({truth_mean_0:+.1f}°)",
    )
    ax_cand.axvline(
        truth_mean_1,
        color=METHOD_COLORS["music_masked"],
        linewidth=2.4,
        linestyle=(0, (6, 3)),
        label=f"Truth T1 ({truth_mean_1:+.1f}°)",
    )
    for name, azimuth in clutter_azimuths.items():
        ax_cand.axvline(
            azimuth,
            color=METHOD_COLORS["fft_masked"],
            linewidth=2.2,
            linestyle=(0, (1.0, 1.8)),
            label=f"Clutter: {name} ({azimuth:+.0f}°)",
        )
    for name, azimuth in multipath_azimuths.items():
        ax_cand.axvline(
            azimuth,
            color="#4D4D4D",
            linewidth=2.0,
            linestyle=(0, (8.0, 2.2, 1.4, 2.2)),
            alpha=0.95,
            label=f"Multipath: {name} ({azimuth:+.0f}°)",
        )
    ax_cand.set_xlabel("Azimuth (°)")
    ax_cand.set_ylabel("Candidate count (across 64 trials)")
    ax_cand.set_title("Azimuth candidates dominated by clutter branches", loc="left", fontsize=12, fontweight="bold")
    ax_cand.legend(fontsize=7.5, loc="upper left", frameon=False)
    ax_cand.set_xlim(-75, 75)
    ax_cand.grid(True, axis="y", alpha=0.20)

    det_arr = np.asarray(det_azimuths)
    n_near_nuis = int(np.sum(np.abs(det_arr - (-21.7)) < 3.0))
    n_near_t0 = int(np.sum(np.abs(det_arr - truth_mean_0) < 3.0))
    n_near_t1 = int(np.sum(np.abs(det_arr - truth_mean_1) < 3.0))
    n_other = len(det_arr) - n_near_nuis - n_near_t0 - n_near_t1

    categories = [
        f"Near T0\n({truth_mean_0:+.1f}°)",
        f"Near T1\n({truth_mean_1:+.1f}°)",
        "Near left rack\n(-21.7°)",
        "Other",
    ]
    counts = [n_near_t0, n_near_t1, n_near_nuis, n_other]
    colors = [METHOD_COLORS["music_masked"], METHOD_COLORS["music_masked"], METHOD_COLORS["fft_masked"], "#AAAAAA"]
    bars = ax_det.bar(categories, counts, color=colors, edgecolor="#444444", linewidth=0.6, alpha=0.75)
    for bar, count in zip(bars, counts, strict=True):
        ax_det.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, str(count), ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax_det.set_ylabel(f"Detection count (of {len(det_arr)} total)")
    ax_det.set_title("Final detections: nuisance branch captures 39% of outputs", loc="left", fontsize=12, fontweight="bold")
    ax_det.grid(True, axis="y", alpha=0.20)

    fig.suptitle("Rack-aisle azimuth-stage failure diagnostic\n64-trial FR1 nominal point, MDL model order", fontsize=14, fontweight="bold", y=1.02)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.13, top=0.88, wspace=0.28)
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

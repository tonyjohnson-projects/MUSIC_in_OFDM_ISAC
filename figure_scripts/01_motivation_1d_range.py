#!/usr/bin/env python3
"""Generate figures/01_motivation_1d_range.png."""

from __future__ import annotations

from scipy.signal import find_peaks

from common import METHOD_COLORS, build_plot_parser, np, plt, save_figure
from aisle_isac.estimators import fbss_covariance, music_pseudospectrum


def make_figure(output_path) -> None:
    n_subcarriers = 96
    subcarrier_spacing_hz = 30.0e3
    bandwidth_hz = n_subcarriers * subcarrier_spacing_hz
    speed_of_light_mps = 299_792_458.0
    range_resolution_m = speed_of_light_mps / (2.0 * bandwidth_hz)

    separation_factor = 0.70
    center_range_m = 20.0
    separation_m = separation_factor * range_resolution_m
    range_1 = center_range_m - separation_m / 2
    range_2 = center_range_m + separation_m / 2

    snr_db = 20.0
    n_snapshots = 16
    n_trials = 200
    rng = np.random.default_rng(42)

    frequencies_hz = np.arange(n_subcarriers, dtype=float) * subcarrier_spacing_hz
    frequencies_hz -= np.mean(frequencies_hz)

    n_search = 1001
    search_range_m = np.linspace(
        center_range_m - 3 * range_resolution_m,
        center_range_m + 3 * range_resolution_m,
        n_search,
    )

    delay_matrix = np.zeros((n_subcarriers, n_search), dtype=np.complex128)
    for index, range_m in enumerate(search_range_m):
        delay_s = 2.0 * range_m / speed_of_light_mps
        delay_matrix[:, index] = np.exp(-1j * 2.0 * np.pi * frequencies_hz * delay_s)

    avg_fft_spectrum = np.zeros(n_search)
    avg_music_spectrum = np.zeros(n_search)
    music_resolve_count = 0
    fft_resolve_count = 0
    threshold_m = 0.35 * range_resolution_m

    for _ in range(n_trials):
        signal_power = 1.0
        noise_var = signal_power / (10.0 ** (snr_db / 10.0))

        data = np.zeros((n_subcarriers, n_snapshots), dtype=np.complex128)
        for range_true in (range_1, range_2):
            delay_s = 2.0 * range_true / speed_of_light_mps
            steering = np.exp(-1j * 2.0 * np.pi * frequencies_hz * delay_s)
            source = rng.standard_normal((1, n_snapshots)) + 1j * rng.standard_normal((1, n_snapshots))
            source *= np.sqrt(signal_power / 2.0)
            data += steering[:, np.newaxis] * source

        noise = (rng.standard_normal(data.shape) + 1j * rng.standard_normal(data.shape)) * np.sqrt(
            noise_var / 2.0
        )
        data += noise

        window = np.hanning(n_subcarriers)
        windowed = data[:, 0] * window
        fft_result = np.fft.fftshift(np.fft.fft(windowed, n=n_search))
        fft_range_axis = (
            np.fft.fftshift(np.fft.fftfreq(n_search, d=subcarrier_spacing_hz))
            * speed_of_light_mps
            / 2.0
        )
        fft_power = np.abs(fft_result) ** 2
        fft_interp = np.interp(search_range_m, fft_range_axis + center_range_m, fft_power)
        fft_interp /= max(np.max(fft_interp), 1e-12)
        avg_fft_spectrum += fft_interp

        peaks_fft, _ = find_peaks(fft_interp, distance=5)
        if len(peaks_fft) >= 2:
            peak_scores = fft_interp[peaks_fft]
            top2 = peaks_fft[np.argsort(peak_scores)[-2:]]
            top2_ranges = sorted(search_range_m[top2])
            if abs(top2_ranges[0] - range_1) < threshold_m and abs(top2_ranges[1] - range_2) < threshold_m:
                fft_resolve_count += 1

        subarray_len = max(3, int(round(0.67 * n_subcarriers)))
        covariance = fbss_covariance(data, subarray_len)
        music_spec = music_pseudospectrum(covariance, 2, delay_matrix[:subarray_len, :])
        music_spec_norm = music_spec / max(np.max(music_spec), 1e-12)
        avg_music_spectrum += music_spec_norm

        peaks_music, _ = find_peaks(music_spec_norm, distance=5)
        if len(peaks_music) >= 2:
            peak_scores = music_spec_norm[peaks_music]
            top2 = peaks_music[np.argsort(peak_scores)[-2:]]
            top2_ranges = sorted(search_range_m[top2])
            if abs(top2_ranges[0] - range_1) < threshold_m and abs(top2_ranges[1] - range_2) < threshold_m:
                music_resolve_count += 1

    avg_fft_spectrum /= n_trials
    avg_music_spectrum /= n_trials
    avg_fft_spectrum /= max(np.max(avg_fft_spectrum), 1e-12)
    avg_music_spectrum /= max(np.max(avg_music_spectrum), 1e-12)

    fft_resolve_prob = fft_resolve_count / n_trials
    music_resolve_prob = music_resolve_count / n_trials

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 4))
    x_cells = (search_range_m - center_range_m) / range_resolution_m
    truth_color = "#6F6F6F"

    ax_left.plot(
        x_cells,
        10 * np.log10(np.maximum(avg_fft_spectrum, 1e-6)),
        color=METHOD_COLORS["fft_masked"],
        linewidth=1.6,
        label="FFT",
        zorder=2,
    )
    ax_left.plot(
        x_cells,
        10 * np.log10(np.maximum(avg_music_spectrum, 1e-6)),
        color=METHOD_COLORS["music_masked"],
        linewidth=1.7,
        label="MUSIC (K=2)",
        zorder=3,
    )
    ax_left.axvline(
        -separation_factor / 2,
        color=truth_color,
        linestyle=(0, (5, 4)),
        linewidth=0.6,
        alpha=0.7,
        label="Truth",
        zorder=1,
    )
    ax_left.axvline(
        separation_factor / 2,
        color=truth_color,
        linestyle=(0, (5, 4)),
        linewidth=0.6,
        alpha=0.7,
        zorder=1,
    )
    ax_left.set_xlabel("Range offset (resolution cells)")
    ax_left.set_ylabel("Normalized spectrum (dB)")
    ax_left.set_title(f"1-D range: {separation_factor:.2f}-cell separation, {snr_db:.0f} dB SNR")
    ax_left.set_xlim(-2.5, 2.5)
    ax_left.set_ylim(-30, 3)
    ax_left.legend(fontsize=8)
    ax_left.grid(True, alpha=0.3)

    methods = ["FFT", "MUSIC"]
    probabilities = [fft_resolve_prob, music_resolve_prob]
    bars = ax_right.bar(
        methods,
        probabilities,
        color=[METHOD_COLORS["fft_masked"], METHOD_COLORS["music_masked"]],
        width=0.5,
    )
    for bar, probability in zip(bars, probabilities, strict=True):
        ax_right.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{probability:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax_right.set_ylabel("Resolution probability")
    ax_right.set_title(f"P(resolve) over {n_trials} trials")
    ax_right.set_ylim(0, 1.15)
    ax_right.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"1-D Range-Only Motivation: Full Support, {n_subcarriers} Subcarriers, {n_snapshots} Snapshots",
        fontsize=11,
    )
    fig.tight_layout()
    save_figure(fig, output_path, dpi=150, bbox_inches="tight")
    print(f"FFT resolve: {fft_resolve_prob:.3f}, MUSIC resolve: {music_resolve_prob:.3f}")


def main() -> None:
    parser = build_plot_parser(
        "Generate the 1-D MUSIC motivation figure.",
        "01_motivation_1d_range.png",
    )
    args = parser.parse_args()
    make_figure(args.output)


if __name__ == "__main__":
    main()

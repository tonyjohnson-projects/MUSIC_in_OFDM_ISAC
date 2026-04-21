"""Generate 1-D range-only MUSIC vs FFT motivating figure.

Shows that in a clean 1-D setting with full support, MUSIC clearly resolves
two targets at sub-resolution separation while FFT cannot. This motivates
the project question: does this advantage survive waveform limitation?
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aisle_isac.estimators import (
    covariance_matrix,
    fbss_covariance,
    music_pseudospectrum,
)


def generate_1d_motivation(output_path: Path) -> None:
    """Generate a 1-D range MUSIC vs FFT comparison figure."""

    # --- Parameters ---
    n_subcarriers = 96
    subcarrier_spacing_hz = 30.0e3
    bandwidth_hz = n_subcarriers * subcarrier_spacing_hz
    c = 299_792_458.0
    range_resolution_m = c / (2.0 * bandwidth_hz)

    # Two targets at 0.7x resolution separation (sub-Rayleigh)
    separation_factor = 0.70
    center_range_m = 20.0
    separation_m = separation_factor * range_resolution_m
    range_1 = center_range_m - separation_m / 2
    range_2 = center_range_m + separation_m / 2

    snr_db = 20.0
    n_snapshots = 16
    n_trials = 200
    rng = np.random.default_rng(42)

    # Frequency axis
    frequencies_hz = np.arange(n_subcarriers, dtype=float) * subcarrier_spacing_hz
    frequencies_hz -= np.mean(frequencies_hz)

    # Dense search grid for spectra
    n_search = 1001
    search_range_m = np.linspace(
        center_range_m - 3 * range_resolution_m,
        center_range_m + 3 * range_resolution_m,
        n_search,
    )

    # Steering matrix for MUSIC
    delay_matrix = np.zeros((n_subcarriers, n_search), dtype=np.complex128)
    for i, r in enumerate(search_range_m):
        delay_s = 2.0 * r / c
        delay_matrix[:, i] = np.exp(-1j * 2.0 * np.pi * frequencies_hz * delay_s)

    # --- Accumulate average spectra ---
    avg_fft_spectrum = np.zeros(n_search)
    avg_music_spectrum = np.zeros(n_search)
    music_resolve_count = 0
    fft_resolve_count = 0
    threshold_m = 0.35 * range_resolution_m

    for _ in range(n_trials):
        # Generate signal
        signal_power = 1.0
        noise_var = signal_power / (10.0 ** (snr_db / 10.0))

        data = np.zeros((n_subcarriers, n_snapshots), dtype=np.complex128)
        for r_true in [range_1, range_2]:
            delay_s = 2.0 * r_true / c
            steering = np.exp(-1j * 2.0 * np.pi * frequencies_hz * delay_s)
            source = rng.standard_normal((1, n_snapshots)) + 1j * rng.standard_normal((1, n_snapshots))
            source *= np.sqrt(signal_power / 2.0)
            data += steering[:, np.newaxis] * source

        noise = (rng.standard_normal(data.shape) + 1j * rng.standard_normal(data.shape)) * np.sqrt(noise_var / 2.0)
        data += noise

        # FFT spectrum (zero-padded)
        n_fft = n_search
        window = np.hanning(n_subcarriers)
        windowed = data[:, 0] * window
        fft_result = np.fft.fftshift(np.fft.fft(windowed, n=n_fft))
        # Map FFT bins to range
        fft_range_axis = np.fft.fftshift(np.fft.fftfreq(n_fft, d=subcarrier_spacing_hz)) * c / 2.0
        fft_power = np.abs(fft_result) ** 2
        # Interpolate onto search grid
        fft_interp = np.interp(search_range_m, fft_range_axis + center_range_m, fft_power)
        fft_interp /= max(np.max(fft_interp), 1e-12)
        avg_fft_spectrum += fft_interp

        # Check FFT resolution: can it find two distinct peaks near truth?
        from scipy.signal import find_peaks
        peaks_fft, _ = find_peaks(fft_interp, distance=5)
        if len(peaks_fft) >= 2:
            peak_ranges = search_range_m[peaks_fft]
            peak_scores = fft_interp[peaks_fft]
            top2 = peaks_fft[np.argsort(peak_scores)[-2:]]
            top2_ranges = sorted(search_range_m[top2])
            if abs(top2_ranges[0] - range_1) < threshold_m and abs(top2_ranges[1] - range_2) < threshold_m:
                fft_resolve_count += 1

        # MUSIC spectrum (FBSS for decorrelation)
        subarray_len = max(3, int(round(0.67 * n_subcarriers)))
        cov = fbss_covariance(data, subarray_len)
        music_spec = music_pseudospectrum(cov, 2, delay_matrix[:subarray_len, :])
        music_spec_norm = music_spec / max(np.max(music_spec), 1e-12)
        avg_music_spectrum += music_spec_norm

        # Check MUSIC resolution
        peaks_music, _ = find_peaks(music_spec_norm, distance=5)
        if len(peaks_music) >= 2:
            peak_ranges = search_range_m[peaks_music]
            peak_scores = music_spec_norm[peaks_music]
            top2 = peaks_music[np.argsort(peak_scores)[-2:]]
            top2_ranges = sorted(search_range_m[top2])
            if abs(top2_ranges[0] - range_1) < threshold_m and abs(top2_ranges[1] - range_2) < threshold_m:
                music_resolve_count += 1

    avg_fft_spectrum /= n_trials
    avg_music_spectrum /= n_trials

    # Normalize to [0, 1]
    avg_fft_spectrum /= max(np.max(avg_fft_spectrum), 1e-12)
    avg_music_spectrum /= max(np.max(avg_music_spectrum), 1e-12)

    fft_resolve_prob = fft_resolve_count / n_trials
    music_resolve_prob = music_resolve_count / n_trials

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Normalized range axis
    x_cells = (search_range_m - center_range_m) / range_resolution_m

    fft_color = "#D55E00"
    music_color = "#0072B2"
    truth_color = "#6F6F6F"
    ax1.plot(
        x_cells,
        10 * np.log10(np.maximum(avg_fft_spectrum, 1e-6)),
        color=fft_color,
        linewidth=1.6,
        label="FFT",
        zorder=2,
    )
    ax1.plot(
        x_cells,
        10 * np.log10(np.maximum(avg_music_spectrum, 1e-6)),
        color=music_color,
        linewidth=1.7,
        label="MUSIC (K=2)",
        zorder=3,
    )
    ax1.axvline(
        -separation_factor / 2,
        color=truth_color,
        linestyle=(0, (5, 4)),
        linewidth=0.6,
        alpha=0.7,
        label="Truth",
        zorder=1,
    )
    ax1.axvline(
        separation_factor / 2,
        color=truth_color,
        linestyle=(0, (5, 4)),
        linewidth=0.6,
        alpha=0.7,
        zorder=1,
    )
    ax1.set_xlabel("Range offset (resolution cells)")
    ax1.set_ylabel("Normalized spectrum (dB)")
    ax1.set_title(f"1-D range: {separation_factor:.2f}-cell separation, {snr_db:.0f} dB SNR")
    ax1.set_xlim(-2.5, 2.5)
    ax1.set_ylim(-30, 3)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Resolution probability bar chart
    methods = ["FFT", "MUSIC"]
    probs = [fft_resolve_prob, music_resolve_prob]
    colors = [fft_color, music_color]
    bars = ax2.bar(methods, probs, color=colors, width=0.5)
    for bar, prob in zip(bars, probs):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{prob:.2f}", ha="center", va="bottom", fontsize=10)
    ax2.set_ylabel("Resolution probability")
    ax2.set_title(f"P(resolve) over {n_trials} trials")
    ax2.set_ylim(0, 1.15)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"1-D Range-Only Motivation: Full Support, {n_subcarriers} Subcarriers, {n_snapshots} Snapshots",
        fontsize=11,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")
    print(f"FFT resolve: {fft_resolve_prob:.3f}, MUSIC resolve: {music_resolve_prob:.3f}")


if __name__ == "__main__":
    output = REPO_ROOT / "results" / "figures" / "motivation_1d_range.png"
    generate_1d_motivation(output)

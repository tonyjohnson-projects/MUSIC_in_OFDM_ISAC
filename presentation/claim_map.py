"""Claim-to-evidence mapping for the presentation deck."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ClaimEntry:
    """One claim that appears on a slide."""

    slide_id: str
    slide_title: str
    claim: str
    evidence: tuple[str, ...]


def get_claim_map() -> tuple[ClaimEntry, ...]:
    """Return the deck's claim map."""

    return (
        ClaimEntry(
            slide_id="s02-why-isac",
            slide_title="Why sense from the communication waveform?",
            claim="ISAC is now a formal 6G usage scenario rather than a niche side topic.",
            evidence=("itu_imt2030", "itu_imt2030_framework", "gonzalez2024", "etsi_2023"),
        ),
        ClaimEntry(
            slide_id="s04-fit-in-research",
            slide_title="This project asks an applicability question inside the broader ISAC push",
            claim="Current ISAC work emphasizes network-native sensing, deployment realism, and evaluation assumptions, which makes a waveform-limited applicability study relevant.",
            evidence=("gonzalez2024", "nextg_2025", "etsi_2023", "repo_report"),
        ),
        ClaimEntry(
            slide_id="s05-1d-motivation",
            slide_title="Clean 1-D data shows genuine sub-Rayleigh upside",
            claim="Under full support and known order, the motivating 1-D study resolves a 0.70-cell separation with MUSIC while FFT fails to separate the pair.",
            evidence=("repo_report",),
        ),
        ClaimEntry(
            slide_id="s07-fft-vs-music-vs-crb",
            slide_title="MUSIC can beat FFT binning, but not information limits",
            claim="The project compares FFT resolution limits against subspace super-resolution while explicitly avoiding a claim of CRB attainment for the masked staged architecture.",
            evidence=("repo_report", "stoica1989", "schmidt1986"),
        ),
        ClaimEntry(
            slide_id="s08-scope-and-fairness",
            slide_title="The comparison is fair and intentionally narrow",
            claim="Both headline methods use the same de-embedded masked observation, and the FFT baseline includes local refinement rather than a weak raw FFT front end.",
            evidence=("repo_report", "repo_readme"),
        ),
        ClaimEntry(
            slide_id="s09-waveform-mask",
            slide_title="The nominal point is fragmented full-span OFDM, not a toy narrowband case",
            claim="The nominal submission setting uses a 3.5 GHz FR1 anchor, 100 MHz occupied bandwidth, 96 simulated subcarriers, 16 snapshots, and 50% fragmented occupancy that still spans the full bandwidth and slow-time aperture.",
            evidence=("repo_report",),
        ),
        ClaimEntry(
            slide_id="s10-scene-regimes",
            slide_title="The three scenes are composite operating regimes",
            claim="Open aisle, intersection, and rack aisle differ in SNR, separations, target mix, power imbalance, clutter, and multipath, so the study interprets them as composite regimes rather than a clean coherence-only sweep.",
            evidence=("repo_report",),
        ),
        ClaimEntry(
            slide_id="s12-nominal-verdict",
            slide_title="MUSIC only wins the nominal intersection scene",
            claim="At the saved FR1 nominal point, MUSIC reaches P_joint = 0.703 in intersection versus 0.156 for FFT, but loses in open aisle and collapses in rack aisle.",
            evidence=("repo_nominal_summary", "repo_report"),
        ),
        ClaimEntry(
            slide_id="s13-representative-case",
            slide_title="On a saved nominal intersection trial, MUSIC resolves both movers while FFT latches onto a false branch",
            claim="A reconstructed saved nominal trial shows FFT locking onto a high-range false target while MUSIC lands on both movers.",
            evidence=("repo_trial_level",),
        ),
        ClaimEntry(
            slide_id="s14-regime-map",
            slide_title="MUSIC's value clusters by regime rather than dominating everywhere",
            claim="Sweep-level results show broad support-limited gains in intersection, almost no strong-win region in open aisle, and no rescue of nominal rack aisle.",
            evidence=("repo_report", "repo_nominal_summary"),
        ),
        ClaimEntry(
            slide_id="s15-model-order",
            slide_title="Much of the open-aisle loss is a model-order problem, not a missing super-resolution effect",
            claim="Expected-order MUSIC recovers open aisle to parity with FFT and strengthens intersection, while rack aisle remains unresolved even when the model order is fixed.",
            evidence=("repo_model_order", "repo_model_order_eigengap", "repo_report"),
        ),
        ClaimEntry(
            slide_id="s16-rack-failure",
            slide_title="Rack aisle fails because nuisance-aligned azimuth structure captures the spatial search",
            claim="The rack-aisle diagnostic shows a persistent nuisance-aligned azimuth branch near the left-rack clutter and roughly 39% of final detections landing on that nuisance branch.",
            evidence=("repo_stage_diag", "repo_report"),
        ),
        ClaimEntry(
            slide_id="s17-thesis",
            slide_title="The thesis is a resolution-versus-robustness map",
            claim="The strongest defensible claim is conditional MUSIC benefit: favorable in intersection, recoverable in open aisle with better order control, and unusable in clutter-dominated rack aisle.",
            evidence=("repo_report", "repo_nominal_summary", "repo_model_order"),
        ),
        ClaimEntry(
            slide_id="a03-nuisance-sweep",
            slide_title="Expected-order nuisance sweep",
            claim="Once order control is fixed, open aisle stays competitive only at lower nuisance levels, intersection remains robust across the tested range, and rack aisle remains at zero.",
            evidence=("repo_expected_nuisance_open", "repo_expected_nuisance_intersection", "repo_expected_nuisance_rack", "repo_report"),
        ),
        ClaimEntry(
            slide_id="a04-runtime",
            slide_title="Runtime comparison",
            claim="At the nominal FR1 point, staged MUSIC adds roughly 0.19 to 0.26 s per point over FFT, corresponding to roughly a 25-30% increase in total point runtime.",
            evidence=("repo_runtime_summary", "repo_report"),
        ),
    )


def get_claim_dicts() -> list[dict[str, object]]:
    """Return plain dictionaries for serialization."""

    return [asdict(entry) for entry in get_claim_map()]


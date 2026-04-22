"""Reference metadata for the presentation deck."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SourceSpec:
    """One citation or internal evidence source."""

    id: str
    title: str
    citation: str
    kind: str
    url: str | None = None
    note: str | None = None


def get_sources() -> tuple[SourceSpec, ...]:
    """Return all sources used by the deck and claim map."""

    return (
        SourceSpec(
            id="repo_readme",
            title="Project README",
            citation="Project README summary and thesis framing.",
            kind="internal",
            note="README.md",
        ),
        SourceSpec(
            id="repo_report",
            title="Current project assessment",
            citation="Tony Johnson, 'Super-Resolution in Waveform-Limited OFDM ISAC: Capability and Robustness of Staged MUSIC,' April 2026.",
            kind="internal",
            note="report/current_assessment.tex",
        ),
        SourceSpec(
            id="repo_nominal_summary",
            title="Nominal submission summary",
            citation="Saved FR1 nominal summary for FFT and MUSIC across the three scene classes.",
            kind="internal",
            note="results/submission/data/nominal_summary.csv",
        ),
        SourceSpec(
            id="repo_trial_level",
            title="Per-trial nominal outcomes",
            citation="Saved trial-level nominal results used for paired comparisons and representative-trial reconstruction.",
            kind="internal",
            note="results/submission/data/trial_level_results.csv",
        ),
        SourceSpec(
            id="repo_stage_diag",
            title="Stage diagnostics",
            citation="Saved azimuth-stage diagnostics supporting the rack-aisle failure interpretation.",
            kind="internal",
            note="results/submission/data/stage_diagnostics.csv",
        ),
        SourceSpec(
            id="repo_model_order",
            title="Model-order nominal comparison",
            citation="Saved 64-trial nominal comparison for MDL and expected-order MUSIC.",
            kind="internal",
            note="results/analysis/model_order_nominal_64trials.csv",
        ),
        SourceSpec(
            id="repo_model_order_eigengap",
            title="Model-order nominal comparison with eigengap",
            citation="Saved nominal eigengap follow-on comparison.",
            kind="internal",
            note="results/analysis/model_order_nominal_64trials_eigengap.csv",
        ),
        SourceSpec(
            id="repo_runtime_summary",
            title="Runtime summary",
            citation="Saved runtime comparison between the FFT baseline and staged MUSIC.",
            kind="internal",
            note="results/submission/data/runtime_summary.csv",
        ),
        SourceSpec(
            id="repo_expected_nuisance_open",
            title="Expected-order nuisance sweep: open aisle",
            citation="Saved expected-order nuisance-strength sweep for the open-aisle scene.",
            kind="internal",
            note="results/submission_expected_order/open_aisle/data/nuisance_gain_offset.csv",
        ),
        SourceSpec(
            id="repo_expected_nuisance_intersection",
            title="Expected-order nuisance sweep: intersection",
            citation="Saved expected-order nuisance-strength sweep for the intersection scene.",
            kind="internal",
            note="results/submission_expected_order/intersection/data/nuisance_gain_offset.csv",
        ),
        SourceSpec(
            id="repo_expected_nuisance_rack",
            title="Expected-order nuisance sweep: rack aisle",
            citation="Saved expected-order nuisance-strength sweep for the rack-aisle scene.",
            kind="internal",
            note="results/submission_expected_order/rack_aisle/data/nuisance_gain_offset.csv",
        ),
        SourceSpec(
            id="liu2022",
            title="Integrated Sensing and Communications: Toward Dual-Functional Wireless Networks for 6G and Beyond",
            citation="F. Liu et al., IEEE JSAC, vol. 40, no. 6, pp. 1728-1767, 2022.",
            kind="external",
            url="https://doi.org/10.1109/JSAC.2022.3156632",
        ),
        SourceSpec(
            id="gonzalez2024",
            title="The Integrated Sensing and Communication Revolution for 6G: Vision, Techniques, and Applications",
            citation="N. Gonzalez-Prelcic et al., Proceedings of the IEEE, vol. 112, no. 7, pp. 676-723, 2024.",
            kind="external",
            url="https://doi.org/10.1109/JPROC.2024.3397609",
        ),
        SourceSpec(
            id="itu_imt2030",
            title="IMT-2030: Technical requirements for the 6G future",
            citation="International Telecommunication Union, technical update on IMT-2030 usage scenarios and requirements, March 2026.",
            kind="external",
            url="https://www.itu.int/hub/2026/03/imt-2030-technical-requirements-for-the-6g-future/",
        ),
        SourceSpec(
            id="itu_imt2030_framework",
            title="IMT towards 2030 and beyond (IMT-2030)",
            citation="International Telecommunication Union Radiocommunication Sector, IMT-2030 framework page, accessed April 20, 2026.",
            kind="external",
            url="https://www.itu.int/en/ITU-R/study-groups/rsg5/rwp5d/imt-2030/Pages/default.aspx",
        ),
        SourceSpec(
            id="etsi_2023",
            title="ETSI launches a new group for integrated sensing and communications",
            citation="ETSI press release on pre-standard ISAC work for 6G, November 2023.",
            kind="external",
            url="https://www.etsi.org/newsroom/press-releases/2291-etsi-launches-a-new-group-for-integrated-sensing-and-communications-a-candidate-technology-for-6g/",
        ),
        SourceSpec(
            id="nextg_2025",
            title="Integrated Sensing and Communications Readiness Report, Phase I",
            citation="Next G Alliance, Phase I readiness report summary for ISAC deployment and evaluation priorities, September 2025.",
            kind="external",
            url="https://nextgalliance.org/white_papers/integrated-sensing-and-communications-readiness-report-phase-i/",
        ),
        SourceSpec(
            id="schmidt1986",
            title="Multiple Emitter Location and Signal Parameter Estimation",
            citation="R. O. Schmidt, IEEE Transactions on Antennas and Propagation, vol. 34, no. 3, pp. 276-280, 1986.",
            kind="external",
        ),
        SourceSpec(
            id="shan1985",
            title="On Spatial Smoothing for Direction-of-Arrival Estimation of Coherent Signals",
            citation="T.-J. Shan, M. Wax, and T. Kailath, IEEE TASSP, vol. 33, no. 4, pp. 806-811, 1985.",
            kind="external",
            url="https://doi.org/10.1109/TASSP.1985.1164649",
        ),
        SourceSpec(
            id="pillai1989",
            title="Forward/Backward Spatial Smoothing Techniques for Coherent Signal Identification",
            citation="S. U. Pillai and B. H. Kwon, IEEE TASSP, vol. 37, no. 1, pp. 8-15, 1989.",
            kind="external",
            url="https://doi.org/10.1109/29.17496",
        ),
        SourceSpec(
            id="stoica1989",
            title="MUSIC, Maximum Likelihood, and Cramer-Rao Bound",
            citation="P. Stoica and A. Nehorai, IEEE TASSP, vol. 37, no. 5, pp. 720-741, 1989.",
            kind="external",
            url="https://doi.org/10.1109/29.17564",
        ),
        SourceSpec(
            id="sturm2009",
            title="An OFDM system concept for joint radar and communications operations",
            citation="C. Sturm, T. Zwick, and W. Wiesbeck, IEEE VTC Spring, 2009.",
            kind="external",
            url="https://doi.org/10.1109/VETECS.2009.5073387",
        ),
    )


def get_source_dicts() -> list[dict[str, str | None]]:
    """Return plain dictionaries for serialization."""

    return [asdict(source) for source in get_sources()]


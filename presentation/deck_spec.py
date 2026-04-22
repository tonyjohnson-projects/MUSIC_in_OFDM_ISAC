"""Deck specification for the final project presentation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class SlideSpec:
    """One slide in the final deck."""

    id: str
    section: str
    title: str
    takeaway: str
    visible_copy: tuple[str, ...]
    visuals: tuple[str, ...]
    speaker_notes: tuple[str, ...]
    sources: tuple[str, ...]
    appendix: bool = False
    layout: str = "bullets"
    metadata: dict[str, object] = field(default_factory=dict)


def get_slides() -> tuple[SlideSpec, ...]:
    """Return the full deck specification."""

    return (
        SlideSpec(
            id="s01-title",
            section="Opening",
            title="Super-resolution in waveform-limited OFDM ISAC",
            takeaway="This talk asks where staged MUSIC remains useful once sensing is forced onto a communications-style OFDM grid.",
            visible_copy=(
                "MSEE final project",
                "Capability and robustness of staged MUSIC under fragmented OFDM support, nuisance clutter, and a fair FFT baseline",
                "20-25 minute research talk with appendix backup",
            ),
            visuals=(),
            speaker_notes=(
                "Open with the narrow claim: this is not a generic 'MUSIC beats FFT' talk.",
                "State the problem as a regime-boundary question for communications-limited sensing.",
                "Preview the answer: MUSIC helps, but only in some operating regions.",
            ),
            sources=("repo_report", "repo_readme"),
            layout="title",
            metadata={"subtitle": "Final presentation deck", "author": "Tony Johnson", "date": "Spring 2026"},
        ),
        SlideSpec(
            id="s02-why-isac",
            section="Opening",
            title="Why sense from the communication waveform?",
            takeaway="ISAC matters because 6G is pushing networks to share spectrum, infrastructure, and environmental awareness instead of treating sensing as a separate radar stack.",
            visible_copy=(
                "Infrastructure already transmits wideband OFDM and already owns the aperture, synchronization, and compute.",
                "Using that waveform for sensing could add range, Doppler, angle, and context without a separate radar band.",
                "That makes the real question practical: what sensing quality survives when the waveform is communications-driven rather than radar-designed?",
            ),
            visuals=(),
            speaker_notes=(
                "Use ITU and current overview work to frame ISAC as part of the 6G roadmap, not a side curiosity.",
                "Stress that communications-native sensing is attractive precisely because it reuses the network's existing assets.",
                "Transition into the central difficulty: a communications waveform is not an ideal radar waveform.",
            ),
            sources=("itu_imt2030", "itu_imt2030_framework", "gonzalez2024", "liu2022"),
            metadata={"kicker": "6G context"},
        ),
        SlideSpec(
            id="s03-project-question",
            section="Opening",
            title="The project question is conditional, not 'is MUSIC always better?'",
            takeaway="The study asks whether MUSIC keeps any practically useful super-resolution upside once the observation is masked by communications scheduling and nuisance structure.",
            visible_copy=(
                "Can staged MUSIC, aided by FBSS, beat a fair masked FFT baseline on the same known-symbol OFDM observation?",
                "The testbed is intentionally realistic enough to include fragmented support, coherent multipath, and structured clutter.",
                "The outcome should be a regime map, not a broad victory claim.",
            ),
            visuals=(),
            speaker_notes=(
                "Read the question almost verbatim, because it defines the scope and keeps the talk honest.",
                "Mention that the earlier 1-D result motivates the question but does not settle it.",
                "Use 'regime map' as the phrase that will come back in the conclusion.",
            ),
            sources=("repo_report", "repo_readme"),
        ),
        SlideSpec(
            id="s04-fit-in-research",
            section="Opening",
            title="This project asks an applicability question inside the broader ISAC push",
            takeaway="The contribution is not a new theorem or a new waveform; it is a study of when classical super-resolution machinery is worth carrying into communications-limited OFDM sensing.",
            visible_copy=(
                "Current 6G ISAC work emphasizes network-native sensing, deployment realism, evaluation assumptions, and use-case fit.",
                "This project sits below that system vision and asks whether subspace super-resolution remains valuable once waveform support is irregular.",
                "That makes the result useful even without novelty at the estimator-theory level.",
            ),
            visuals=(),
            speaker_notes=(
                "Place the work as an engineering applicability study rather than an algorithm invention paper.",
                "This is a strong framing move for a faculty audience because it anticipates the obvious novelty question.",
                "If asked why OFDM: it is the natural communications anchor in ISAC and the literature already treats it that way.",
            ),
            sources=("gonzalez2024", "nextg_2025", "etsi_2023", "repo_report"),
            metadata={"kicker": "Research fit"},
        ),
        SlideSpec(
            id="s05-1d-motivation",
            section="Theory",
            title="Clean 1-D data shows genuine sub-Rayleigh upside",
            takeaway="The super-resolution machinery is real in the clean setting: with full support and known order, MUSIC separates a pair that FFT merges.",
            visible_copy=(
                "Two targets at 0.70 resolution-cell separation, 96 subcarriers, 16 snapshots, 20 dB SNR.",
                "FFT cannot separate the pair at all in the motivating study.",
                "MUSIC with known K = 2 resolves both targets across all 200 trials.",
            ),
            visuals=("motivation_1d_range",),
            speaker_notes=(
                "Use this slide to establish why MUSIC is even worth carrying into a harsher setting.",
                "Do not oversell it: the clean 1-D case is a motivation, not the main evidence base for the thesis.",
                "Transition by asking how much of this survives once support is masked and nuisance structure appears.",
            ),
            sources=("repo_report",),
            layout="figure_focus",
            metadata={"figure_caption": "Motivating 1-D range-only study"},
        ),
        SlideSpec(
            id="s06-masked-observation",
            section="Theory",
            title="The sensing cube is masked by the communication schedule before either estimator ever sees it",
            takeaway="Both methods operate on the same de-embedded known-symbol observation, but that observation is sparse, irregular, and hostile to covariance estimation.",
            visible_copy=(
                "Observation model: y(h,k,n) combines steering, delay, Doppler, transmitted symbols, a binary support mask, and noise.",
                "Known-symbol de-embedding produces a masked sensing cube with structured zeros on unknown or unused resource elements.",
                "That masking is the core reason the problem is harder than classical radar super-resolution.",
            ),
            visuals=("masked_observation_equation",),
            speaker_notes=(
                "Write the masked observation equation on the slide and talk through only the parts that matter for the argument.",
                "The key message is that the schedule and symbol knowledge directly shape the sensing object.",
                "That gives you the bridge to why FFT and MUSIC both suffer, but in different ways.",
            ),
            sources=("repo_report",),
            layout="equation",
            metadata={
                "equation_note": "Known-symbol sensing forms ỹ(h,k,n) = y(h,k,n) / x(k,n) on the known support set Ω, and 0 elsewhere.",
            },
        ),
        SlideSpec(
            id="s07-fft-vs-music-vs-crb",
            section="Theory",
            title="MUSIC can beat FFT binning, but not information limits",
            takeaway="The relevant comparison is FFT resolution versus subspace super-resolution under masked covariance, with the CRB acting as a bound rather than as an achieved target in this staged architecture.",
            visible_copy=(
                "FFT is tied to sampled aperture and bin spacing, even with oversampling and local refinement.",
                "MUSIC can resolve below FFT bin spacing when the covariance estimate preserves a clean signal/noise subspace split.",
                "The masked staged pipeline does not claim CRB efficiency, especially once order error and conditioning loss appear.",
            ),
            visuals=(),
            speaker_notes=(
                "Keep this slide conceptual; do not try to derive the CRB here.",
                "Use it to say what the project can and cannot claim about theoretical limits.",
                "This is also where you define FBSS as a tool for coherent-source recovery, not as a guarantee.",
            ),
            sources=("repo_report", "schmidt1986", "shan1985", "pillai1989", "stoica1989"),
        ),
        SlideSpec(
            id="s08-scope-and-fairness",
            section="Study design",
            title="The comparison is fair and intentionally narrow",
            takeaway="The study is narrow enough to defend, but strong enough to say something useful about real OFDM sensing regimes.",
            visible_copy=(
                "Two movers, three indoor industrial scene classes, monostatic FR1 OFDM, and known-symbol sensing.",
                "Both headline methods start from the same de-embedded masked cube.",
                "The FFT baseline includes local matched-filter refinement, so the comparison is not MUSIC versus a strawman FFT.",
            ),
            visuals=("music_pseudospectrum_equation",),
            speaker_notes=(
                "Be explicit that the project is not claiming unknown-data reception, tracking, or hardware validation.",
                "This slide is mainly about credibility: tight scope, fair baseline, and auditability.",
                "Set up the next slide by noting that the nominal point is not a toy fully contiguous radar cube.",
            ),
            sources=("repo_report", "repo_readme"),
            layout="cards",
            metadata={
                "cards": (
                    {"title": "Scope", "body": "2 movers, 3 scene classes, FR1 anchor, known-symbol sensing"},
                    {"title": "FFT path", "body": "Weighted masked FFT plus local refinement"},
                    {"title": "MUSIC path", "body": "Staged azimuth, range, Doppler MUSIC with support-aware FBSS"},
                )
            },
        ),
        SlideSpec(
            id="s09-waveform-mask",
            section="Study design",
            title="The nominal point is fragmented full-span OFDM, not a toy narrowband case",
            takeaway="Even at 50% occupancy, the nominal point spans the full bandwidth and full slow-time aperture, so the challenge comes from fragmentation and masking rather than from an artificially tiny waveform.",
            visible_copy=(
                "FR1 anchor: 3.5 GHz carrier, 100 MHz occupied bandwidth, 30 kHz SCS.",
                "Study grid: 96 simulated subcarriers, 16 slot-spaced snapshots, 16 effective horizontal channels.",
                "Nominal mask: fragmented scheduled PRB occupancy at 50%, but full bandwidth span and full slow-time span.",
            ),
            visuals=("nominal_resource_mask",),
            speaker_notes=(
                "Use the table to justify the engineering setup in physical terms: range, Doppler, and angle resolution all come from this slide.",
                "Emphasize that the support is fragmented, not simply small.",
                "That makes the masked-covariance problem more interesting than a low-bandwidth toy case.",
            ),
            sources=("repo_report",),
            layout="table_figure",
            metadata={
                "table_rows": (
                    ("Carrier frequency", "3.5 GHz"),
                    ("Occupied bandwidth", "100 MHz"),
                    ("Subcarrier spacing", "30 kHz"),
                    ("Sampled subcarriers", "96"),
                    ("Snapshots", "16 over 8 ms CPI"),
                    ("Occupied fraction", "0.50 fragmented full-span"),
                ),
            },
        ),
        SlideSpec(
            id="s10-scene-regimes",
            section="Study design",
            title="The three scenes are composite operating regimes",
            takeaway="Scene labels are convenient, but the real interpretation is three different operating regimes with different SNR, separations, clutter, and multipath difficulty.",
            visible_copy=(
                "Open aisle: higher SNR and wider separations, but not automatically MUSIC-favorable once FFT is refined.",
                "Intersection: moderate SNR but geometry that leaves room for MUSIC to exploit resolution structure.",
                "Rack aisle: lower SNR, tighter separations, stronger clutter and coherent nuisance paths.",
            ),
            visuals=(),
            speaker_notes=(
                "Be careful here: do not call this a pure coherence sweep.",
                "Say explicitly that the scenes are composite regimes and that the project uses that wording in the final interpretation.",
                "This slide sets up why the verdict will not be one-dimensional.",
            ),
            sources=("repo_report",),
            layout="table_cards",
            metadata={
                "table_rows": (
                    ("Open aisle", "20 dB", "1.25 / 1.35 / 1.25 cells", "2 clutter / 1 multipath"),
                    ("Intersection", "18 dB", "1.00 / 1.25 / 1.55 cells", "3 clutter / 1 multipath"),
                    ("Rack aisle", "16 dB", "0.85 / 0.90 / 0.90 cells", "3 clutter / 2 multipath"),
                ),
                "table_headers": ("Scene", "Nominal SNR", "Range / Vel / Angle sep.", "Static / multipath"),
            },
        ),
        SlideSpec(
            id="s11-bespoke-stack",
            section="Study design",
            title="The study required a bespoke simulation and evaluation stack",
            takeaway="The contribution is not one function; it is the full end-to-end machinery needed to ask the question fairly and auditably.",
            visible_copy=(
                "Masked OFDM observation generation and resource-grid scheduling model.",
                "Scene, clutter, and multipath generator for three industrial regimes.",
                "Weighted masked FFT baseline, staged MUSIC with support-aware FBSS, Monte Carlo sweep harness, and trial evaluation/reporting pipeline.",
            ),
            visuals=(),
            speaker_notes=(
                "This replaces the earlier 'custom vs COTS' idea entirely.",
                "Describe the pipeline as bespoke engineering for this study: observation model, estimator paths, metrics, and reporting.",
                "Do not mention NumPy or SciPy on the slide; they are not the interesting part.",
            ),
            sources=("repo_readme", "repo_report"),
            layout="process",
            metadata={
                "steps": (
                    "Scenario + clutter generator",
                    "Masked OFDM observation",
                    "Shared FFT front end",
                    "FFT path + local refinement",
                    "Staged MUSIC + FBSS",
                    "Trial metrics + sweep reporting",
                )
            },
        ),
        SlideSpec(
            id="s12-nominal-verdict",
            section="Results",
            title="MUSIC only wins the nominal intersection scene",
            takeaway="The saved 64-trial nominal point already rules out any broad claim that MUSIC is generically better in waveform-limited OFDM sensing.",
            visible_copy=(
                "Open aisle favors the FFT baseline after refinement.",
                "Intersection is a clear nominal win for staged MUSIC.",
                "Rack aisle is a hard failure regime, with MUSIC collapsing at the nominal point.",
            ),
            visuals=("story_nominal_verdict",),
            speaker_notes=(
                "Walk left to right through the scene verdicts and keep the claim narrow.",
                "The strongest sound bite is that the nominal result alone rejects a universal-MUSIC story.",
                "Use the confidence intervals briefly, then move on to mechanism slides.",
            ),
            sources=("repo_nominal_summary", "repo_report"),
            layout="figure_focus",
            metadata={"figure_caption": "Saved FR1 nominal result with 95% Wilson intervals"},
        ),
        SlideSpec(
            id="s13-representative-case",
            section="Results",
            title="On a saved nominal intersection trial, MUSIC resolves both movers while FFT latches onto a false branch",
            takeaway="The visual mechanism is concrete: the refined FFT path can still jump to a plausible false alternative, while the MUSIC path lands on the true pair in the same nominal trial.",
            visible_copy=(
                "Saved nominal trial 55 from the paired submission bundle.",
                "FFT finds one true mover and one high-range false branch.",
                "MUSIC lands on both movers within the joint gate on the same trial.",
            ),
            visuals=("representative_intersection_case",),
            speaker_notes=(
                "Call this a reconstructed saved nominal trial so the provenance is clear.",
                "Use the left panel to point out the false FFT branch and the right panel to connect that with the conditioned azimuth slice.",
                "This is the highest-value 'at a glance' figure in the main deck after the verdict slide.",
            ),
            sources=("repo_trial_level",),
            layout="figure_focus",
            metadata={"figure_caption": "Representative nominal intersection trial reconstructed from the saved bundle"},
        ),
        SlideSpec(
            id="s14-regime-map",
            section="Results",
            title="MUSIC's value clusters by regime rather than dominating everywhere",
            takeaway="The more useful project result is the regime map: support-limited and axis-isolated sweeps show where the earlier super-resolution upside survives and where it does not.",
            visible_copy=(
                "Intersection shows broad support-limited upside for MUSIC.",
                "Open aisle offers almost no strong-win region once the FFT baseline is refined.",
                "Rack aisle retains no meaningful support-limited rescue region, even if some easier separation points remain positive.",
            ),
            visuals=("story_regime_map",),
            speaker_notes=(
                "Split the slide verbally into support-limited versus separation sweeps; the figure already does this visually.",
                "Say that this is the main high-level result of the project.",
                "Use it to transition into why open aisle is not purely 'MUSIC is bad' and why rack aisle is different.",
            ),
            sources=("repo_report",),
            layout="figure_focus",
            metadata={"figure_caption": "Mean MUSIC-minus-FFT delta by sweep family"},
        ),
        SlideSpec(
            id="s15-model-order",
            section="Analysis",
            title="Much of the open-aisle loss is a model-order problem, not a missing super-resolution effect",
            takeaway="Fixing the order recovers open aisle to parity and strengthens intersection, which shows that the super-resolution machinery is still there when subspace dimensioning is controlled.",
            visible_copy=(
                "Open aisle: MDL MUSIC loses, eigengap nearly recovers, and expected-order MUSIC reaches FFT parity.",
                "Intersection: both eigengap and expected-order MUSIC improve on the already-strong MDL result.",
                "Rack aisle: no order-control fix rescues the scene, so the failure mechanism is elsewhere.",
            ),
            visuals=("model_order_nominal_comparison",),
            speaker_notes=(
                "This is the key diagnosis slide for the open-aisle story.",
                "Stress that expected-order is an oracle diagnostic, not a deployment-ready solution.",
                "The real point is causal: open aisle is largely an order-estimation problem, rack aisle is not.",
            ),
            sources=("repo_model_order", "repo_model_order_eigengap", "repo_report"),
            layout="figure_focus",
            metadata={"figure_caption": "Nominal P_joint across FFT, MDL MUSIC, eigengap MUSIC, and expected-order MUSIC"},
        ),
        SlideSpec(
            id="s16-rack-failure",
            section="Analysis",
            title="Rack aisle fails because nuisance-aligned azimuth structure captures the spatial search",
            takeaway="Rack aisle is not just a harder nominal point; it is a structurally different failure regime in which nuisance-aligned candidates dominate the azimuth stage early enough that later processing cannot recover.",
            visible_copy=(
                "A persistent nuisance-aligned azimuth branch appears near the left-rack clutter angle.",
                "The true second mover is not stably preserved in the useful candidate set.",
                "Final detections are repeatedly pulled onto the nuisance branch instead of the true pair.",
            ),
            visuals=("story_rack_aisle_diagnostic",),
            speaker_notes=(
                "Use the candidate histogram first, then the final-detection histogram.",
                "Tie this back to the fact that fixing the order does not change the rack-aisle outcome.",
                "This slide is the main evidence for the 'hard regime boundary' language in the conclusion.",
            ),
            sources=("repo_stage_diag", "repo_report"),
            layout="figure_focus",
            metadata={"figure_caption": "Saved rack-aisle azimuth-stage failure diagnostic"},
        ),
        SlideSpec(
            id="s17-thesis",
            section="Conclusion",
            title="The thesis is a resolution-versus-robustness map",
            takeaway="The final answer is conditional MUSIC benefit: clear in one regime, recoverable in another with better order control, and absent in the clutter-dominated regime.",
            visible_copy=(
                "Intersection: clear super-resolution value that survives waveform limitation.",
                "Open aisle: useful capability exists, but the default MDL path undercuts it.",
                "Rack aisle: clutter-dominated nuisance capture defines a hard boundary for this staged architecture.",
            ),
            visuals=(),
            speaker_notes=(
                "State the thesis exactly here and keep it tight.",
                "This is the slide where you explicitly say the project contributes a map, not a generic superiority claim.",
                "Use the three-scene structure because the audience will remember it from the verdict slide.",
            ),
            sources=("repo_report", "repo_nominal_summary", "repo_model_order"),
            layout="cards",
            metadata={
                "cards": (
                    {"title": "Favorable", "body": "Intersection retains clear MUSIC upside under the saved nominal and sweep evidence."},
                    {"title": "Recoverable", "body": "Open aisle becomes competitive once subspace dimensioning is controlled."},
                    {"title": "Hard boundary", "body": "Rack aisle remains clutter-limited even when the model order is fixed."},
                )
            },
        ),
        SlideSpec(
            id="s18-next-steps",
            section="Conclusion",
            title="What survives, what does not, and what should be done next",
            takeaway="The next research steps are not more optimistic claims; they are better order control, clutter-aware subspace handling, and broader sensing realism.",
            visible_copy=(
                "What survives: real super-resolution value in the right masked-OFDM regimes.",
                "What does not: a universal-MUSIC story under cluttered, communications-limited conditions.",
                "Next steps: masked-covariance order estimation, clutter-aware spatial processing, and limited-symbol-knowledge sensing.",
            ),
            visuals=(),
            speaker_notes=(
                "End on the measured contribution and the concrete next steps.",
                "If time allows, mention that the staged architecture itself is another obvious target for future work.",
                "Then transition to appendix backup material for questions.",
            ),
            sources=("repo_report",),
            layout="closing",
        ),
        SlideSpec(
            id="a01-metrics",
            section="Appendix",
            title="Metric definitions and active gate settings",
            takeaway="The headline metric is joint-resolution probability under a 0.35-cell threshold on all three axes after a detection gate.",
            visible_copy=(
                "P_det: both movers detected.",
                "P_joint: both movers assigned within the joint-resolution gate.",
                "Axis-specific range, Doppler, and angle probabilities are also tracked.",
            ),
            visuals=(),
            speaker_notes=(
                "Use this slide only if someone asks about evaluation fairness or thresholds.",
                "Point out that gate sensitivity was checked from saved assignments without rerunning Monte Carlo.",
            ),
            sources=("repo_report",),
            appendix=True,
            layout="table_bullets",
            metadata={
                "table_rows": (
                    ("Joint gate", "0.35 resolution cells on range, velocity, and azimuth"),
                    ("Detection NMS radius", "0.9 resolution cells"),
                    ("FFT back end", "Local matched-filter refinement"),
                    ("Model order", "MDL by default, oracle K=2 for diagnosis"),
                )
            },
        ),
        SlideSpec(
            id="a02-coherence",
            section="Appendix",
            title="Configured coherence does not explain the scene results by itself",
            takeaway="The scene comparison should be read as a composite-regime comparison, not as a pure coherence sweep.",
            visible_copy=(
                "Configured target coherence differs across scenes.",
                "Empirical finite-snapshot coherence overlaps strongly across the saved nominal trials.",
                "The scene verdicts therefore need broader regime language.",
            ),
            visuals=("story_coherence_overlap",),
            speaker_notes=(
                "This is the honesty slide if the committee pushes on mechanism claims.",
                "It narrows the story, but it does not weaken the estimator comparison.",
            ),
            sources=("repo_report",),
            appendix=True,
            layout="figure_focus",
        ),
        SlideSpec(
            id="a03-nuisance-sweep",
            section="Appendix",
            title="Expected-order nuisance sweep",
            takeaway="Once the order confound is removed, nuisance strength cleanly separates the favorable, recoverable, and failure regimes.",
            visible_copy=(
                "Open aisle stays competitive only before clutter becomes too strong.",
                "Intersection stays robust across the tested nuisance range.",
                "Rack aisle remains at zero throughout.",
            ),
            visuals=("expected_order_nuisance_sweep",),
            speaker_notes=(
                "Use this if asked how direct the clutter evidence really is.",
                "It is the cleanest slide for separating nuisance strength from order-estimation error.",
            ),
            sources=("repo_expected_nuisance_open", "repo_expected_nuisance_intersection", "repo_expected_nuisance_rack", "repo_report"),
            appendix=True,
            layout="figure_focus",
        ),
        SlideSpec(
            id="a04-runtime",
            section="Appendix",
            title="Runtime comparison at the nominal FR1 point",
            takeaway="The cost of staged MUSIC is real but not the main obstacle; regime fit matters more than the 25-30% runtime increase.",
            visible_copy=(
                "FFT nominal total runtime is about 0.76 s per point.",
                "MUSIC nominal total runtime is about 0.96 to 0.99 s per point.",
                "The stronger objection is regime dependence, not runtime alone.",
            ),
            visuals=("runtime_comparison",),
            speaker_notes=(
                "This is a good answer if someone asks whether MUSIC is simply too expensive.",
                "The slide keeps that concern in proportion.",
            ),
            sources=("repo_runtime_summary", "repo_report"),
            appendix=True,
            layout="figure_focus",
        ),
        SlideSpec(
            id="a05-derivation",
            section="Appendix",
            title="Why FBSS and staged MUSIC appear in the architecture",
            takeaway="FBSS is the coherence remedy, while the staged search is a tractable approximation rather than a true joint 3-D solver.",
            visible_copy=(
                "MUSIC evaluates a pseudospectrum using the estimated noise subspace.",
                "FBSS restores rank in the presence of coherence when a valid smoothing aperture exists.",
                "The active estimator is staged: dense azimuth, then conditioned range, then conditioned Doppler.",
            ),
            visuals=("music_pseudospectrum_equation",),
            speaker_notes=(
                "Use this slide if the questions turn mathematical.",
                "Be explicit that the architecture is staged and therefore not equivalent to joint 3-D MUSIC.",
            ),
            sources=("repo_report", "schmidt1986", "shan1985", "pillai1989"),
            appendix=True,
            layout="equation",
            metadata={
                "equation_note": "FBSS is used spatially by default, and along range/Doppler only when the support admits a valid smoothing aperture.",
            },
        ),
        SlideSpec(
            id="a06-references",
            section="Appendix",
            title="Selected references",
            takeaway="The deck rests on a small set of primary ISAC framing sources, classical MUSIC/FBSS theory, and saved internal evidence.",
            visible_copy=(
                "Liu et al. (JSAC 2022), Gonzalez-Prelcic et al. (Proc. IEEE 2024), ITU IMT-2030 materials, ETSI and Next G Alliance updates.",
                "Schmidt (1986), Shan et al. (1985), Pillai and Kwon (1989), Stoica and Nehorai (1989).",
            ),
            visuals=(),
            speaker_notes=(
                "This slide is just a compact visible bibliography; the full references live in the generated sources file.",
            ),
            sources=("liu2022", "gonzalez2024", "itu_imt2030", "schmidt1986", "shan1985", "pillai1989", "stoica1989"),
            appendix=True,
            layout="references",
        ),
    )


def get_slide_dicts() -> list[dict[str, object]]:
    """Return plain dictionaries for serialization."""

    return [asdict(slide) for slide in get_slides()]

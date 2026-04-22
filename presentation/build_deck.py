"""Build the final project presentation deck from Python sources."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from textwrap import dedent


REPO_ROOT = Path(__file__).resolve().parents[1]
PRESENTATION_ROOT = REPO_ROOT / "presentation"
OUTPUT_ROOT = REPO_ROOT / "artifacts" / "presentation"
PAYLOAD_PATH = OUTPUT_ROOT / "deck_payload.json"
OUTPUT_PPTX_PATH = OUTPUT_ROOT / "final_presentation.pptx"
OUTPUT_PDF_PATH = OUTPUT_ROOT / "final_presentation.pdf"
PREVIEW_DIR = OUTPUT_ROOT / "previews"
NARRATIVE_PLAN_PATH = OUTPUT_ROOT / "narrative_plan.md"
CLAIM_MAP_MD_PATH = OUTPUT_ROOT / "claim_map.md"
SOURCES_MD_PATH = OUTPUT_ROOT / "sources.md"


for extra_path in (REPO_ROOT, REPO_ROOT / "src"):
    extra = str(extra_path)
    if extra not in sys.path:
        sys.path.insert(0, extra)


from presentation.claim_map import get_claim_dicts
from presentation.deck_spec import get_slide_dicts
from presentation.figures import generate_figures
from presentation.sources import get_source_dicts


def _bundled_node() -> Path:
    candidate = Path.home() / ".cache" / "codex-runtimes" / "codex-primary-runtime" / "dependencies" / "node" / "bin" / "node"
    if candidate.exists():
        return candidate
    which_node = shutil.which("node")
    if which_node is None:
        raise FileNotFoundError("Could not find a Node.js runtime for deck export.")
    return Path(which_node)


def _bundled_node_modules() -> Path:
    candidate = Path.home() / ".cache" / "codex-runtimes" / "codex-primary-runtime" / "dependencies" / "node" / "node_modules"
    if not candidate.exists():
        raise FileNotFoundError("Could not find the bundled Node.js modules path required for deck export.")
    return candidate


def _ensure_local_node_modules_link() -> None:
    target = _bundled_node_modules()
    link_path = PRESENTATION_ROOT / "node_modules"
    if link_path.is_symlink():
        if Path(os.readlink(link_path)).resolve() == target.resolve():
            return
        link_path.unlink()
    elif link_path.exists():
        return
    link_path.symlink_to(target)


def _ensure_output_dirs() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)


def _write_support_documents(slides: list[dict[str, object]], claims: list[dict[str, object]], sources: list[dict[str, object]]) -> None:
    NARRATIVE_PLAN_PATH.write_text(
        dedent(
            """\
            # Narrative Plan

            Audience: UCSB EE faculty at oral-defense level.

            Objective:
            - Motivate communications-native sensing inside current ISAC research.
            - Explain why super-resolution is possible, why it becomes fragile under masked OFDM support, and what the study actually supports.
            - Defend a narrow thesis: staged MUSIC has conditional value in waveform-limited OFDM ISAC, not universal superiority.

            Narrative arc:
            1. Why ISAC and why this question matters.
            2. Clean 1-D super-resolution intuition.
            3. Masked observation model and FFT versus MUSIC framing.
            4. Study design, fairness, waveform choices, and bespoke simulation stack.
            5. Nominal verdict, representative case, regime map, order diagnosis, and rack-aisle failure mechanism.
            6. Conditional thesis, limitations, and next steps.

            Visual system:
            - Conservative academic tone with off-white background, navy text, and restrained blue/gold accents.
            - Existing scientific figures reused where they already communicate the point well.
            - New visuals limited to reproducible, evidence-backed figures and native slide geometry for process diagrams and cards.

            Editability plan:
            - Python files remain the only durable authoring surface.
            - The PPTX is a generated artifact, not a hand-edited source.
            """
        ),
        encoding="utf-8",
    )

    claim_lines = ["# Claim Map", ""]
    for claim in claims:
        evidence = ", ".join(claim["evidence"])
        claim_lines.append(f"## {claim['slide_id']} — {claim['slide_title']}")
        claim_lines.append(f"- Claim: {claim['claim']}")
        claim_lines.append(f"- Evidence: {evidence}")
        claim_lines.append("")
    CLAIM_MAP_MD_PATH.write_text("\n".join(claim_lines), encoding="utf-8")

    source_lines = ["# Sources", ""]
    for source in sources:
        source_lines.append(f"## {source['id']}")
        source_lines.append(f"- Title: {source['title']}")
        source_lines.append(f"- Citation: {source['citation']}")
        source_lines.append(f"- Kind: {source['kind']}")
        if source.get("url"):
            source_lines.append(f"- URL: {source['url']}")
        if source.get("note"):
            source_lines.append(f"- Note: {source['note']}")
        source_lines.append("")
    SOURCES_MD_PATH.write_text("\n".join(source_lines), encoding="utf-8")


def _write_payload() -> None:
    slides = get_slide_dicts()
    claims = get_claim_dicts()
    sources = get_source_dicts()
    figures = generate_figures()
    _write_support_documents(slides, claims, sources)

    payload = {
        "deck": {
            "title": "Super-resolution in waveform-limited OFDM ISAC",
            "subtitle": "Final presentation deck",
            "author": "Tony Johnson",
            "main_slide_count": sum(0 if slide["appendix"] else 1 for slide in slides),
            "total_slide_count": len(slides),
        },
        "slides": slides,
        "claims": claims,
        "sources": sources,
        "figures": figures,
    }
    PAYLOAD_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _export_pdf_if_possible() -> None:
    power_point_app = Path("/Applications/Microsoft PowerPoint.app")
    if not power_point_app.exists():
        return
    script = dedent(
        f"""\
        set inputPath to POSIX file "{OUTPUT_PPTX_PATH}"
        set outputPath to POSIX file "{OUTPUT_PDF_PATH}"
        tell application "Microsoft PowerPoint"
            activate
            open inputPath
            save active presentation in outputPath as save as PDF
            close active presentation
        end tell
        """
    )
    try:
        subprocess.run(["osascript", "-e", script], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        pass


def build() -> None:
    _ensure_output_dirs()
    _ensure_local_node_modules_link()
    _write_payload()

    node_bin = _bundled_node()
    subprocess.run(
        [
            str(node_bin),
            str(PRESENTATION_ROOT / "builder.mjs"),
            str(PAYLOAD_PATH),
            str(OUTPUT_PPTX_PATH),
            str(PREVIEW_DIR),
        ],
        cwd=PRESENTATION_ROOT,
        check=True,
    )
    if not OUTPUT_PPTX_PATH.exists():
        raise FileNotFoundError(f"Deck export did not produce {OUTPUT_PPTX_PATH}")
    _export_pdf_if_possible()


if __name__ == "__main__":
    build()

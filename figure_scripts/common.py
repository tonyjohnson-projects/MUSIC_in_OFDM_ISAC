"""Small shared helpers for the per-figure scripts."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.colors import TwoSlopeNorm
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = REPO_ROOT / "figures"
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

METHOD_ORDER = ("fft_masked", "music_masked")
METHOD_LABELS = {
    "fft_masked": "Masked FFT + Local Refinement",
    "music_masked": "Masked Staged MUSIC + FBSS",
}
METHOD_COLORS = {
    "fft_masked": "#D55E00",
    "music_masked": "#0072B2",
}
SCENE_ORDER = ("intersection", "open_aisle", "rack_aisle")
SCENE_COLORS = {
    "intersection": "#2F6B9A",
    "open_aisle": "#A65E2E",
    "rack_aisle": "#6B6B6B",
}
SWEEP_LABELS = {
    "allocation_family": "Allocation",
    "occupied_fraction": "Occupancy",
    "fragmentation": "Fragmentation",
    "bandwidth_span": "Bandwidth",
    "slow_time_span": "Slow-Time",
    "range_separation": "Range Sep.",
    "velocity_separation": "Velocity Sep.",
    "angle_separation": "Angle Sep.",
}
SUPPORT_SWEEP_ORDER = (
    "allocation_family",
    "occupied_fraction",
    "fragmentation",
    "bandwidth_span",
    "slow_time_span",
)
SEPARATION_SWEEP_ORDER = (
    "range_separation",
    "velocity_separation",
    "angle_separation",
)
STRONG_WIN_THRESHOLD = 0.10


def build_plot_parser(
    description: str,
    default_filename: str,
    *,
    default_input_root: Path | None = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    if default_input_root is not None:
        parser.add_argument(
            "--input-root",
            type=Path,
            default=default_input_root,
            help="Result root containing a data/ directory, or the data/ directory itself.",
        )
    parser.add_argument(
        "--output",
        type=Path,
        default=FIGURES_DIR / default_filename,
        help="Output PNG path.",
    )
    return parser


def resolve_data_dir(input_root: Path) -> Path:
    input_root = input_root.resolve()
    data_dir = input_root if input_root.name == "data" else input_root / "data"
    if not data_dir.is_dir():
        raise SystemExit(f"Data directory not found: {data_dir}")
    return data_dir


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def save_figure(
    fig,
    output_path: Path,
    *,
    dpi: int = 180,
    bbox_inches: str | None = None,
    pad_inches: float | None = None,
    trim: bool = False,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs: dict[str, object] = {"dpi": dpi}
    if bbox_inches is not None:
        save_kwargs["bbox_inches"] = bbox_inches
    if pad_inches is not None:
        save_kwargs["pad_inches"] = pad_inches
    fig.savefig(output_path, **save_kwargs)
    plt.close(fig)
    if trim:
        trim_figure_whitespace(output_path)
    print(f"Saved: {output_path}")


def trim_figure_whitespace(
    path: Path,
    *,
    padding_px: int = 18,
    threshold: int = 248,
) -> None:
    with Image.open(path) as image:
        rgba = image.convert("RGBA")
        pixels = np.asarray(rgba)
        content_mask = (pixels[..., 3] > 0) & np.any(pixels[..., :3] < threshold, axis=2)
        if not np.any(content_mask):
            return
        rows, cols = np.where(content_mask)
        top = max(0, int(rows.min()) - padding_px)
        bottom = min(rgba.height, int(rows.max()) + padding_px + 1)
        left = max(0, int(cols.min()) - padding_px)
        right = min(rgba.width, int(cols.max()) + padding_px + 1)
        cropped = rgba.crop((left, top, right, bottom))
        save_kwargs: dict[str, object] = {}
        if "dpi" in image.info:
            save_kwargs["dpi"] = image.info["dpi"]
        cropped.save(path, **save_kwargs)


def flatten_png_to_rgb(path: Path) -> None:
    with Image.open(path) as image:
        rgb = Image.new("RGB", image.size, "white")
        alpha = image.getchannel("A") if "A" in image.getbands() else None
        rgb.paste(image.convert("RGB"), mask=alpha)
        rgb.save(path)


def render_equation_asset(
    output_path: Path,
    lines: tuple[str, ...],
    *,
    font_size: int,
    line_gap: float = 0.34,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(
        {
            "mathtext.fontset": "cm",
            "font.family": "serif",
            "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
        }
    ):
        fig = plt.figure(
            figsize=(11.0, 1.7 + 0.42 * max(0, len(lines) - 1)),
            dpi=320,
        )
        fig.patch.set_facecolor("white")
        fig.patch.set_alpha(1.0)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.set_facecolor("white")
        ax.axis("off")
        start_y = 0.56 + 0.17 * max(0, len(lines) - 1)
        for index, line in enumerate(lines):
            ax.text(
                0.0,
                start_y - index * line_gap,
                line,
                fontsize=font_size,
                color="#10233F",
                ha="left",
                va="center",
            )
        fig.savefig(
            output_path,
            dpi=320,
            facecolor="white",
            transparent=False,
            bbox_inches="tight",
            pad_inches=0.02,
        )
        plt.close(fig)
    trim_figure_whitespace(output_path, padding_px=12, threshold=250)
    flatten_png_to_rgb(output_path)
    print(f"Saved: {output_path}")


def scene_key(scene_class: str) -> tuple[int, str]:
    if scene_class in SCENE_ORDER:
        return (SCENE_ORDER.index(scene_class), scene_class)
    return (len(SCENE_ORDER), scene_class)


def scene_label_from_rows(rows: list[dict[str, str]], scene_class: str) -> str:
    for row in rows:
        if row["scene_class"] == scene_class:
            return row.get("scene_label", scene_class.replace("_", " ").title())
    return scene_class.replace("_", " ").title()


def nominal_joint_deltas(rows: list[dict[str, str]]) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for scene_class in {row["scene_class"] for row in rows}:
        fft_row = next(
            (
                row
                for row in rows
                if row["scene_class"] == scene_class and row["method"] == "fft_masked"
            ),
            None,
        )
        music_row = next(
            (
                row
                for row in rows
                if row["scene_class"] == scene_class and row["method"] == "music_masked"
            ),
            None,
        )
        if fft_row is None or music_row is None:
            continue
        deltas[scene_class] = float(music_row["joint_resolution_probability"]) - float(
            fft_row["joint_resolution_probability"]
        )
    return deltas


def nominal_headline_rows(
    rows: list[dict[str, str]],
    *,
    method_name: str | None = None,
) -> list[dict[str, str]]:
    filtered = [
        row
        for row in rows
        if row["estimator_family"] == "headline"
        and row["sweep_name"] == "nominal"
        and row["knowledge_mode"] == "known_symbols"
    ]
    if method_name is not None:
        filtered = [row for row in filtered if row["method"] == method_name]
    return filtered


def contrast_text_effects(text_color: str) -> list[patheffects.AbstractPathEffect]:
    outline = "#111111" if text_color == "white" else "#FFFFFF"
    return [patheffects.withStroke(linewidth=1.6, foreground=outline)]

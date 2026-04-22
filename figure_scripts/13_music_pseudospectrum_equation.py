#!/usr/bin/env python3
"""Generate figures/13_music_pseudospectrum_equation.png."""

from __future__ import annotations

from common import build_plot_parser, render_equation_asset


def make_figure(output_path) -> None:
    render_equation_asset(
        output_path,
        (r"$P_{\mathrm{MUSIC}}(\phi)=\frac{1}{\left\|E_n^H a(\phi)\right\|_2^2}$",),
        font_size=38,
        line_gap=0.0,
    )


def main() -> None:
    parser = build_plot_parser(
        "Generate the MUSIC pseudospectrum equation figure.",
        "13_music_pseudospectrum_equation.png",
    )
    args = parser.parse_args()
    make_figure(args.output)


if __name__ == "__main__":
    main()

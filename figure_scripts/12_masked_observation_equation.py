#!/usr/bin/env python3
"""Generate figures/12_masked_observation_equation.png."""

from __future__ import annotations

from common import build_plot_parser, render_equation_asset


def make_figure(output_path) -> None:
    render_equation_asset(
        output_path,
        (
            r"$y_{h,k,n}=m_{k,n}x_{k,n}\sum_{p=1}^{P}\alpha_p a_h(\theta_p)$",
            r"$\exp(-j2\pi f_k\tau_p)\,\exp(j2\pi\nu_p t_n)\,s_p[n]+w_{h,k,n}$",
        ),
        font_size=32,
    )


def main() -> None:
    parser = build_plot_parser(
        "Generate the masked-observation equation figure.",
        "12_masked_observation_equation.png",
    )
    args = parser.parse_args()
    make_figure(args.output)


if __name__ == "__main__":
    main()

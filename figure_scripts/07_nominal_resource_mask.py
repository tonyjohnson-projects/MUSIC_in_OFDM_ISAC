#!/usr/bin/env python3
"""Generate figures/07_nominal_resource_mask.png."""

from __future__ import annotations

from common import build_plot_parser, matplotlib, np, plt, save_figure, trim_figure_whitespace
from aisle_isac.resource_grid import build_resource_grid


def make_figure(output_path) -> None:
    grid = build_resource_grid(
        "fragmented_prb",
        96,
        16,
        prb_size=12,
        n_prb_fragments=4,
        pilot_subcarrier_period=4,
        pilot_symbol_period=4,
    )

    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    role_colors = {0: "#C9D1DB", 1: "#4E79A7", 2: "#D55E00", 3: "#8E6C8A"}
    role_labels = {0: "Muted", 1: "Pilot", 2: "Data", 3: "Punctured"}
    unique_roles = sorted(int(value) for value in np.unique(grid.role_grid))
    cmap = matplotlib.colors.ListedColormap([role_colors[role] for role in unique_roles])
    bounds = np.arange(len(unique_roles) + 1) - 0.5
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    role_to_index = {role: index for index, role in enumerate(unique_roles)}
    role_index_grid = np.vectorize(role_to_index.get)(grid.role_grid)

    image = ax.imshow(
        role_index_grid.T,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )
    ax.set_xlabel("Simulated subcarrier index")
    ax.set_ylabel("Slow-time snapshot")
    ax.set_title("Nominal fragmented scheduled PRB mask", loc="left", fontsize=14, fontweight="bold")
    ax.set_xticks(np.arange(-0.5, grid.role_grid.shape[0], 12), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.role_grid.shape[1], 4), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8, alpha=0.55)
    ax.tick_params(which="minor", bottom=False, left=False)
    colorbar = fig.colorbar(image, ax=ax, ticks=np.arange(len(unique_roles)), fraction=0.046, pad=0.03)
    colorbar.ax.set_yticklabels([role_labels[role] for role in unique_roles])
    fig.tight_layout()
    save_figure(fig, output_path)
    trim_figure_whitespace(output_path)


def main() -> None:
    parser = build_plot_parser(
        "Generate the nominal resource-mask figure.",
        "07_nominal_resource_mask.png",
    )
    args = parser.parse_args()
    make_figure(args.output)


if __name__ == "__main__":
    main()

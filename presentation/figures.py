"""Compatibility wrapper around the unified figure-generation script."""

from __future__ import annotations

from generate_figures import generate_figures


__all__ = ["generate_figures"]


if __name__ == "__main__":
    generate_figures()

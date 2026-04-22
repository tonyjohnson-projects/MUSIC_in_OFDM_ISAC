#!/usr/bin/env python3
"""Compatibility wrapper for the unified figure-generation entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generate_figures import legacy_motivation_main


if __name__ == "__main__":
    raise SystemExit(legacy_motivation_main())

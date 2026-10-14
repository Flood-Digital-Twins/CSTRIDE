#!/usr/bin/env python3
"""Run the C-STRIDE forecast latents workflow from a source checkout."""
import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
    runpy.run_module("cstride.forecast_latents", run_name="__main__")

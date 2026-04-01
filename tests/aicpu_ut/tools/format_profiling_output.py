#!/usr/bin/env python3
import runpy
from pathlib import Path

TARGET = Path(__file__).resolve().parent / "profiling" / "format_profiling_output.py"
runpy.run_path(str(TARGET), run_name="__main__")

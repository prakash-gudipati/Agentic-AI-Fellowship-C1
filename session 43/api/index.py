"""Vercel entry point — exposes the FastAPI app to the @vercel/python runtime."""
import os
import sys
from pathlib import Path

# Put the project root on the path so `main` is importable on Vercel.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from main import app  # noqa: E402,F401

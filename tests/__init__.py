"""Test package: makes the src layout importable without installation.

Author: Olshansky Grigory
"""
import sys
from pathlib import Path

__author__ = "Olshansky Grigory"

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

"""
A simple OptKing wrapper for the Psi4 quantum chemistry program
"""

import numpy
import pytest

# Try to pull in Psi4
try:
    import psi4

    found_psi4 = True
except ImportError:
    found_psi4 = False

# Wrap Psi4 in ifden
using_psi4 = pytest.mark.skipif(found_psi4, reason="Psi4 not found, skipping.")

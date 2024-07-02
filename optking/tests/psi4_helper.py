"""
A simple OptKing wrapper for the Psi4 quantum chemistry program
"""

import numpy
import pytest
from qcelemental.util import which_import

# Try to pull in Psi4
try:
    import psi4

    found_psi4 = True
except ImportError:
    found_psi4 = False

# Wrap Psi4 in ifden
using_psi4 = pytest.mark.skipif(found_psi4, reason="Psi4 not found, skipping.")
using_qcmanybody = pytest.mark.skipif(
    which_import("qcmanybody", return_bool=True) is False,
    reason="cound not find qcmanybody. please install the package to enable tests",
)

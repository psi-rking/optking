"""Regression tests for issue #115: a near-0 interior angle in a torsion is a reset, not a linear bend, and the search for torsions past a
collinear segment must not loop forever when the LINEAR bends in the coordinate set form a cycle. Pure python (no QC engine)."""
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import optking
from optking import addIntcos, bend, v3d
from optking.exceptions import AlgError

PHI_LIM = 0.017  # rad, the order of v3d_tors_angle_lim


def test_near_zero_interior_angle_resets_instead_of_adding_linear_bends():
    with pytest.raises(AlgError) as e:
        v3d.linear_torsion_check(0.005, 1.9, PHI_LIM, [0, 15, 2, 3])
    assert e.value.back_transformation is True
    assert not e.value.linear_bends


def test_near_180_interior_angle_still_reports_linear_bends():
    with pytest.raises(AlgError) as e:
        v3d.linear_torsion_check(np.pi - 0.005, 1.9, PHI_LIM, [0, 1, 2, 3])
    assert e.value.back_transformation is False
    assert e.value.linear_bends


CYCLE_SCRIPT = textwrap.dedent(
    """
    import numpy as np
    from optking import addIntcos, bend
    from optking.exceptions import AlgError
    # four atoms on a line, bonded 0-1, 1-2, 2-3; the regular bend 1-2-3 is absent (collinear segment 1-2-3, atom 2 with two
    # bonds), and the coordinate set carries LINEAR bends 0-1-3 and 1-0-3 -- the permuted pair that linear_torsion_check used to add.
    geom = np.array([[0.0, 0, 0], [1.06, 0, 0], [2.27, 0, 0], [3.7, 0, 0]])
    C = np.zeros((4, 4), bool)
    for a, b in ((0, 1), (1, 2), (2, 3)):
        C[a, b] = C[b, a] = True
    intcos = [bend.Bend(0, 1, 3, bend_type="LINEAR"), bend.Bend(1, 0, 3, bend_type="LINEAR")]
    try:
        addIntcos.add_tors_from_connectivity(C, intcos, geom)
    except AlgError as e:
        print("ALGERROR", e.back_transformation)
    else:
        print("RETURNED")
    """
)


def test_linear_bend_cycle_raises_instead_of_looping():
    # run in a subprocess with a timeout, so that a regression (the former infinite loop) fails the test instead of hanging it
    r = subprocess.run([sys.executable, "-c", CYCLE_SCRIPT], capture_output=True, text=True, timeout=60)
    assert "ALGERROR True" in r.stdout, r.stdout + r.stderr

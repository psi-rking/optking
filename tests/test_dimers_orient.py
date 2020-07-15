import optking
import pytest

# Tests the dimer orientation code.
# 1) Create random geometries for fragments with NA and NB atoms, respectively.
# 2) Randomly choose 1-3 atoms on each fragment with which to define the
# reference points on each fragment.  Assign them random weights.
# The linear combination prescribes each reference point.
# 3) Choose an arbitrary displacement of the interfragment coordinates, rotate
# the geometry of second fragment to match precisely the new values.
# 4) If number of atoms in a fragment is <3, then tests code for all the
# annoying situations in which some of the 6 interfragment coordinates drop out.

@pytest.mark.dimers
@pytest.mark.parametrize("NA,NB", [(i,j) for i in range(1,5) for j in range(1,5)])
def test_dimerfrag_orient(NA, NB):
    rms_error = optking.dimerfrag.test_orient(NA, NB)
    print('Error: {:10.5e}'.format(rms_error))
    assert rms_error < 1.0e-10


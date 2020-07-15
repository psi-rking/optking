import psi4
import optking
import numpy as np
import pytest

# Demonstrate and test positioning two water molecules by specifying
# their interfragment reference points and coordinates.
@pytest.mark.dimers
def test_dimers_bmat():
    h2oA = psi4.geometry("""
         O
         H 1 1.0
         H 1 1.0 2 104.5
    """)
    Axyz = h2oA.geometry().np
    h2oB = psi4.geometry("""
         O
         H 1 1.0
         H 1 1.0 2 104.5
    """)
    Bxyz = Axyz.copy() + 5.0 # Move B not right on top of A.
    RefAtomsA = [ [0],[1,2],[2] ]
    RefAtomsB = [ [0],[1,2],[2] ]
    Itest = optking.dimerfrag.DimerFrag(0, RefAtomsA, 1, RefAtomsB,
                A_lbl="Water A", B_lbl="Water B")
    max_error = Itest.test_B(Axyz,Bxyz)
    print('Max error in positioning water dimer: {:8.3e}'.format(max_error))
    assert max_error < 1.0e-9


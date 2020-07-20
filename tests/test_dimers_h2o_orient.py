import psi4
import optking
import numpy as np
import pytest

# Demonstrate and test positioning two water molecules by specifying
# their interfragment reference points and coordinates.
@pytest.mark.dimers
def test_dimerfrag_orient_h2o_dimers():
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

    # Choose some reference atoms for each fragment.
    # The numbering starts at zero.  These would be
    # the O atom, the point between the two H atoms,
    # and one of the hydrogen atoms for each fragment.
    RefAtomsA = [ [0],[1,2],[2] ]
    RefAtomsB = [ [0],[1,2],[2] ]

    # The default weights are equal between involved atoms but 
    # may be specified.  Fragment labels are optional.
    Itest = optking.dimerfrag.DimerFrag(0, RefAtomsA, 1, RefAtomsB,
                A_lbl="Water A", B_lbl="Water B")
    Itest.update_reference_geometry(Axyz,Bxyz)
    #print(Itest)

    # Create arbitrary target for displacement with illustrative names.
    R_A1B1       =  3.4
    theta_A2A1B1 =  2.5
    theta_A1B1B2 =  2.7
    tau_A2A1B1B2 = -1.5
    phi_A3A2A1B1 =  0.3
    phi_A1B1B2B3 =  0.6

    q_target = np.array( [R_A1B1, theta_A2A1B1, theta_A1B1B2,
                tau_A2A1B1B2, phi_A3A2A1B1, phi_A1B1B2B3] )
    Bxyz_new = Itest.orient_fragment(Axyz, Bxyz, q_target)

    # Test how we did
    Itest.update_reference_geometry(Axyz, Bxyz_new)
    rms_error = np.sqrt( np.mean((q_target - Itest.q_array())**2) )
    print('RMS deviation from target interfragment coordinates: {:8.3e}'.format(rms_error))
    assert rms_error < 1.0e-10

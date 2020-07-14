import psi4
import optking
import pytest

# Tests the dimer orientation code.
# 1) Create random geometries for fragments with NA and NB atoms, respectively.
# 2) Randomly choose 1-3 atoms on each fragment with which to define the
# reference points on each fragment.  Assign them random weights.
# The linear combination prescribes each reference point.
# 3) Choose an arbitrary displacement of the interfragment coordinates, rotate
# the geometry of each fragment to match precisely the new values.
# 4) If number of atoms in a fragment is <3, then tests code for all the
# annoying situations in which some of the 6 interfragment coordinates drop out.

@pytest.mark.parametrize("NA,NB", [(i,j) for i in range(1,5) for j in range(1,5)])
def test_dimerfrag_orient(NA, NB):
    error = optking.dimerfrag.test_orient(NA, NB)
    assert error < 1.0e-10


"""
def test_dimers():
    h2oA = psi4.geometry(
         O
         H 1 1.0
         H 1 1.0 2 104.5
    )
    xyzA = h2oA.geometry().np
    print(xyzA)

    h2oB = psi4.geometry(
         O
         H 1 1.0
         H 1 1.0 2 104.5
    )
    # Change COM to other location
    COMx, COMy, COMz = 1.0, 4.0, 4.0
    xyzB = xyzA.copy()
    for atom in range( xyzB.shape[0] ):
        xyzB[atom,:] += np.array( [COMx, COMy, COMz] )
    print(xyzB)
    h2oB.set_geometry( psi4.core.Matrix.from_array(xyzB) )

    quit()

    psi4.set_options(psi4_options)
    json_output = optking.optimize_psi4('hf')
    E = json_output['energies'][-1]
    print(E)

    #psi4.core.clean_options()
    #psi4_options = {
    #  'diis': False,
    #  'basis': 'dz',
    #  'scf_type': 'pk',
    #}
    #nucenergy = json_output['trajectory'][-1]['properties']['nuclear_repulsion_energy']
    #refnucenergy =   8.9064983474
    #refenergy    = -74.9659011923
    #assert psi4.compare_values(refnucenergy, nucenergy, 3, "Nuclear repulsion energy")
    #assert psi4.compare_values(refenergy, E, 6, "Reference energy")
test_dimers()
"""


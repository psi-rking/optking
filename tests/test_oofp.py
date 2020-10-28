#! Test out-of-plane angles.
#  Problem for a situation like this, very nearly planar formaldehyde
# the three bends become redundant as one approaches planarity.  So
# the eigenvalues of (B*B^t) will contain a very small eval like 10^-8.
# If you include this in your matrix inversion by taking its reciprocal,
# it blows up, and if you don't include it you can't tightly reach planarity.
#    form = psi4.geometry("""
#       O      0.6   -0.00007   0.0
#       C     -0.6   -0.00007   0.0
#       H     -1.2    0.24    -0.9
#       H     -1.2   -0.24     0.9
#       symmetry c1
#    """)

import psi4
import optking

b3lyp_ref_energy = -114.41550257

def test_oofp_formaldehyde():
    form = psi4.geometry("""
       O      0.6   -0.00007   0.0
       C     -0.6   -0.00007   0.0
       H     -1.2    0.24    -0.9
       H     -1.2   -0.24     0.9
       symmetry c1
    """)
    psi4_options = {
      'basis': 'def2-SVP',
      'g_convergence': 'gau_tight',
      'test_B' : True,
    }
    psi4.set_options(psi4_options)

    result = optking.optimize_psi4('b3lyp')
    E = result['energies'][-1] #TEST

    assert psi4.compare_values(b3lyp_ref_energy, E, 8, "B3LYP energy")




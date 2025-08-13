# Constrained optimization of tyrosine side group with methylthiophene.
# The relative orientation of the two monomers is fixed.  The monomers
# themselves and the distance between them are optimized.
import psi4
import optking

# import numpy as np
import pytest

# import qcelemental as qcel
# import qcengine as qcng
from qcengine.testing import has_program


@pytest.mark.long
@pytest.mark.dimers
@pytest.mark.dftd3
@pytest.mark.skipif(
    (has_program("dftd3") or has_program("s-dftd3")) is False,
    reason="Neither DFTD3 nor s-DFTD3 is findable",
)
def test_dimers_mt_tyr_frozen_orientation():
    # Starting at R ~ 5 Angstroms
    # To shorten this test the geometry of the second fragment has been preoptimized
    init_xyz = """
        C       -1.258686      0.546935      0.436840
        H       -0.683650      1.200389      1.102833
        C       -0.699036     -0.349093     -0.396608
        C       -2.693370      0.550414      0.355311
        H       -3.336987      1.206824      0.952052
        C       -3.159324     -0.343127     -0.536418
        H       -4.199699     -0.558111     -0.805894
        S       -1.883829     -1.212288     -1.301525
        C        0.786082     -0.656530     -0.606057
        H        1.387673     -0.016033      0.048976
        H        1.054892     -0.465272     -1.651226
        H        0.978834     -1.708370     -0.365860
        --
        C       -6.562836      0.088927     -1.511277
        C       -6.594269      0.025253      1.308215
        C       -6.676671      1.271056     -0.779037
        C       -6.505626     -1.131881     -0.829734
        C       -6.520887     -1.158260      0.559346
        C       -6.682066      1.235842      0.615561
        H       -6.740534      2.224314     -1.293107
        H       -6.436728     -2.040745     -1.412676
        H       -6.459292     -2.110657      1.073787
        H       -6.755418      2.164270      1.170820
        O       -6.515574      0.041680     -2.895823
        H       -6.529678      0.962018     -3.265664
        C       -6.546283     -0.014213      2.822460
        H       -7.184314     -0.813172      3.215202
        H       -6.883394      0.935506      3.248925
        H       -5.521759     -0.199035      3.169367
        nocom
        unit angstrom
    """
    # Note that nocom is needed so psi4 does not move the fragment COM's.

    # Define the reference atoms for each fragment, as a linear combination
    # of the positions of one or more atoms of the fragment.
    # If no weights are given, then the atoms are equally weighted.
    MTdimer = {
        "Natoms per frag": [12, 16],
        "A Frag": 1,
        "A Ref Atoms": [[1, 3, 4, 6, 8], [8], [11]],
        "A Label": "methylthiophene",
        "B Frag": 2,
        "B Ref Atoms": [[13, 14, 15, 16, 17, 18], [13], [15]],
        "B Label": "tyrosine",
        "Frozen": ["theta_A", "theta_B", "tau", "phi_A", "phi_B"],
    }
    # Here are the dimer coordinates that are used with their definitions.
    #  R         : Distance A1 to B1
    #  theta_A   : Angle,          A2-A1-B1
    #  theta_B   : Angle,          A1-B1-B2
    #  tau       : Dihedral angle, A2-A1-B1-B2
    #  phi_A     : Dihedral angle, A3-A2-A1-B1
    #  phi_B     : Dihedral angle, A1-B1-B2-B3

    # mTmol = qcel.models.Molecule.from_data(init_xyz)
    # Build the psi4 molecule.
    MTmol = psi4.geometry(init_xyz)

    # To see the values of the interfragment coordinates, do this:
    # MTdimerCoord = optking.dimerfrag.DimerFrag.fromUserDict(MTdimer)
    # Axyz = MTmol.geometry().np[0:12,]
    # Bxyz = MTmol.geometry().np[12:,]
    # MTdimerCoord.update_reference_geometry(Axyz, Bxyz)
    # print( MTdimerCoord )
    # quit()

    # input_data = {"input_specification": {
    #                   "model": {
    #                       "method": "B3LYP-d3mbj",
    #                       "basis": "6-31G(d)"},
    #                   "keywords": {"d_convergence": 9}
    #                   },
    #               "keywords": {
    #                  "frag_mode": "multi",
    #                  "interfrag_coords": str(MTdimer)},
    #               "initial_molecule": mTmol
    #              }

    # opt_input = qcel.models.OptimizationInput(**input_data)
    # result = qcng.compute_procedure(opt_input, "optking")

    # For the moment, 'interfrag_coords' is a non-standard keyword and so
    # must be passed like this.
    # Optimize fragments and R but not interfragment angular coordinates.
    # psi4.set_options({"d_convergence": 9, "basis": "6-31G(d)", "interfrag_coords": str(MTdimer)})
    # result = optking.optimize_psi4("b3lyp-d3mbj")
    psi4.set_options({"d_convergence": 9, "basis": "3-21G"})
    E = psi4.optimize("b3lyp-d3mbj", optimizer_keywords={"interfrag_coords": str(MTdimer)})
    # result = optking.optimize_psi4("b3lyp-d3mbj", **{"interfrag_coords": str(MTdimer)})
    # E = result["energies"][-1]

    # REF_631Gd_Energy = -939.169521
    REF_321G_Energy = -934.237170
    assert psi4.compare_values(REF_321G_Energy, E, 4, "B3LYP-D3MBJ energy")  # TEST

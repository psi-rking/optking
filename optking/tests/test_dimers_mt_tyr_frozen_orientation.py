# Constrained optimization of tyrosine side group with methylthiophene.
# The relative orientation of the two monomers is fixed.  The monomers
# themselves and the distance between them are optimized.
import psi4
import optking
import numpy as np
import pytest


@pytest.mark.long
@pytest.mark.dimers
def test_dimers_mt_tyr_frozen_orientation():
    # Starting at R ~ 5 Angstroms
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
     C       -6.955593     -0.119764     -1.395442
     C       -6.977905     -0.135060      1.376787
     C       -7.111625      1.067403     -0.697024
     C       -6.810717     -1.314577     -0.707746
     C       -6.821873     -1.322226      0.678369
     C       -7.122781      1.059754      0.689090
     H       -7.226173      2.012097     -1.240759
     H       -6.687348     -2.253224     -1.259958
     H       -6.707325     -2.266920      1.222105
     H       -7.246150      1.998400      1.241304
     O       -6.944245     -0.111984     -2.805375
     H       -7.058224      0.807436     -3.049180
     C       -6.990227     -0.143507      2.907714
     H       -8.018305     -0.274985      3.264065
     H       -6.592753      0.807024      3.281508
     H       -6.368443     -0.968607      3.273516
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

    # Build the psi4 molecule.
    MTmol = psi4.geometry(init_xyz)

    # To see the values of the interfragment coordinates, do this:
    # MTdimerCoord = optking.dimerfrag.DimerFrag.fromUserDict(MTdimer)
    # Axyz = MTmol.geometry().np[0:12,]
    # Bxyz = MTmol.geometry().np[12:,]
    # MTdimerCoord.update_reference_geometry(Axyz, Bxyz)
    # print( MTdimerCoord )
    # quit()

    # Choose a theory
    psi4.set_memory("4000.0 MB")
    psi4.core.clean_options()
    psi4_options = {
        "basis": "6-31G(d)",
        "d_convergence": 9,
        "frag_mode": "multi",
    }
    psi4.set_options(psi4_options)

    # For the moment, 'interfrag_coords' is a non-standard keyword and so
    # must be passed like this.
    # Optimize fragments and R but not interfragment angular coordinates.
    result = optking.optimize_psi4("b3lyp-d3mbj", xtra_opt_params={"interfrag_coords": str(MTdimer)})

    E = result["energies"][-1]

    REF_631Gd_Energy = -939.169521
    REF_321G_Energy = -934.237170
    assert psi4.compare_values(REF_631Gd_Energy, E, 4, "B3LYP-D3MBJ energy")  # TEST

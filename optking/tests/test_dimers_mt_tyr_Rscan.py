# PES scan of tyrosine side group with methylthiophene.
# The fragments and the relative orientation of the two molecules is frozen.
# The distance between the center of the rings of the monomers is varied.
import psi4
import optking
import numpy as np
import pytest
import qcelemental as qcel

au2kcal = qcel.constants.hartree2kcalmol


@pytest.mark.dimers
def test_dimers_mt_tyr_Rscan():
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
    dimer = {
        "Natoms per frag": [12, 16],
        "A Frag": 1,
        "A Ref Atoms": [
            [1, 3, 4, 6, 8],  # Average of position of ring atoms
            [8],  # S, shouldn't matter much which chosen
            [11],
        ],  # H, best to avoid a collinear A3-A2-A1 angle,
        # also, avoid phi_B ~180 deg
        "A Label": "methylthiophene",  # labels are optional
        "B Frag": 2,
        "B Ref Atoms": [[13, 14, 15, 16, 17, 18], [13], [15]],  # Average of ring C's  # one C  # another C
        "B label": "tyrosine",  # optional
    }

    # Here are the dimer coordinates that are used with their definitions.
    #  R         : Distance A1 to B1
    #  theta_A   : Angle,          A2-A1-B1
    #  theta_B   : Angle,          A1-B1-B2
    #  tau       : Dihedral angle, A2-A1-B1-B2
    #  phi_A     : Dihedral angle, A3-A2-A1-B1
    #  phi_B     : Dihedral angle, A1-B1-B2-B3

    dimerCoord = optking.dimerfrag.DimerFrag.fromUserDict(dimer)

    # Build the psi4 molecule.
    MT_mol = psi4.geometry(init_xyz)

    # Use the psi4 molecule to provide the fragment geometries.
    Axyz = MT_mol.geometry().np[
        0:12,
    ]
    Bxyz = MT_mol.geometry().np[
        12:,
    ]

    # To see the values of the interfragment coordinates, do this:
    # dimerCoord.update_reference_geometry(Axyz, Bxyz)
    ## in au and radians:
    # print( dimerCoord.q() )
    ## in Angstroms and degrees:
    # print( dimerCoord.q_show() )
    # quit()

    # Here are the dimer coordinates defined in Ang and deg.
    # We will only be varying the R one, distance between the ring centers.
    (R, theta_A, theta_B, tau, phi_A, phi_B) = [5.03575, 94.64, 86.36, -46.50, 167.38, -97.60]

    # Choose a theory.
    psi4.set_memory("4000.0 MB")
    psi4.core.clean_options()
    psi4_options = {"basis": "6-31G(d)", "d_convergence": "9"}
    psi4.set_options(psi4_options)

    # Vary R distance between the 1st reference point on each fragment.
    # Default units are au and radians; we'll use Angstroms and degrees.
    E_Rab = []
    for R_AB in np.arange(4.0, 8.1, 1.0):
        q_target = np.array([R_AB, theta_A, theta_B, tau, phi_A, phi_B])
        # This function returns the geometry of the second fragment.
        Bxyz[:] = dimerCoord.orient_fragment(Axyz, Bxyz, q_target, unit_angle="deg", unit_length="Angstrom")
        xyz = psi4.core.Matrix.from_array(np.concatenate((Axyz, Bxyz)))
        # Put the geometry into psi4, compute energy, save results.
        MT_mol.set_geometry(xyz)
        E_Rab.append([R_AB, xyz.to_array(), psi4.energy("b3lyp-d3mbj")])

    # Convert the energies to relative energies in kcal/mol.
    E_Rab = np.array(E_Rab, dtype=object)
    Emin = min(E_Rab[:, 2])
    E_Rab[:, 2] = au2kcal * (E_Rab[:, 2] - Emin)

    print("Relative energy (kcal/mol) vs. R")
    for pts in E_Rab:
        print("{:10.3f}  {:10.3f}".format(pts[0], pts[2]))

    # """ To plot the energy scan, do this:
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_axes( [0.10,0.10,0.80,0.80] )
    # ax.set_title('tyrosine-methylthiophene dimer')
    # ax.set_xlabel('R')
    # ax.set_ylabel('Energy')
    # ax.plot( E_Rab[:,0], E_Rab[:,2], 'g+-')
    # plt.xlim(4,10)
    # plt.show()
    # """
    # To get the geometry for a certain value of R, e.g., 5.0 Angstroms:
    # row = np.where(E_Rab[:,0]==5.0)[0][0]
    # MT_mol.set_geometry( psi4.core.Matrix.from_array(E_Rab[row,1]) )
    # MT_mol.print_out()

    REF_Energy_kcal_60 = 2.029
    row = np.where(E_Rab[:, 0] == 6.0)[0][0]
    assert psi4.compare_values(REF_Energy_kcal_60, E_Rab[row, 2], 3)


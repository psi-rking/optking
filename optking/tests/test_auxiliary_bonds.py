#! Test of adding auxiliary bonds. These are stre coordinates between
#! two non-H atoms that are < 2.5(sum of covalent radii) but NOT already
#! connected by 3 stretches as in A-x-y-B.

# Here are number of iterations to convergence as of Jan 2022 for 3 molecules
# in Baker database:
#     menthone                 ACHTAR10                 histidine
# 2 bonds added            1 bond added             1 bond added
# RHF/6-31+G*   11 -> 10   RHF/6-31+G*   13 -> 11   RHF/6-31+G*   15 -> 16
# B3LYP/6-31+G* 10 ->  9   B3LYP/6-31+G* 12 -> 11   B3LYP/6-31+G* 14 -> 14
# TODO: explore FISCHER Hessian guess alongside auxiliary bonds performance

import pathlib
import psi4
import optking
import pytest
from .utils import utils

menthone = """
0 1
O        0.00000000     0.00000000     4.83502957
C       -5.06597212    -1.27592091     0.49885049
C       -3.60348796    -1.49111229    -2.01995066
C       -1.13779972     0.12182250    -2.02402508
C        0.69335828    -0.53324847     0.24699141
C       -0.81879368    -0.76420189     2.79442766
C       -3.41755812    -2.06413311     2.77868746
C        0.08327139    -0.18247958    -4.66184769
C        3.16977849     1.11788425     0.33780916
C        5.23967937     0.00000000     2.05851212
C        2.74820737     3.91648659     1.03692914
H       -5.73534045     0.69223903     0.75660810
H       -6.80139535    -2.44289264     0.43045930
H       -3.15419510    -3.50140339    -2.39715388
H       -4.86109777    -0.90264082    -3.58603040
H       -1.71463208     2.12647811    -1.82542348
H        1.33530286    -2.48925976    -0.10949068
H       -4.41049264    -1.64891601     4.56938165
H       -3.10227312    -4.12767303     2.77142958
H       -1.27515064     0.19340176    -6.20625544
H        0.83979297    -2.10810531    -4.96157719
H        1.65711962     1.15531285    -4.96195049
H        4.01314574     1.10167735    -1.57473542
H        4.69908810     0.02990650     4.07747056
H        7.03475689     1.05859686     1.90111311
H        5.66887645    -1.98486988     1.56898286
H        4.52277834     5.01677786     0.95132487
H        1.98900684     4.13531008     2.97264568
H        1.40402606     4.85096335    -0.25821233
units bohr
"""

ACHTAR10 = """
0 1
O        0.00000000     0.00000000     3.93735249
O        1.79875939     0.00000000    -0.09531034
N       -4.40589519     1.32037243    -3.31810156
C       -2.43021636    -0.18962157    -2.05696026
C       -0.22185404     1.49597798    -1.20775357
C        1.69726730    -0.59259412     2.46067577
C        3.97685548    -2.11479138     3.27934906
H       -3.68043380     2.27933244    -4.84082518
H       -5.10144333     2.68085421    -2.12147722
H       -3.24985392    -1.18842676    -0.41051393
H       -1.74547418    -1.68142667    -3.35347133
H        0.55351430     2.51912058    -2.85842920
H       -0.88071695     2.99188292     0.10524925
H        5.73529679    -1.04410557     2.94759034
H        4.08562680    -3.90736002     2.21955987
H        3.86856770    -2.56921447     5.31306580
units bohr
"""

aux_bonds = [(menthone, [[1, 10], [1, 11]]), (ACHTAR10, [[1, 4]])]

HF_expected_noaux = {'menthone': 11, 'ACHTAR10': 13}
HF_expected_aux = {'menthone': 10, 'ACHTAR10': 11}
HF_E = {'menthone':  -464.0439981504222, 'ACHTAR10':  -360.90442278650494}

B3LYP_expected_noaux = {'menthone': 10, 'ACHTAR10': 12}
B3LYP_expected_aux = {'menthone': 9, 'ACHTAR10': 11}
B3LYP_E = {'menthone': -467.157103348465, 'ACHTAR10': -363.065807664032}

@pytest.mark.long
@pytest.mark.skip("Tests are long. Replaced with simpler, more direct tests but left in for now.")
def test_auxiliary_bonds(check_iter):
    for molname in ['menthone', 'ACHTAR10']:
        psi4.core.set_active_molecule(eval(molname))

        psi4.core.clean_options()
        psi4.set_options({ "basis": "6-31+G*", })

        result = optking.optimize_psi4("HF")
        utils.compare_iterations(result, HF_expected_noaux[molname], check_iter)
        E = result["energies"][-1]
        assert psi4.compare_values(HF_E[molname], E, 5, "HF energy")

        psi4.core.set_active_molecule(eval(molname))
        psi4.core.clean_options()

        psi4.set_options({ "basis": "6-31+G*", })
        optking_options = {"add_auxiliary_bonds": True}

        result = optking.optimize_psi4("HF", **optking_options)
        utils.compare_iterations(result, HF_expected_aux[molname], check_iter)
        E = result["energies"][-1]
        assert psi4.compare_values(HF_E[molname], E, 5, "HF energy")


@pytest.mark.parametrize("molecule, aux_indices", aux_bonds)
def test_add_aux_bonds(molecule, aux_indices):
    """ Check that auxiliary bonds can be added for molecules from baker tests """
    for mol_name in ['menthone', 'ACHTAR10']:

        psi4.core.clean_options()
        mol = psi4.geometry(molecule)

        params = optking.optwrapper.initialize_options({"add_auxiliary_bonds": True})
        opt_molsys, _ = optking.molsys.Molsys.from_psi4(mol)
        optking.make_internal_coords(opt_molsys, params)

        opt_molsys.print_intcos()
        all_intcos = [intco for frag in opt_molsys.fragments for intco in frag.intcos]
        for bond in aux_indices:
            zero_indexed = [val - 1 for val in bond]
            assert optking.stre.Stre(*zero_indexed) in all_intcos

        del opt_molsys
        del params
        params = optking.optwrapper.initialize_options({"add_auxiliary_bonds": False})
        opt_molsys, _ = optking.molsys.Molsys.from_psi4(mol)
        optking.make_internal_coords(opt_molsys, params)

        opt_molsys.print_intcos()
        all_intcos = [intco for frag in opt_molsys.fragments for intco in frag.intcos]
        for bond in aux_indices:
            zero_indexed = [val - 1 for val in bond]
            assert optking.stre.Stre(*zero_indexed) not in all_intcos


def test_aux_opt(check_iter):
    """ Perform a single optimization with auxiliary bonds.
    I've added a hessian to try and speedup the opt """

    psi4.geometry(ACHTAR10)
    psi4.set_options({
                "basis": "6-31+G*",
                "add_auxiliary_bonds": True,
                "cart_hess_read": True,
    })

    test_dir = pathlib.Path(__file__).parent
    result = optking.optimize_psi4("HF", **{"hessian_file": f"{test_dir}/test_data/C4H9NO2.hess"})
    utils.compare_iterations(result, 8, check_iter)
    E = result["energies"][-1]
    assert psi4.compare_values(HF_E["ACHTAR10"], E, 5, "HF energy")

import psi4
import optking
import json
from .utils import utils

#! SCS-OMP2 cc-pVDZ geometry optimization for the H2O molecule.
def test_scsmp2_opt(check_iter):

    h2o = psi4.geometry(
        """
        0 1
        o
        h 1 0.958
        h 1 0.958 2 104.4776
    """
    )

    psi4.core.clean_options()
    psi4_options = {"basis": "cc-pvdz", "max_energy_g_convergence": 7}
    psi4.set_options(psi4_options)

    result = optking.optimize_psi4("scs-omp2")

    this_nuc = result["trajectory"][-1]["properties"]["nuclear_repulsion_energy"]  # TEST
    this_scf = result["trajectory"][-1]["extras"]["qcvars"]["SCF TOTAL ENERGY"]  # TEST
    this_energy = result["energies"][-1]  # TEST
    REF_nuc = 9.1123208123
    REF_scf = -76.0260868661
    REF_scsomp2 = -76.2280452486
    assert psi4.compare_values(REF_nuc, this_nuc, 3, "Nuclear Repulsion Energy (a.u.)")
    # TEST
    assert psi4.compare_values(REF_scf, this_scf, 6, "SCF Energy (a.u.)")
    # TEST
    assert psi4.compare_values(REF_scsomp2, this_energy, 6, "SCS-OMP2 Total Energy (a.u.)")
    # TEST

    utils.compare_iterations(result, 5, check_iter)


#! SCS-OMP3 cc-pVDZ geometry optimization for the H2O molecule.
def test_scsmp3_opt(check_iter):

    h2o = psi4.geometry(
        """
        0 1
        o
        h 1 0.958
        h 1 0.958 2 104.4776
    """
    )

    psi4.core.clean_options()
    psi4_options = {"basis": "cc-pvdz", "max_energy_g_convergence": 7}
    psi4.set_options(psi4_options)

    result = optking.optimize_psi4("scs-omp3")
    print(json.dumps(result, indent=2))

    this_nuc = result["trajectory"][-1]["properties"]["nuclear_repulsion_energy"]  # TEST
    this_scf = result["trajectory"][-1]["extras"]["qcvars"]["SCF TOTAL ENERGY"]  # TEST
    this_energy = result["energies"][-1]  # TEST
    REF_nuc = 9.1193753755  # TEST
    REF_scf = -76.0261614278  # TEST
    REF_scsomp3 = -76.2296260036  # TEST
    assert psi4.compare_values(REF_nuc, this_nuc, 3, "Nuclear Repulsion Energy (a.u.)")
    # TEST
    assert psi4.compare_values(REF_scf, this_scf, 6, "SCF Energy (a.u.)")
    # TEST
    assert psi4.compare_values(REF_scsomp3, this_energy, 6, "SCS-OMP3 Total Energy (a.u.)")
    # TEST

    utils.compare_iterations(result, 5, check_iter)


#! SOS-OMP2 cc-pVDZ geometry optimization for the H2O molecule.
def test_sosmp2_opt(check_iter):

    h2o = psi4.geometry(
        """
        0 1
        o
        h 1 0.958
        h 1 0.958 2 104.4776
    """
    )

    psi4.core.clean_options()
    psi4_options = {"basis": "cc-pvdz", "max_energy_g_convergence": 7}
    psi4.set_options(psi4_options)

    result = optking.optimize_psi4("sos-omp2")
    print(json.dumps(result, indent=2))

    this_nuc = result["trajectory"][-1]["properties"]["nuclear_repulsion_energy"]  # TEST
    this_scf = result["trajectory"][-1]["extras"]["qcvars"]["SCF TOTAL ENERGY"]  # TEST
    this_energy = result["energies"][-1]  # TEST
    REF_nuc = 9.1236764248  # TEST
    REF_scf = -76.0262152850  # TEST
    REF_sosomp2 = -76.2106507336  # TEST
    assert psi4.compare_values(REF_nuc, this_nuc, 3, "Nuclear Repulsion Energy (a.u.)")
    # TEST
    assert psi4.compare_values(REF_scf, this_scf, 6, "SCF Energy (a.u.)")
    # TEST
    assert psi4.compare_values(REF_sosomp2, this_energy, 6, "SOS-OMP2 Total Energy (a.u.)")
    # TEST

    utils.compare_iterations(result, 4, check_iter)


#! SOS-OMP3 cc-pVDZ geometry optimization for the H2O molecule.
def test_sosmp3_opt(check_iter):

    h2o = psi4.geometry(
        """
        0 1
        o
        h 1 0.958
        h 1 0.958 2 104.4776
    """
    )

    psi4.core.clean_options()
    psi4_options = {"basis": "cc-pvdz", "max_energy_g_convergence": 7}
    psi4.set_options(psi4_options)

    result = optking.optimize_psi4("sos-omp3")
    print(json.dumps(result, indent=2))

    this_nuc = result["trajectory"][-1]["properties"]["nuclear_repulsion_energy"]  # TEST
    this_scf = result["trajectory"][-1]["extras"]["qcvars"]["SCF TOTAL ENERGY"]  # TEST
    this_energy = result["energies"][-1]  # TEST
    REF_nuc = 9.1134855397  # TEST
    REF_scf = -76.0261191302  # TEST
    REF_sosomp3 = -76.2277207554  # TEST
    assert psi4.compare_values(REF_nuc, this_nuc, 3, "Nuclear Repulsion Energy (a.u.)")
    # TEST
    assert psi4.compare_values(REF_scf, this_scf, 6, "SCF Energy (a.u.)")
    # TEST
    assert psi4.compare_values(REF_sosomp3, this_energy, 6, "SOS-OMP3 Total Energy (a.u.)")
    # TEST

    utils.compare_iterations(result, 5, check_iter)

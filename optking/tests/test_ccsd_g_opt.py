import psi4
import optking
import pytest
from .utils import utils

#! RHF-CCSD 6-31G** all-electron opt of H2O, default convergence
def test_ccsd_h2o(check_iter):

    h2o = psi4.geometry(
        """
        O
        H 1 0.97
        H 1 0.97 2 103.0
    """
    )

    psi4.core.clean_options()
    psi4_options = {"basis": "6-31G**", "scf_type": "pk"}
    psi4.set_options(psi4_options)

    result = optking.optimize_psi4("ccsd")
    print(result["trajectory"][-1].keys())

    this_scf = result["trajectory"][-1]["properties"]["scf_total_energy"]  # TEST
    this_ccsd = result["trajectory"][-1]["properties"]["ccsd_correlation_energy"]  # TEST
    this_total = result["trajectory"][-1]["properties"]["return_energy"]  # TEST
    REF_scf = -76.0229406477  # TEST
    REF_ccsd = -0.2082378354  # TEST
    REF_total = -76.2311784830  # TEST
    assert psi4.compare_values(REF_scf, this_scf, 4, "SCF energy")  # TEST
    assert psi4.compare_values(REF_ccsd, this_ccsd, 4, "CCSD contribution")  # TEST
    assert psi4.compare_values(REF_total, this_total, 4, "Total energy")  # TEST

    utils.compare_iterations(result, 3, check_iter)


#! ROHF-CCSD/cc-pVDZ $^{3}B@@1$ CH2 geometry opt, analytic gradients, tight
def test_ccsd_ch2(check_iter):

    ch2 = psi4.geometry(
        """
        0 3
        C
        H 1 1.1
        H 1 1.1 2 109.0
    """
    )

    psi4.core.clean_options()
    psi4_options = {
        "reference": "rohf",
        "basis": "cc-pvdz",
        "max_disp_g_convergence": 1e-6,
        "max_force_g_convergence": 1.0e-6,
        "max_energy_g_convergence": 7,
        "e_convergence": 10,
        "r_convergence": 10,
        "scf_type": "pk",
    }
    psi4.set_options(psi4_options)

    result = optking.optimize_psi4("CCSD")

    this_scf = result["trajectory"][-1]["properties"]["scf_total_energy"]  # TEST
    this_ccsd = result["trajectory"][-1]["properties"]["ccsd_correlation_energy"]  # TEST
    this_total = result["trajectory"][-1]["properties"]["return_energy"]  # TEST
    REF_scf = -38.9213947335  # TEST
    REF_cor = -0.1204840983  # TEST
    REF_tot = -39.0418788319  # TEST
    assert psi4.compare_values(REF_scf, this_scf, 6, "ROHF energy")  # TEST
    assert psi4.compare_values(REF_cor, this_ccsd, 6, "ROHF CCSD contribution")  # TEST
    assert psi4.compare_values(REF_tot, this_total, 6, "ROHF CCSD Total energy")  # TEST

    utils.compare_iterations(result, 9, check_iter)


#! UHF-CCSD/cc-pVDZ $^{3}B@@1$ CH2 geometry opt via analytic gradients, tight
def test_uccsd_ch2(check_iter):

    ch2 = psi4.geometry(
        """
        0 3
        C
        H 1 1.1
        H 1 1.1 2 109.0
    """
    )

    psi4.core.clean_options()
    psi4_options = {
        "reference": "uhf",
        "basis": "cc-pvdz",
        "max_disp_g_convergence": 1e-6,
        "max_force_g_convergence": 1.0e-6,
        "max_energy_g_convergence": 7,
        "e_convergence": 10,
        "r_convergence": 10,
        "scf_type": "pk",
    }
    psi4.set_options(psi4_options)

    result = optking.optimize_psi4("CCSD")

    this_scf = result["trajectory"][-1]["properties"]["scf_total_energy"]  # TEST
    this_ccsd = result["trajectory"][-1]["properties"]["ccsd_correlation_energy"]  # TEST
    this_total = result["trajectory"][-1]["properties"]["return_energy"]  # TEST
    REF_scf = -38.9265869596  # TEST
    REF_ccsd = -0.1153361899  # TEST
    REF_total = -39.0419231495  # TEST
    assert psi4.compare_values(REF_scf, this_scf, 6, "UHF energy")  # TEST
    assert psi4.compare_values(REF_ccsd, this_ccsd, 6, "UHF CCSD contribution")  # TEST
    assert psi4.compare_values(REF_total, this_total, 6, "UCCSD Total energy")  # TEST

    utils.compare_iterations(result, 9, check_iter)


#! UHF-CCSD(T)/cc-pVDZ $^{3}B@@1$ CH2 geometry optimization via analytic gradients
@pytest.mark.long
def test_uccsdpt_ch2(check_iter):

    ch2 = psi4.geometry(
        """
        0 3
        C
        H 1 1.1
        H 1 1.1 2 109.0
    """
    )

    psi4.core.clean_options()
    psi4_options = {
        "reference": "uhf",
        "basis": "cc-pvdz",
        "max_disp_g_convergence": 1e-6,
        "max_force_g_convergence": 1.0e-6,
        "max_energy_g_convergence": 7,
        "e_convergence": 10,
        "r_convergence": 10,
        "scf_type": "pk",
    }
    psi4.set_options(psi4_options)

    result = optking.optimize_psi4("CCSD(T)")

    this_ccsd_t = result["trajectory"][-1]["properties"]["ccsd_prt_pr_correlation_energy"]  # TEST
    this_total = result["trajectory"][-1]["properties"]["ccsd_prt_pr_total_energy"]  # TEST
    REF_scf = -38.9265520844  # TEST. Value is not currently included in trajectory output
    REF_ccsd_t = -0.1171601876  # TEST
    REF_total = -39.0437122710  # TEST
    assert psi4.compare_values(REF_ccsd_t, this_ccsd_t, 6, "CCSD(T) contribution")  # TEST
    assert psi4.compare_values(REF_total, this_total, 6, "Total CCSD(T) energy")  # TEST

    utils.compare_iterations(result, 9, check_iter)


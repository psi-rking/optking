import psi4
import optking

from .utils import utils

_schver = 2 if utils.psi4_runs_v2_qcschema(psi4.__version__) else 1

def test_charged_anion(check_iter):
    no2 = psi4.geometry(
        """
      -1 1
      N  -0.0001995858  -0.8633595176   0.0000000000
      O  -0.7026831726   0.4173795866   0.0000000000
      O   0.7028810651   0.4175165407   0.0000000000
    """
    )

    psi4.core.clean_options()
    psi4_options = {
        "basis": "cc-pvdz",
    }
    psi4.set_options(psi4_options)

    json_output = optking.optimize_psi4("hf")

    if _schver == 1:
        E = json_output["energies"][-1]
    elif _schver == 2:
        E = json_output["trajectory_properties"][-1]["return_energy"]
    refenergy = -203.894394347422
    assert psi4.compare_values(refenergy, E, 6, "RHF singlet NO2- energy")
    utils.compare_iterations(json_output, 3, check_iter)


def test_neutral_triplet(check_iter):
    o2 = psi4.geometry(
        """
      0 3
      O
      O 1 1.2
    """
    )

    psi4.core.clean_options()
    psi4_options = {
        "basis": "cc-pvdz",
        "reference": "uhf",
    }
    psi4.set_options(psi4_options)

    result = optking.optimize_psi4("hf")

    if _schver == 1:
        E = result["energies"][-1]
    elif _schver == 2:
        E = result["trajectory_properties"][-1]["return_energy"]
    REF_uhf = -149.6318688
    assert psi4.compare_values(REF_uhf, E, 6, "UHF triplet O2 Energy")  # TEST
    utils.compare_iterations(result, 3, check_iter)

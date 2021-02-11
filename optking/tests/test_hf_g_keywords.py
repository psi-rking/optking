import psi4
import optking

# HF SCF CC-PVDZ geometry optimization of HOOH with Z-matrix input
def test_B_dB_matrices():

    hooh = psi4.geometry(
        """
      H
      O 1 0.9
      O 2 1.4 1 100.0
      H 3 0.9 2 100.0 1 114.0
    """
    )

    psi4.core.clean_options()
    psi4_options = {
        "basis": "cc-pvdz",
        "g_convergence": "gau_tight",
        "scf_type": "pk",
        "TEST_B": True,
        "TEST_DERIVATIVE_B": True,
        "G_CONVERGENCE": "gau_tight",
    }
    psi4.set_options(psi4_options)

    json_output = optking.optimize_psi4("hf")  # Uses default program (psi4)

    E = json_output["energies"][-1]  # TEST
    nucenergy = json_output["trajectory"][-1]["properties"]["nuclear_repulsion_energy"]  # TEST
    refnucenergy = 38.06177  # TEST
    refenergy = -150.786766850  # TEST
    assert "test_b" in json_output["keywords"]  # TEST
    assert "test_derivative_b" in json_output["keywords"]  # TEST
    assert "g_convergence" in json_output["keywords"]  # TEST
    assert psi4.compare_values(refnucenergy, nucenergy, 3, "Nuclear repulsion energy")  # TEST
    assert psi4.compare_values(refenergy, E, 8, "Reference energy")  # TEST


def test_maxiter():

    h2o = psi4.geometry(
        """
     O
     H 1 1.0
     H 1 1.0 2 104.5
    """
    )

    psi4.core.clean_options()
    psi4options = {
        # Throw a bunch of options at psi4
        "diis": 0,
        "basis": "STO-3G",
        "e_convergence": 1e-10,
        "d_convergence": 1e-10,
        "scf_type": "PK",
        "geom_maxiter": 2,
    }
    psi4.set_options(psi4options)

    json_output = optking.optimize_psi4("hf")

    assert "geom_maxiter" in json_output["keywords"]  # TEST
    assert "Maximum number of steps exceeded" in json_output["error"]["error_message"]  # TEST
    assert "OptError" in json_output["error"]["error_type"]  # TEST


# Test the energy of geometry output, when maxiter is reached.
def test_maxiter_geom():

    h2o = psi4.geometry(
        """
        O
        H 1 1.0
        H 1 1.0 2 104.5
    """
    )

    psi4.core.clean_options()
    psi4options = {"basis": "cc-pvdz", "e_convergence": 10, "d_convergence": 10, "scf_type": "pk", "geom_maxiter": 2}
    psi4.set_options(psi4options)

    result = optking.optimize_psi4("hf")

    nextStepSchema = result["final_molecule"]  # TEST
    nextStepMolecule = psi4.core.Molecule.from_schema(nextStepSchema)  # TEST
    psi4.core.set_active_molecule(nextStepMolecule)  # TEST
    psi4.set_options(psi4options)
    nextStepEnergy = psi4.driver.energy("scf/cc-pvdz")  # TEST
    REF_energy = -76.0270381300  # TEST
    assert psi4.compare_values(REF_energy, nextStepEnergy, 5, "Energy of next-step molecule")
    # TEST

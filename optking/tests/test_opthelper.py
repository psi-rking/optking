#! optimization of water using OptHelper and explicit loop.
#  1. Showing how to use external gradient.
#  2. optking still has a module level parameters and history,
#       that could be eliminated, so not yet multi-object safe.
#  3. Have not yet restarted from json or disk, but should be close to working.
import optking
import pytest


def test_step_by_step():
    import psi4
    h2o = psi4.geometry(
        """
         O
         H 1 1.0
         H 1 1.0 2 104.5
    """
    )

    psi4.core.clean_options()
    psi4_options = {
        "diis": False,
        "basis": "sto-3g",
        "e_convergence": 10,
        "d_convergence": 10,
        "scf_type": "pk",
    }
    psi4.set_options(psi4_options)

    opt = optking.CustomHelper(h2o)

    for step in range(30):

        grad, wfn = psi4.gradient("hf", return_wfn=True)
        opt.gX = grad.np.reshape(-1)
        opt.E = wfn.energy()

        opt.compute()
        opt.take_step()
        conv = opt.test_convergence()
        if conv is True:
            print("Optimization SUCCESS:")
            break

        geom = psi4.core.Matrix.from_array(opt.molsys.geom)
        h2o.set_geometry(geom)
        h2o.update_geometry()

    else:
        print("Optimization FAILURE:\n")

    # print(opt.history.summary_string())
    json_output = opt.close()

    E = json_output["energies"][-1]  # TEST

    nucenergy = json_output["trajectory"][-1]["properties"]["nuclear_repulsion_energy"]
    refnucenergy = 8.9064983474  # TEST
    refenergy = -74.9659011923  # TEST
    assert psi4.compare_values(refnucenergy, nucenergy, 3, "Nuclear repulsion energy")
    assert psi4.compare_values(refenergy, E, 6, "Reference energy")


def test_lj_external_gradient():
    import qcelemental as qcel
    import numpy as np

    h2o = qcel.models.Molecule.from_data(
        """
        O  0.00000000  0.00000000 -0.12947689
        H  0.00000000 -1.49418674  1.02744610
        H  0.00000000  1.49418674  1.02744610
        """
    )

    optking_options = {"g_convergence": "gau_verytight", "intrafrag_hess": "SIMPLE"}

    opt = optking.CustomHelper(h2o, optking_options)

    for step in range(30):
        # Compute one's own energy and gradient
        E, gX = optking.lj_functions.calc_energy_and_gradient(opt.geom, 2.5, 0.01, True)
        # Insert these values into the 'user' computer.
        opt.E = E
        opt.gX = gX

        opt.compute()
        opt.take_step()
        conv = opt.test_convergence()
        if conv is True:
            print("Optimization SUCCESS:")
            break
    else:
        print("Optimization FAILURE:\n")

    # print(opt.history.summary_string())
    json_output = opt.close()

    assert conv is True
    E = json_output["energies"][-1]  # TEST
    RefEnergy = -0.03  # - epsilon * 3, where -epsilon is depth of each Vij well
    assert np.isclose(RefEnergy, E, rtol=1e-05, atol=1e-6)


# Demonstrate a complete export/import of OptHelper object
def test_stepwise_export():
    import psi4
    import pprint

    h2o = psi4.geometry(
        """
         O
         H 1 1.0
         H 1 1.0 2 104.5
    """
    )

    psi4.core.clean_options()
    psi4_options = {
        "diis": False,
        "basis": "sto-3g",
        "e_convergence": 10,
        "d_convergence": 10,
        "scf_type": "pk",
    }
    psi4.set_options(psi4_options)

    opt = optking.CustomHelper(h2o)
    optSaved = opt.to_dict()

    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(optSaved)

    print(opt.molsys.show_geom())

    for _ in range(10):
        opt = optking.CustomHelper.from_dict(optSaved)

        grad, wfn = psi4.gradient("hf", return_wfn=True)
        opt.gX = grad.np.reshape(-1)
        opt.E = wfn.energy()
        opt.compute()
        opt.take_step()
        conv = opt.test_convergence()
        if conv is True:
            print("Optimization SUCCESS:")
            break
        else:
            optSaved = opt.to_dict()

        geom = psi4.core.Matrix.from_array(opt.molsys.geom)
        h2o.set_geometry(geom)
        h2o.update_geometry()

    else:
        print("Optimization FAILURE:\n")

    # print(opt.history.summary_string())
    json_output = opt.close()

    E = json_output["energies"][-1]  # TEST

    nucenergy = json_output["trajectory"][-1]["properties"]["nuclear_repulsion_energy"]
    refnucenergy = 8.9064983474  # TEST
    refenergy = -74.9659011923  # TEST
    assert psi4.compare_values(refnucenergy, nucenergy, 3, "Nuclear repulsion energy")
    assert psi4.compare_values(refenergy, E, 6, "Reference energy")

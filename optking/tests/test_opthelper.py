#! optimization of water using OptHelper and explicit loop.
#  1. Showing how to use external gradient.
#  2. optking still has a module level parameters and history,
#       that could be eliminated, so not yet multi-object safe.
#  3. Have not yet restarted from json or disk, but should be close to working.
import psi4
import optking


def test_step_by_step():
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

    opt = optking.OptHelper("hf", comp_type="psi4")
    opt.build_coordinates()

    for step in range(30):
        opt.energy_gradient_hessian()
        opt.step()
        conv = opt.test_convergence()
        if conv is True:
            print("Optimization SUCCESS:")
            break
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
    h2o = psi4.geometry(
        """
         O
         H 1 1.0
         H 1 1.0 2 104.5
    """
    )

    psi4.core.clean_options()
    psi4_options = {
        "basis": "sto-3g",
        "g_convergence": "gau_verytight",
    }
    psi4.set_options(psi4_options)

    opt = optking.OptHelper("hf", comp_type="user")
    opt.build_coordinates()

    for step in range(30):
        # Compute one's own energy and gradient
        E, gX = optking.lj_functions.calc_energy_and_gradient(opt.geom, 2.5, 0.01, True)
        # Insert these values into the 'user' computer.
        opt.E = E
        opt.gX = gX

        opt.energy_gradient_hessian()
        opt.step()
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
    assert psi4.compare_values(RefEnergy, E, 6, "L-J Energy upon optimization")


# Demonstrate a complete export/import of OptHelper object
def test_stepwise_export():
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

    opt = optking.OptHelper("hf", init_mode="setup")
    opt.build_coordinates()
    optSaved = opt.to_dict()

    import pprint

    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(optSaved)

    print(opt.molsys.show_geom())

    for _ in range(10):
        opt = optking.OptHelper.from_dict(optSaved)
        opt.energy_gradient_hessian()
        opt.step()
        conv = opt.test_convergence()
        optSaved = opt.to_dict()
        if conv is True:
            print("Optimization SUCCESS:")
            break
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

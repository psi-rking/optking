#! Test handling of atom.  No optimization, computes energy.
import json
import psi4
import optking

RefEnergy = -127.8037761406364581
RefSuccessRval = False

def test_atom():
    mol = psi4.geometry(
      """
        0 1
        Ne 0.0 0.0 0.0
      """
    )

    psi4.core.clean_options()
    psi4.set_options({ "basis": "3-21G" })

    json_output = optking.optimize_psi4("hf")

    E = json_output['energies'][-1]
    success = json_output['success']
    Etraj = json_output["trajectory"][-1]['extras']['qcvars']['CURRENT ENERGY']
    #error_msg = json_output['error']['error_message']
    #print(error_msg)

    assert psi4.compare_values(RefEnergy, E, 6, "Atom energy (energies)")  # TEST
    assert psi4.compare_values(RefSuccessRval, success, "Atom success Rval") # TEST
    assert psi4.compare_values(RefEnergy, E, 6, "Atom energy (trajectory)")  # TEST


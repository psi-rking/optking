#! Test handling of atom.  No optimization, computes energy.
import psi4
import optking

RefEnergy = -127.8037761406364581


def test_atom():
    mol = psi4.geometry(""" Ne 0.0 0.0 0.0 """)
    psi4.core.clean_options()
    psi4.set_options({"basis": "3-21G"})

    qc_output = optking.optimize_psi4("hf")

    E = qc_output["energies"][-1]
    success = qc_output["success"]
    Etraj = qc_output["trajectory"][-1]["extras"]["qcvars"]["CURRENT ENERGY"]

    assert not qc_output["success"]
    assert (
        qc_output["error"]["error_message"]
        == "There is only 1 atom. Nothing to optimize. Computing energy."
    )
    assert psi4.compare_values(E, RefEnergy, 6, "Atom energy (energies)")
    assert psi4.compare_values(Etraj, RefEnergy, 6, "Atom energy (trajectory)")

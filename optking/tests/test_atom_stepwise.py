#! Test handling of atom for non optimize_psi4 case. No optimization, computes energy.
import psi4
import optking
import numpy as np

from optking.optparams import OptParams
from optking.molsys import Molsys
from optking.optimize import OptimizationManager
from optking.optwrapper import make_computer
from optking.history import History
from optking.exceptions import OptError

RefEnergy = -127.8037761406364581

def test_atom_stepwise():
    neon = psi4.geometry(""" Ne """)

    psi4.core.clean_options()

    params = OptParams()
    history = History(params)
    opt_mol, qcschema_mol = Molsys.from_psi4(neon)

    opt_input = {
        #"keywords": {}, for optimizer, optional
        "initial_molecule": qcschema_mol,
        "input_specification": {
            "model": {"basis": "3-21G", "method": 'hf'},
            "driver": "gradient",
            "keywords": {},
        },
    }
    computer = make_computer(opt_input, "psi4")

    opt_manager = OptimizationManager(opt_mol, history, params, computer)

    try:
        opt_manager.start_step(np.array(0))
        assert False
    except OptError as error:
        qc_output = opt_manager.opt_error_handler(error)
        E = qc_output['energies'][-1]
        Etraj = qc_output["trajectory"][-1]['extras']['qcvars']['CURRENT ENERGY']

        assert not qc_output['success']
        assert qc_output["error"]["error_message"] == "There is only 1 atom. Nothing to optimize. Computing energy."
        assert psi4.compare_values(RefEnergy, E, 6, "Atom energy (energies)")
        assert psi4.compare_values(RefEnergy, E, 6, "Atom energy (trajectory)")


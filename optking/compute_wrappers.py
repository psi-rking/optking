import json
import logging
from copy import deepcopy

import numpy as np
from qcelemental.models import AtomicInput, AtomicResult, Molecule
from qcelemental.util.serialization import json_dumps

from .exceptions import OptError
from . import log_name

logger = logging.getLogger(f"{log_name}{__name__}")


class ComputeWrapper:
    """An implementation of MolSSI's qc schema

    Parameters
    ----------
    JSON_dict : dict
        should match qcschema_input format. version 1

    """

    def __init__(self, molecule, model, keywords, program):
        self.molecule = molecule
        # ensure molecule orientation does not differ from optking regardless of how mol was created
        self.molecule.update({'fix_com': True, 'fix_orientation': True})
        self.model = model
        self.keywords = keywords
        self.program = program
        self.trajectory = []
        self.energies = []

    @classmethod
    def init_full(cls, molecule, model, keywords, program, trajectory, energies):
        wrapper = cls(molecule, model, keywords, program)
        wrapper.trajectory = trajectory
        wrapper.energies = energies
        return wrapper

    def update_geometry(self, geom: np.ndarray):
        """Updates EngineWrapper for requesting calculation

        Parameters
        ----------
        geom : np.ndarray
            cartesian geometry 1D list

        Returns
        -------
        json_for_input : dict
        """

        self.molecule["geometry"] = [i for i in geom.flat]

    def generate_schema_input(self, driver):
        molecule = Molecule(**self.molecule)
        inp = AtomicInput(
            molecule=molecule, model=self.model, keywords=self.keywords, driver=driver
        )

        return inp

    def generate_schema_input_for_procedure(self, driver):
        molecule = Molecule(**self.molecule)
        mbspec = self.keywords
        mbspec["driver"] = driver

        return {"molecule": molecule, "specification": mbspec}

    def _compute(self, driver):
        """Abstract style method for child classes"""
        pass

    def compute(self, geom, driver, return_full=True, print_result=False):
        """Perform calculation of type driver

        Parameters
        ----------
        geom: np.ndarray
        driver: str
        return_full: boolean
        print_result: boolean

        Returns
        -------
        dict
        """

        self.update_geometry(geom)
        ret = self._compute(driver)
        # Decodes the Result Schema to remove numpy elements (Makes ret JSON serializable)
        ret = json.loads(json_dumps(ret))
        self.trajectory.append(ret)

        if print_result:
            logger.debug(json.dumps(ret, indent=2))

        if ret["success"]:
            self.energies.append(ret["properties"]["return_energy"])
        else:
            raise OptError(
                f"Error encountered for {driver} calc. {ret['error']['error_message']}",
                ret["error"]["error_type"],
            )

        if return_full:
            return ret
        else:
            return ret["return_result"]

    def energy(self, return_full=False):
        return self._compute("energy")

    def gradient(self, return_full=False):
        return self._compute("gradient")

    def hessian(self, return_full=False):
        return self._compute("hessian")


def make_computer_from_dict(computer_type, d):
    mol = d.get("molecule")
    mod = d.get("model")
    key = d.get("keywords")
    prog = d.get("program")
    traj = d.get("trajectory")
    ener = d.get("energies")

    if computer_type == "psi4":
        return Psi4Computer.init_full(mol, mod, key, prog, traj, ener)
    elif computer_type == "qc":
        return QCEngineComputer.init_full(mol, mod, key, prog, traj, ener)
    elif computer_type == "user":
        return UserComputer.init_full(mol, mod, key, prog, traj, ener)
    else:
        raise OptError("computer_type is unknown")


class Psi4Computer(ComputeWrapper):
    def _compute(self, driver):
        import psi4

        inp = self.generate_schema_input(driver)

        if "1.3" in psi4.__version__:
            ret = psi4.json_wrapper.run_json_qcschema(inp.dict(), clean=True)
        else:
            ret = psi4.schema_wrapper.run_json_qcschema(
                inp.dict(), clean=True, json_serialization=True
            )
        ret = AtomicResult(**ret)
        return ret


class QCEngineComputer(ComputeWrapper):
    def _compute(self, driver):
        import qcengine

        task_config = {}
        if self.program == "psi4":
            import psi4

            task_config["memory"] = psi4.core.get_memory() / 1000000000
            task_config["ncores"] = psi4.core.get_num_threads()

        if self.model == "(proc_spec_in_options)":
            logger.debug("QCEngineComputer.path: ManyBody")
            inp = self.generate_schema_input_for_procedure(driver)
            ret = qcengine.compute_procedure(inp, "qcmanybody", True, task_config=task_config)

        else:
            logger.debug("QCEngineComputer.path: Atomic")
            inp = self.generate_schema_input(driver)
            ret = qcengine.compute(inp, self.program, True, task_config=task_config)

        return ret


# Class to produce a compliant output with user provided energy/gradient/hessian
class UserComputer(ComputeWrapper):
    def __init__(self, molecule, model, keywords, program):
        super().__init__(molecule, model, keywords, program)
        self.external_energy = None
        self.external_gradient = None
        self.external_hessian = None

    output_skeleton = {
        "id": None,
        "schema_name": "qcschema_output",
        "schema_version": 1,
        "model": {"method": "unknown", "basis": "unknown"},
        "provenance": {"creator": "User", "version": "0.1"},
        "properties": {},
        "extras": {"qcvars": {}},
        "stdout": "User provided energy, gradient, or hessian is returned",
        "stderr": None,
        "success": True,
        "error": None,
    }

    def _compute(self, driver):
        E = self.external_energy
        gX = self.external_gradient
        HX = self.external_hessian

        if driver == "hessian":
            if HX is None or gX is None or E is None:
                raise OptError("Must provide hessian, gradient, and energy.")
        elif driver == "gradient":
            if gX is None or E is None:
                raise OptError("Must provide gradient and energy.")
        elif driver == "energy":
            if E is None:
                raise OptError("Must provide energy.")

        result = deepcopy(UserComputer.output_skeleton)
        result["driver"] = driver
        mol = Molecule(**self.molecule)
        result["molecule"] = mol
        NRE = mol.nuclear_repulsion_energy()
        result["properties"]["nuclear_repulsion_energy"] = NRE
        result["extras"]["qcvars"]["NUCLEAR REPULSION ENERGY"] = NRE

        result["properties"]["return_energy"] = E
        result["extras"]["qcvars"]["CURRENT ENERGY"] = E

        if driver in ["gradient", "hessian"]:
            result["extras"]["qcvars"]["CURRENT GRADIENT"] = gX

        if driver == "hessian":
            result["extras"]["qcvars"]["CURRENT HESSIAN"] = HX

        if driver == "energy":
            result["return_result"] = E
        elif driver == "gradient":
            result["return_result"] = gX
        elif driver == "hessian":
            result["return_result"] = HX

        return AtomicResult(**result)

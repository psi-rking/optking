import copy
import math
import numpy as np

from qcelemental.models import ResultInput, Result

from . import history


class ComputeWrapper:
    """ An implementation of MolSSI's qc schema

    Parameters
    ----------
    JSON_dict : dict
        should match qcschema_input format. version 1

    """

    def __init__(self, molecule, model, keywords):

        self.molecule = molecule
        self.model = model
        self.keywords = keywords
        self.trajectory = []
        self.energies = []

    def update_geometry(self, geom):
        """Updates EngineWrapper for requesting calculation

        Parameters
        ----------
        geom : list of float
            cartesian geometry 1D list
        driver : str, optional
            deafult is gradient. Other options: hessian or energy

        Returns
        -------
        json_for_input : dict
        """

        return json_for_input

    def generate_schema_input(self, driver):

        inp = ResultInput(molecule=self.molecule, model=self.model, keywords=self.keywords, driver=driver)

        return inp

    def compute(self, driver, return_full):

        ret = self._compute(driver)

        self.trajectory.append(ret)
        self.energies.append(ret.properties.return_energy)
        if return_full:
            return ret
        else:
            return ret.return_result

    def energy(self, return_full=False):
        return self._compute("energy", return_full)

    def gradient(self, return_full=False):
        return self._compute("gradient", return_full)

    def hessian(self, return_full=False):
        return self._compute("hessian", return_full)


class Psi4Computer(ComputeWrapper):

    def _compute(self, driver):

        import psi4

        inp = self.generate_schema_input(driver)
        ret = psi4.json_wrapper.run_json(inp.dict())
        ret = Result(**ret)
        return ret

class QCEngineComputer(ComputeWrapper):

    def _compute(self, driver):

        import qcengine

        inp = self.generate_schema_input(driver)
        ret = qcengine.compute(inp, self.program)
        return ret

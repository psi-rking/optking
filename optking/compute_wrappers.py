import copy
import math
import json
import numpy as np
import logging

from qcelemental.models import AtomicInput, Result, Molecule
from qcelemental.util.serialization import json_dumps

from . import history
from .exceptions import OptError

class ComputeWrapper:
    """ An implementation of MolSSI's qc schema

    Parameters
    ----------
    JSON_dict : dict
        should match qcschema_input format. version 1

    """

    def __init__(self, molecule, model, keywords, program):

        self.molecule = molecule
        self.model = model
        self.keywords = keywords
        self.program = program
        self.trajectory = []
        self.energies = []

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

        self.molecule['geometry'] = [i for i in geom.flat]

    def generate_schema_input(self, driver):

        molecule = Molecule(**self.molecule)
        inp = AtomicInput(molecule=molecule, model=self.model, keywords=self.keywords, driver=driver)

        return inp

    def _compute(self, driver):
        """ Abstract style method for child classes"""
        pass

    def compute(self, geom, driver, return_full=True, print_result=False):
        """ Perform calculation of type driver
            
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

        
        logger = logging.getLogger(__name__)
        
        self.update_geometry(geom)
        ret = self._compute(driver)
        # Decodes the Result Schema to remove numpy elements (Makes ret JSON serializable)
        ret = json.loads(json_dumps(ret))
        self.trajectory.append(ret)
        
        if print_result:
            logger.debug(json.dumps(ret, indent=2))
        
        if ret['success']: 
            self.energies.append(ret['properties']['return_energy'])
        else:
           raise OptError(f"Error encountered for {driver} calc. ret['error']['error_message']",
                          qc_result['error']['error_type'])

        if return_full:
            return ret
        else:
            return ret['return_result']

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

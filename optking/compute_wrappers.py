from copy import deepcopy
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
        self.external_energy = None
        self.external_gradient = None
        self.external_hessian = None

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
           raise OptError(f"Error encountered for {driver} calc. {ret['error']['error_message']}",
                          ret['error']['error_type'])

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

# Class to produce a compliant output with user provided energy/gradient/hessian
class UserComputer(ComputeWrapper):

    output_skeleton = {
        'id': None,
        'schema_name': 'qcschema_output',
        'schema_version': 1,
        'model': {'method': 'unknown', 'basis': 'unknown'},
        'provenance': {'creator': 'User', 'version': '0.1'},
        'properties': {},
        'extras': { 'qcvars':{} },
        'stdout': "User provided energy, gradient, or hessian is returned",
        'stderr': None, 'success': True, 'error': None
    }

    def _compute(self, driver):
        logger = logging.getLogger(__name__)
        logger.info('UserComputer only returning provided values')
        E = self.external_energy
        gX = self.external_gradient
        HX = self.external_hessian

        if driver == 'hessian':
            if (Hx is None) or (gX is None) or (E is None):
                raise OptError("Must provide hessian, gradient, and energy.")
        elif driver == 'gradient':
            if (gX is None) or (E is None):
                raise OptError("Must provide gradient and energy.")
        elif driver == 'energy':
            if E is None:
                raise OptError("Must provide energy.")

        result = deepcopy(UserComputer.output_skeleton)
        result['driver'] = driver
        mol = Molecule(**self.molecule)
        result['molecule'] = mol
        NRE = mol.nuclear_repulsion_energy()
        result['properties']['nuclear_repulsion_energy'] = NRE
        result['extras']['qcvars']['NUCLEAR REPULSION ENERGY'] = NRE

        result['properties']['return_energy'] = E
        result['extras']['qcvars']['CURRENT ENERGY'] = E

        if driver in ['gradient', 'hessian']:
            result['extras']['qcvars']['CURRENT GRADIENT'] = gX

        if driver == 'hessian':
            result['extras']['qcvars']['CURRENT HESSIAN'] = HX

        if driver == 'energy':
            result['return_result'] = E
        elif driver == 'gradient':
            result['return_result'] = gX
        elif driver == 'hessian':
            result['return_result'] = HX

        # maybe do this to protect against repeatedly going back for same?
        self.external_energy = None
        self.external_gradient = None
        self.external_hessian = None
        return Result(**result)


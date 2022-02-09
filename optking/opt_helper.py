"""
Helper to provide high-level interface for optking.  In particular to be able
to input your own gradients and run 1 step at a time.

Does NOT support the following features from the more complex function
optimize.optimize():

  IRC
  backward steps
  dynamic level parameter changing with automatic restart

OptHelper can be used with in the following modes:

  init_mode = 'run' : psi4 will do optimization
  init_mode = 'setup' : setup params, molsys, computer, history
  init_mode = 'restart' : do minimal initialization

A optHelper object from / in a psi4 input file

A step may be taken by setting the class attributes gX and E and then calling the class methods
energy_gradient_hessian() and step(). The class attribute HX may also be set at any time as
desired, if this is not set then optking will perform its normal update/guess procedure.
test_convergence may be used to determine compliance with optking's convergence criteria


Optking will create a OptimizationResult as output in this process. This will be written upon
calling close()

For an example please see tests/test_opthelper

"""

import logging
import json
from collections import ABC

import numpy as np
import qcelemental as qcel

from . import compute_wrappers, hessian, history, molsys, optwrapper
from .convcheck import conv_check
from .exceptions import OptError
from .optimize import get_pes_info, make_internal_coords, prepare_opt_output, get_hessian_protocol
from .stepAlgorithms import take_step
from .misc import import_psi4
from . import optparams as op


class Helper(ABC):
    def __init__(self, params={}):
        """Initialize options. Still need a molecule to create molsys and computer """

        optwrapper.initialize_options(params)
        self.params = op.params

        self.computer: compute_wrappers.ComputeWrapper
        self.step_num = 0
        self.irc_step_num = 0  # IRC not supported by OptHelper for now.
        # The following are not used before being computed:

        self._Hq = None
        self.HX = None
        self.gX = None
        self.E = None

        self.molsys: molsys.Molsys
        self.history = history.History()

    def build_coordinates(self):
        make_internal_coords(self.molsys, self.params)

    def show(self):
        logger = logging.getLogger(__name__)
        logger.info("Molsys:\n" + str(self.molsys))
        return

    def energy_gradient_hessian(self):
        """E and gX must be set by the user before calling this method. """

        self.compute()
        self.fq = self.molsys.gradient_to_internals(self.gX, -1.0)

        self.molsys.apply_external_forces(self.fq, self._Hq, self.step_num)
        self.molsys.project_redundancies_and_constraints(self.fq, self._Hq)

        self.history.append(self.molsys.geom, self.E, self.fq)
        self.history.nuclear_repulsion_energy = self.computer.trajectory[-1]["properties"][
                "nuclear_repulsion_energy"]
        self.history.current_step_report()

    def step(self):
        self.Dq = take_step(
            self.molsys,
            self.E,
            self.fq,
            self._Hq,
            self.params.step_type,
            self.computer,
            self.history,
        )
        self.new_geom = self.molsys.geom
        self.step_num += 1

    def test_convergence(self):
        converged = conv_check(self.step_num, self.molsys, self.Dq, self.fq, self.computer.energies)
        return converged

    def close(self):
        del self._Hq
        del self.history
        del self.params
        qcschema_output = prepare_opt_output(self.molsys, self.computer)
        self.molsys.clear()
        return qcschema_output

    @property
    def gX(self):
        return self._gX

    @gX.setter
    def gX(self, val):
        """ gX must be set in order to perform an optimization. Cartesian only"""

        if val is None:
            self._gX = val
        else:
            val = self.attempt_fromiter(val)

            if val.ndim == 1 and val.size == self.molsys.natom * 3:
                self._gX = val
            else:
                raise TypeError(f"Gradient must be a 1D iterable with length " f"{self.molsys.natom * 3}")

    @property
    def HX(self):
        return self._HX

    @HX.setter
    def HX(self, val):
        """HX may be None i.e. not provided."""

        if val is None:
            self._HX = val
        else:
            val = self.attempt_fromiter(val)

            if val.shape == (self.molsys.natom * 3, self.molsys.natom * 3):
                self._HX = val
            else:
                raise TypeError(f"Hessian must be a nxn iterable with n={self.molsys.natom * 3}")

    @property
    def E(self):
        if self._E is None:
            raise ValueError("No energy provided. OptHelper.E must be set")
        return self._E

    @E.setter
    def E(self, val):
        if isinstance(val, float) or val is None:
            self._E = val
        else:
            raise OptError("Energy must be of type float")

    @property
    def geom(self):
        return self.molsys.geom

    @staticmethod
    def attempt_fromiter(array):

        if not isinstance(array, np.ndarray):
            try:
                array = np.fromiter(array, dtype=float)
            except (IndexError, ValueError, TypeError) as error:
                raise ValueError("Could not convert input to numpy array") from error
        return array


class CustomHelper(Helper):
    """ Class allows for easy setup of optking. Accepts custom forces, energies,
    and hessians from user. User will need to write a loop to perform optimization.

    Notes
    -----
    Overrides. gX, Hessian, and Energy to allow for user input. """

    def __init__(self, mol_src, params={}):
        """
        Parameters
        ----------
        mol_src: [dict, qcel.models.Molecule, psi4.qcdb.Molecule]
            psi4 or qcelemental molecule to construct optking molecular system from
        """

        super().__init__(params)

        if isinstance(mol_src, qcel.models.Molecule):
            molecule_input = mol_src.dict()
            self.molsys = molsys.Molsys.from_schema(mol_src)
        elif isinstance(mol_src, dict):
            tmp = qcel.models.Molecule(**mol_src).dict()
            molecule_input = json.loads(qcel.util.serialization.json_dumps(tmp))
            self.molsys = molsys.Molsys.from_schema(molecule_input)
        else:
            import_psi4("Attempting to create molsys from psi4 molecule")
            if isinstance(mol_src, psi4.qcdb.Molecule):
                self.molsys = molsys.Molsys.from_psi4(mol_src)
            else:
                try:
                    self.molsys = molsys.Molsys.from_psi4(psi4.core.get_active_molecule())
                except Exception as error:
                    raise OptError("Failed to grab psi4 molecule as last resort") from error
        opt_input = {'initial_molecule': {}, 'input_specification': {'keywords': {}, 'model': {}}}
        self.computer = optwrapper.make_computer(opt_input)  # create dummy computer.

    def compute(self):
        """The call to computer in this class is essentially a lookup for the value provided by
        the User. """
        if not self.HX:
            if self.step_num == 0:
                self._Hq = hessian.guess(self.molsys)
            else:
                self._Hq = self.history.hessian_update(self._Hq, self.molsys)
            self.gX = self.computer.compute(self.geom, driver="gradient", return_full=False)
        else:
            result = self.computer.compute(self.gX, driver="hessian")
            self.HX = result["return_result"]
            self.gX = result["extras"]["qcvars"]["gradient"]
            self._Hq = self.molsys.hessian_to_internals(self.HX)
            self.HX = None  # set back to None

    def calculations_needed(self):
        """Assume gradient is always needed. Provide tuple with keys for required properties """
        hessian_protocol = get_hessian_protocol(self.step_number, self.irc_step_number)

        if hessian_protocol == 'compute':
            return ('energy', 'gradient', 'hessian')
        else:
            return ('energy', 'gradient')

    @E.setter
    def E(self, val):
        """Set energy in self and computer """
        super().E = val
        self.computer.external_energy = val

    @HX.setter
    def HX(self, val):
        """Set hessian in self and computer """
        super().HX = val
        self.computer.external_hessian = val

    @gX.setter
    def gX(self, val):
        """Set gradient in self and computer """
        super().gX = val
        self.computer.external_gradient = val


class EngineHelper(Helper):
    """Setup an optimization using qcarchive to setup a molecular system and get gradients, energies
    etc... """
    def __init__(self, optimization_input, params={}):

        if isinstance(optimization_input, qcel.models.OptimizationInput):
            self.opt_input = optimization_input.dict()
        elif isinstance(optimization_input, dict):
            tmp = qcel.models.OptimizationInput(**optimization_input).dict()
            self.opt_input = json.loads(qcel.util.serialization.json_dumps(tmp))
        # self.calc_name = self.opt_input['input_specification']['model']['method']

        super(self, params)
        self.molsys = molsys.Molsys.from_schema(self.opt_input['initial_molecule'])
        self.computer = optwrapper.make_computer(self.opt_input, 'qc')

    def compute(self):
        self._Hq, self.gX = get_pes_info(self._Hq,
                                         self.computer,
                                         self.molsys,
                                         self.step_num,
                                         self.irc_step_num)
        self.E = self.computer.energies[-1]

    def optimize(self):
        """ Creating an EngineHelper and calling optimize() is equivalent to simply calling
        optimize_qcengine() with an OptimizationInput. However, EngineHelper will maintain
        its state. """
        self.opt_input = optwrapper.optimize_qcengine(self.opt_input)
        # update molecular system
        # set E, g_x, and hessian to have their last values
        # set step_number
        # set self.history to match history.

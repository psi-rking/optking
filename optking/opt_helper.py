"""
Helper to provide high-level interface for optking.  In particular to be able
to input your own gradients and run 1 step at a time.

Does NOT support the following features from the more complex function
optimize.optimize():

  IRC
  backward steps
  dynamic level parameter changing with automatic restart

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
import numpy as np

from . import optparams
from . import optwrapper
from . import molsys
from . import history
from . import hessian
from . import compute_wrappers
from .optimize import make_internal_coords, prepare_opt_output, get_pes_info
from .intcosMisc import (apply_external_forces,
                         project_redundancies_and_constraints,
                         hessian_to_internals)
from .stepAlgorithms import take_step
from .convCheck import conv_check
from .exceptions import OptError


class OptHelper(object):
    def __init__(self, calc_name, program='psi4', dertype=None, xtra_opt_params=None,
                 comp_type='qc', init_mode='setup'):

        self.calc_name = calc_name
        self.program = program
        self.dertype = dertype  # TODO should be removed; hack for psi4 fd
        if not xtra_opt_params:
            self.xtra_opt_params = {}
        else:
            self.xtra_opt_params = xtra_opt_params
        self.comp_type = comp_type
        self.computer: compute_wrappers.ComputeWrapper

        if init_mode == 'restart':
            pass
        else:

            if init_mode == 'run':
                optwrapper.optimize_psi4(calc_name, program, dertype, comp_type, **xtra_opt_params)
            elif init_mode == 'setup':
                init_tuple = optwrapper.initialize_from_psi4(calc_name,
                                                             program,
                                                             computer_type=comp_type,
                                                             dertype=None,
                                                             **self.xtra_opt_params)
                self.params, self.molsys, self.computer, _ = init_tuple

                self.history = history.History()

            else:
                raise OptError('OptHelper does not know given init_mode')

            self.step_num = 0
            self.irc_step_num = 0  # IRC not supported by OptHelper for now.
            # The following are not used before being computed:

            self._Hq = None
            self.HX = None
            self.gX = None
            self.E = None

            self.fq = None
            self.new_geom = None
            self.Dq = None

    def to_dict(self):
        d = {'calc_name': self.calc_name,
             'program': self.program,
             'dertype': self.dertype,
             'xtra_opt_params': self.xtra_opt_params,
             'comp_type': self.comp_type,
             'step_num': self.step_num,
             'irc_step_num': self.irc_step_num,
             'params': self.params.__dict__,
             'molsys': self.molsys.to_dict(),
             'history': self.history.to_dict(),
             'computer': self.computer.__dict__,
             'hessian': self._Hq}

        return d

    @classmethod
    def from_dict(cls, d):
        calc_name = d.get('calc_name')
        program = d.get('program')
        dertype = d.get('dertyp')
        XtraOptParams = d.get('xtra_opt_params')
        comp_type = d.get('comp_type')

        helper = cls(calc_name, program, dertype, XtraOptParams, comp_type, init_mode='restart')

        helper.params = optparams.OptParams(d.get('params'))
        helper.molsys = molsys.Molsys.from_dict(d.get('molsys'))
        helper.history = history.History.from_dict(d.get('history'))
        helper.computer = compute_wrappers.make_computer_from_dict(comp_type, d.get('computer'))
        helper.step_num = d.get('step_num')
        helper.irc_step_num = d.get('irc_step_num')
        helper._Hq = d.get('hessian')
        return helper

    def build_coordinates(self):
        make_internal_coords(self.molsys, self.params)

    def show(self):
        logger = logging.getLogger(__name__)
        logger.info('Molsys:\n' + str(self.molsys))
        return 

    def energy_gradient_hessian(self):
        """E and gX must be set by the user before calling this method. """

        self.compute()
        self.fq = self.molsys.q_forces(self.gX)

        apply_external_forces(self.molsys, self.fq, self._Hq, self.step_num)
        project_redundancies_and_constraints(self.molsys, self.fq, self._Hq)

        self.history.append(self.molsys.geom, self.E, self.fq)
        self.history.nuclear_repulsion_energy = \
            (self.computer.trajectory[-1]['properties']['nuclear_repulsion_energy'])
        self.history.current_step_report()

    def step(self):
        self.Dq = take_step(self.molsys, self.E, self.fq, self._Hq, self.params.step_type,
                            self.computer, self.history)
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
                if isinstance(self.computer, compute_wrappers.UserComputer):
                    self.computer.external_gradient = val
            else:
                raise TypeError(f'Gradient must be a 1D iterable with length '
                                f'{self.molsys.natom * 3}')

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
                if isinstance(self.computer, compute_wrappers.UserComputer):
                    self.computer.external_hessian = val
            else:
                raise TypeError(f'Hessian must be a nxn iterable with n={self.molsys.natom * 3}')

    @property
    def E(self):
        if self._E is None:
            raise ValueError("No energy provided. OptHelper.E must be set")
        return self._E

    @E.setter
    def E(self, val):
        if isinstance(val, float) or val is None:
            self._E = val
            if isinstance(self.computer, compute_wrappers.UserComputer):
                self.computer.external_energy = val
        else:
            raise OptError("Energy must be of type float")

    @property
    def geom(self):
        return self.molsys.geom

    def compute(self):
        """ If self.computer does not use user input. Use standard method to get any information
        for a step will require execution of an AtomicInput.

        Otherwise, simulate a computation with a mock Computer to CREATE an AtomicResult for
        each step. Guesses or updates the hessian if the user has not provided one. User MUST set
        self.gX and self.E attributes.
        """

        true_computers = (compute_wrappers.Psi4Computer, compute_wrappers.QCEngineComputer)
        if isinstance(self.computer, true_computers):
            self._Hq, self.gX = get_pes_info(self._Hq, self.computer, self.molsys, self.step_num,
                                             self.irc_step_num)
            self.E = self.computer.energies[-1]
            return

        if not self.HX:
            if self.step_num == 0:
                self._Hq = hessian.guess(self.molsys)
            else:
                self._Hq = self.history.hessian_update(self._Hq, self.molsys)
            self.gX = self.computer.compute(self.geom, driver='gradient', return_full=False)
        else:
            result = self.computer.compute(self.gX, driver='hessian')
            self.HX = result['return_result']
            self.gX = result['extras']['qcvars']['gradient']
            self._Hq = hessian_to_internals(self.HX, self.molsys)
            self.HX = None  # set back to None

    @staticmethod
    def attempt_fromiter(array):

        if not isinstance(array, np.ndarray):
            try:
                array = np.fromiter(array, dtype=float)
            except (IndexError, ValueError, TypeError) as error:
                raise ValueError("Could not convert input to numpy array") from error

        return array

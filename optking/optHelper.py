import logging
import numpy as np

from . import optparams
from . import optwrapper
from . import history
from . import molsys
from . import history
from .optimize import get_pes_info, make_internal_coords, prepare_opt_output
from .intcosMisc import apply_fixed_forces, project_redundancies_and_constraints
from .stepAlgorithms import take_step
from .convCheck import conv_check
from .exceptions import AlgError, OptError
import qcelemental as qcel


# Helper to provide high-level interface for optking.  In particular to be able
# to input your own gradients.  Also, to be able to run 1 step at a time.
#
# Does NOT support the following features from the more complex function
# optimize.optimize():
#  IRC
#  backward steps
#  dynamic level parameter changing with automatic restart

class optHelper(object):
    def __init__(self, calc_name, program='psi4', dertype=None, XtraOptParams=None,
                 comp_type='qc'):
        runOpt = [False, None, None, None]

        optwrapper.optimize_psi4(calc_name, program, XtraOptParams,
            runOptimization=runOpt, computer_type=comp_type)

        self.params, self.molsys, self.computer = runOpt[1], runOpt[2], runOpt[3]
        self.__Hq = None
        self.__gX = None
        self.fq = None
        self.step_num = 0
        self.E = 0
        self.irc_step_num = 0  #IRC not supported by optHelper for now.
        self.history = history.History()
        return 

    def build_coordinates(self):
        make_internal_coords(self.molsys, self.params)

    def show(self):
        logger = logging.getLogger(__name__)
        logger.info('Molsys:\n' + str(self.molsys))
        return 

    def energy_gradient_hessian(self):
        self.__Hq, self.__gX = get_pes_info(self.__Hq,
             self.computer, self.molsys, self.step_num, self.irc_step_num, self.history)

        self.E = self.computer.energies[-1]
        self.fq = self.molsys.q_forces(self.__gX)
        apply_fixed_forces(self.molsys, self.fq, self.Hq, self.step_num)
        project_redundancies_and_constraints(self.molsys, self.fq, self.Hq)

        self.history.append(self.molsys.geom, self.E, self.fq)
        self.history.nuclear_repulsion_energy = \
            (self.computer.trajectory[-1]['properties']['nuclear_repulsion_energy'])
        self.history.current_step_report()

    def step(self):
        self.Dq = take_step(self.molsys, self.E, self.fq, self.__Hq,
                             self.params.step_type, self.computer, self.history)
        self.step_num += 1
        self.newGeom = self.molsys.geom
 
    def testConvergence(self):
        conv = False
        return conv_check(self.step_num, self.molsys,
            self.Dq, self.fq, self.computer.energies)

    def close(self):
        del self.__Hq
        del self.history
        del self.params
        qcschema_output = prepare_opt_output(self.molsys, self.computer)
        self.molsys.clear()
        return qcschema_output

    @property
    def gX(self):
        return self.__gX

    @gX.setter
    def gX(self, val):
        if type(val) == np.ndarray:
            self.__gX = val
        else:
            raise OptError('Gradient should have format {}'.format(np.ndarray))

    @property
    def Hq(self):
        return self.__Hq

    @Hq.setter
    def Hq(self, val):
        if type(val) == np.ndarray:
            self.__Hq = val
        else:
            raise OptError('Hessian should have format {}'.format(np.ndarray))

    @property
    def geom(self):
        return self.molsys.geom


import logging
from . import optparams
from . import optwrapper
from . import history
from . import molsys
from . import history
from .optimize import get_pes_info, make_internal_coords, prepare_opt_output
from .intcosMisc import apply_fixed_forces, project_redundancies_and_constraints
from .stepAlgorithms import take_step
from .convCheck import conv_check

# Helper to provide high-level interface for optking.  In particular to be able
# to input your own gradients.  Also, to be able to run 1 step at a time.
#
# Does NOT support the following features from the more complex function
# optimize.optimize():
#  IRC
#  backward steps
#  dynamic level parameter changing with automatic restart

class optHelper(object):
    def __init__(self, calc_name, program='psi4', dertype=None, XtraOptParams=None):
        runOpt = [False, None, None, None]
        optwrapper.optimize_psi4(calc_name, 'psi4', None, None, runOptimization=runOpt)
        self._params, self._molsys, self._computer = runOpt[1], runOpt[2], runOpt[3]
        self._Hq = None
        self._fq = None
        self._step_num = 0
        self._E = 0
        self._irc_step_num = 0  #IRC not supported by optHelper for now.
        self._history = history.History()
        return 

    def build_coordinates(self):
        make_internal_coords(self._molsys, self._params)

    def show(self):
        logger = logging.getLogger(__name__)
        logger.info('Molsys:\n' + str(self._molsys))
        return 

    def energy_gradient_hessian(self):
        self._Hq, self._gX = get_pes_info(self._Hq,
             self._computer, self._molsys, self._step_num, self._irc_step_num, self._history)
        self._E = self._computer.energies[-1]


        self._fq = self._molsys.q_forces(self._gX)
        apply_fixed_forces(self._molsys, self._fq, self._Hq, self._step_num)
        project_redundancies_and_constraints(self._molsys, self._fq, self._Hq)

        self._history.append(self._molsys.geom, self._E, self._fq)
        self._history.nuclear_repulsion_energy = \
            (self._computer.trajectory[-1]['properties']['nuclear_repulsion_energy'])
        self._history.current_step_report()
        return

    def step(self):
        self._Dq = take_step(self._molsys, self._E, self._fq, self._Hq,
                             self._params.step_type, self._computer, self._history)
        self._step_num += 1
        self._newGeom = self._molsys.geom
 
    def testConvergence(self):
        conv = False
        return conv_check(self._step_num, self._molsys,
            self._Dq, self._fq, self._computer.energies)

    def close(self):
        del self._Hq
        del self._history
        del self._params
        qcschema_output = prepare_opt_output(self._molsys, self._computer)
        self._molsys.clear()
        return qcschema_output

    @property
    def gX(self):
        return self._gX

    @property
    def Hq(self):
        return self._Hq

    @property
    def geom(self):
        return self._molsys.geom


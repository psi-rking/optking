import logging
import numpy as np
import qcelemental as qcel

from . import optparams
from . import optwrapper
from . import history
from . import molsys
from . import history
from . import compute_wrappers
from .optimize import get_pes_info, make_internal_coords, prepare_opt_output
from .intcosMisc import apply_fixed_forces, project_redundancies_and_constraints
from .stepAlgorithms import take_step
from .convCheck import conv_check
from .exceptions import AlgError, OptError


# Helper to provide high-level interface for optking.  In particular to be able
# to input your own gradients.  Also, to be able to run 1 step at a time.
#
# Does NOT support the following features from the more complex function
# optimize.optimize():
#  IRC
#  backward steps
#  dynamic level parameter changing with automatic restart
#
#  init_mode = 'run' : psi4 will do optimization
#  init_mode = 'setup' : setup params, molsys, computer, history
#  init_mode = 'restart' : do minimal initialization

class optHelper(object):
    def __init__(self, calc_name, program='psi4', dertype=None, XtraOptParams=None,
                 comp_type='qc', init_mode='setup'):

        self.calc_name = calc_name
        self.program = program
        self.dertype = dertype #should be removed; hack for psi4 fd
        self.XtraOptParams = XtraOptParams
        self.comp_type = comp_type

        if init_mode == 'run':
            runOpt = [True]
            optwrapper.optimize_psi4(calc_name, program, XtraOptParams,
                runOptimization=runOpt, computer_type=comp_type)

        elif init_mode == 'setup':
            runOpt = [False, None, None, None]
            optwrapper.optimize_psi4(calc_name, program, XtraOptParams,
                runOptimization=runOpt, computer_type=comp_type)

            self.params, self.molsys, self.computer = runOpt[1], runOpt[2], runOpt[3]
            self.history = history.History()

        elif init_mode == 'restart':
            pass
        else:
            raise OptError('optHelper does not know given init_mode')

        self.step_num = 0
        self.irc_step_num = 0  #IRC not supported by optHelper for now.
        self.__Hq = None
        # The following are not used before being computed:
        self.__gX = None
        self.fq = None
        self.E = 0
        return 

    def to_dict(self):
        d = {}
        d['calc_name'] = self.calc_name
        d['program'] = self.program
        d['dertype'] = self.dertype
        d['XtraOptParams'] = self.XtraOptParams
        d['comp_type'] = self.comp_type

        d['step_num'] = self.step_num
        d['irc_step_num'] = self.irc_step_num

        d['params']  = self.params.__dict__
        d['molsys']  = self.molsys.to_dict()
        d['history'] = self.history.to_dict()
        d['computer'] = self.computer.__dict__
        d['hessian'] = self.__Hq
        return d

    @classmethod
    def from_dict(cls, d):
        calc_name = d.get('calc_name')
        program = d.get('program')
        dertype = d.get('dertyp')
        XtraOptParams = d.get('XtraOptParams')
        comp_type = d.get('comp_type')


        helper = cls(calc_name, program, dertype, XtraOptParams, comp_type, init_mode='restart')

        helper.params = optparams.OptParams(d.get('params'))
        helper.molsys = molsys.Molsys.from_dict(d.get('molsys'))
        helper.history = history.History.from_dict(d.get('history'))
        helper.computer = compute_wrappers.make_computer_from_dict(comp_type, d.get('computer'))
        helper.step_num = d.get('step_num')
        helper.irc_step_num = d.get('irc_step_num')
        helper.__Hq = d.get('hessian')
        return helper

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
        self.newGeom = self.molsys.geom
 
    def testConvergence(self):
        converged = conv_check(self.step_num, self.molsys, self.Dq, self.fq,
                               self.computer.energies)
        self.step_num += 1
        return converged

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


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
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import qcelemental as qcel

from . import compute_wrappers, hessian, history, molsys, optwrapper
from .convcheck import conv_check
from .exceptions import OptError, AlgError
from .optimize import get_pes_info, make_internal_coords, prepare_opt_output, OptimizationManager
from .misc import import_psi4
from .printTools import print_geom_grad, welcome
from . import optparams as op
from . import log_name

logger = logging.getLogger(f"{log_name}{__name__}")


class Helper(ABC):
    def __init__(self, params={}, **kwargs):
        """Initialize options. Still need a molecule to create molsys and computer """

        optwrapper.initialize_options(params, silent=kwargs.get("silent", False))
        self.params = op.Params

        self.computer: compute_wrappers.ComputeWrapper
        self.step_num = 0
        self.irc_step_num = 0  # IRC not supported by OptHelper for now.
        # The following are not used before being computed:

        self._Hq: Union[np.ndarray, None] = None
        self.HX: Union[np.ndarray, None] = None
        self.gX: Union[np.ndarray, None] = None
        self.fq: Union[np.ndarray, None] = None
        self.dq: Union[np.ndarray, None] = None
        self.new_geom: Union[np.ndarray, None] = None
        self.E: Union[float, None] = None
        self.step_str = None

        self.opt_input: Union[dict, None] = None
        self.molsys: Union[molsys.Molsys, None] = None
        self.history = history.History(self.params)
        self.opt_manager: OptimizationManager

    def pre_step_str(self):
        string = ""
        string += welcome()
        string += str(self.molsys)
        return string

    def post_step_str(self):
        string = ""
        string += self.step_str if self.step_str is not None else ""
        energies = [step.E for step in self.history.steps]
        status = self.status(str_mode='both')
        if status == "CONVERGED" and len(energies) > 0:
            if self.params.opt_type != 'IRC':
                conv_table, criteria_table = conv_check(self.step_num, self.dq, self.fq, energies, str_mode="both")
                string += conv_table
                string += criteria_table
                string += self.history.summary_string()
            else:
                string += self.opt_manager.opt_method.irc_history.progress_report(return_str=True)
        elif "FAILED" not in status and len(energies) > 0:
            string += conv_check(self.step_num, self.dq, self.fq, energies, str_mode="table")

        string += "Next Geometry in Ang \n"
        string += self.molsys.show_geom()
        return string

    def to_dict(self):
        d = {
            "step_num": self.step_num,
            "irc_step_num": self.irc_step_num,
            "params": self.params.__dict__,
            "molsys": self.molsys.to_dict(),
            "history": self.history.to_dict(),
            "computer": self.computer.__dict__,
            "hessian": self._Hq,
            "opt_input": self.opt_input,
        }

        return d

    @classmethod
    def from_dict(cls, d):
        """Construct as far as possible the helper. Child class will need to update computer """
        # creates the initial configuration of the OptHelper. Some options might
        # have changed over the course of the optimization (eg trust radius)

        helper = cls(d.get("opt_input"), params={}, silent=True)

        helper.params = op.OptParams.from_internal_dict(d.get("params"))
        op.Params = helper.params
        # update with current information
        helper.molsys = molsys.Molsys.from_dict(d.get("molsys"))
        helper.history = history.History.from_dict(d.get("history"))
        helper.step_num = d.get("step_num")
        helper.irc_step_num = d.get("irc_step_num")
        helper._Hq = d.get("hessian")
        return helper

    def build_coordinates(self):
        make_internal_coords(self.molsys, self.params)
        self.show()

    def show(self):
        logger.info("Molsys:\n" + str(self.molsys))
        return

    @abstractmethod
    def _compute(self):
        """get energy gradient and hessian """

    def compute(self):
        """Get the energy, gradient, and hessian. Project redundancies and apply constraints / forces """

        if not self.molsys.intcos_present:
            # opt_manager.molsys is the same object as this molsys
            make_internal_coords(self.molsys)
            logger.debug("Molecular system after make_internal_coords:")
            logger.info(str(self.molsys))

        self._compute()
        logger.info("\n\t%s", print_geom_grad(self.geom, self.gX))
        self.fq = self.molsys.gradient_to_internals(self.gX, -1.0)

        self.molsys.apply_external_forces(self.fq, self._Hq, self.step_num)
        self.molsys.project_redundancies_and_constraints(self.fq, self._Hq)

    def take_step(self):
        """Must call compute before calling this method. Takes the next step. """
        self.opt_manager.error = None
        try:
            self.dq, self.step_str = self.opt_manager.take_step(self.fq, self._Hq, self.E, return_str=True)
        except AlgError as e:
            self.opt_manager.alg_error_handler(e)

        self.new_geom = self.molsys.geom
        self.step_num += 1
        self.opt_manager.step_number += 1

        self.show()
        return self.dq

    def test_convergence(self, str_mode=None):
        return self.opt_manager.converged(self.E, self.fq, self.dq, self.step_num, str_mode=str_mode)

    def close(self):
        del self._Hq
        del self.params
        return self.opt_manager.finish()

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
            ncart = self.molsys.natom * 3
            if val.ndim == 1 and val.size == ncart:
                self._gX = val
            else:
                if val.shape == (self.molsys.natom, 3):
                    self._gX = np.ravel(val)
                else:
                    raise TypeError(
                        f"Gradient must an iterable with shape (3, {self.molsys.natom}) or (1, {ncart})"
                    )

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
            dim = self.molsys.natom * 3

            if val.shape == (dim, dim):
                self._HX = val
            elif val.ndim == 1 and len(val) == dim **2:
                self._HX = val.reshape(dim, dim)
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

    def summarize_result(self):
        """ Return final energy and geometry """
        last_step = self.history.steps[-1]
        return last_step.E, last_step.geom

    def status(self, str_mode=None):
        if self.opt_manager.error == "OptError":
            return "FAILED"

        if self.opt_manager.error == "AlgError":
            self.HX = None
            self._Hq = None
            self.step_num = 0
            return "UNFINISHED-FAILED"

        if self.test_convergence(str_mode) is True:
            return "CONVERGED"

        return "UNFINISHED"


class CustomHelper(Helper):
    """ Class allows for easy setup of optking. Accepts custom forces, energies,
    and hessians from user. User will need to write a loop to perform optimization.

    Notes
    -----
    Overrides. gX, Hessian, and Energy to allow for user input. """

    def __init__(self, mol_src, params={}, **kwargs):
        """
        Parameters
        ----------
        mol_src: [dict, qcel.models.Molecule, psi4.qcdb.Molecule]
            psi4 or qcelemental molecule to construct optking molecular system from
        """

        opt_input = {
            "initial_molecule": {"symbols": [], "geometry": []},
            "input_specification": {"keywords": {}, "model": {}},
        }
        self.computer = optwrapper.make_computer(opt_input, "user")
        super().__init__(params, **kwargs)

        if isinstance(mol_src, qcel.models.Molecule):
            self.opt_input = mol_src.dict()
            self.molsys = molsys.Molsys.from_schema(self.opt_input)
        elif isinstance(mol_src, dict):
            tmp = qcel.models.Molecule(**mol_src).dict()
            self.opt_input = json.loads(qcel.util.serialization.json_dumps(tmp))
            self.molsys = molsys.Molsys.from_schema(self.opt_input)
        else:
            import_psi4("Attempting to create molsys from psi4 molecule")
            import psi4

            if isinstance(mol_src, psi4.qcdb.Molecule):
                self.molsys, self.opt_input = molsys.Molsys.from_psi4(mol_src)
            else:
                try:
                    self.molsys, self.opt_input = molsys.Molsys.from_psi4(psi4.core.get_active_molecule())
                except Exception as error:
                    raise OptError("Failed to grab psi4 molecule as last resort") from error

        self.computer.molecule = self.opt_input
        self.build_coordinates()
        self.opt_manager = OptimizationManager(self.molsys, self.history, self.params, self.computer)
        self.opt_manager.step_number = 1

    @classmethod
    def from_dict(cls, d):
        helper = super().from_dict(d)
        helper.computer = compute_wrappers.make_computer_from_dict("user", d.get("computer"))
        helper.opt_manager = OptimizationManager(helper.molsys, helper.history, helper.params, helper.computer)
        return helper

    def _compute(self):
        """The call to computer in this class is essentially a lookup for the value provided by
        the User. """

        if self.HX is None:
            if "hessian" in self.calculations_needed():
                if self.params.cart_hess_read:
                    self.HX = hessian.from_file(self.params.hessian_file)  # set ourselves if file\
                    _ = self.computer.compute(self.geom, driver="hessian")
                    self._Hq = self.molsys.hessian_to_internals(self.HX)
                    self.HX = None
                    self.params.cart_hess_read = False
                    self.params.hessian_file = None
                else:
                    raise RuntimeError("Optking requested a hessian but was not provided one. "
                                       "This could be a driver issue")
            elif self.step_num == 0:
                logger.debug("Guessing hessian")
                self._Hq = hessian.guess(self.molsys, guessType=self.params.intrafrag_hess)
                self.gX = self.computer.compute(self.geom, driver="gradient", return_full=False)
            else:
                logger.debug("Updating hessian")
                self._Hq = self.history.hessian_update(self._Hq, self.molsys)
                self.gX = self.computer.compute(self.geom, driver="gradient", return_full=False)
        else:
            result = self.computer.compute(self.geom, driver="hessian")
            self.HX = result["return_result"]
            self.gX = result["extras"]["qcvars"]["CURRENT GRADIENT"]
            self._Hq = self.molsys.hessian_to_internals(self.HX)
            self.HX = None  # set back to None

    def calculations_needed(self):
        """Assume gradient is always needed. Provide tuple with keys for required properties """
        hessian_protocol = self.opt_manager.get_hessian_protocol()

        if hessian_protocol == "compute":
            return "energy", "gradient", "hessian"
        else:
            return "energy", "gradient"

    @property
    def E(self):
        return super().E

    @property
    def HX(self):
        return super().HX

    @property
    def gX(self):
        return super().gX

    @E.setter
    def E(self, val):
        """Set energy in self and computer """
        # call parent classes setter. Weird python syntax. Class will always be CustomHelper
        # self.__class__ could be a child class type. (No child class currently)
        # super() and super(__class__, self.__class__) should be equivalent but the latter is required?
        super(__class__, self.__class__).E.__set__(self, val)
        self.computer.external_energy = val

    @HX.setter
    def HX(self, val):
        """Set hessian in self and computer """
        super(__class__, self.__class__).HX.__set__(self, val)
        self.computer.external_hessian = val

    @gX.setter
    def gX(self, val):
        """Set gradient in self and computer """
        super(__class__, self.__class__).gX.__set__(self, val)
        self.computer.external_gradient = val


class EngineHelper(Helper):
    """Perform an optimization using qcengine to compute properties. Use OptimizationInput to setup
    a molecular system"""

    def __init__(self, optimization_input, **kwargs):

        if isinstance(optimization_input, qcel.models.OptimizationInput):
            self.opt_input = optimization_input.dict()
        elif isinstance(optimization_input, dict):
            tmp = qcel.models.OptimizationInput(**optimization_input).dict()
            self.opt_input = json.loads(qcel.util.serialization.json_dumps(tmp))
        # self.calc_name = self.opt_input['input_specification']['model']['method']

        super().__init__(optimization_input["keywords"], **kwargs)
        self.molsys = molsys.Molsys.from_schema(self.opt_input["initial_molecule"])
        self.computer = optwrapper.make_computer(self.opt_input, "qc")
        self.computer_type = "qc"
        self.build_coordinates()
        self.opt_manager = OptimizationManager(self.molsys, self.history, self.params, self.computer)

    @classmethod
    def from_dict(cls, d):
        helper = super().from_dict(d)
        helper.computer = compute_wrappers.make_computer_from_dict("qc", d.get("computer"))
        return helper

    def _compute(self):

        protocol = self.opt_manager.get_hessian_protocol()
        requires = self.opt_manager.opt_method.requires()

        self._Hq, self.gX = get_pes_info(self._Hq, self.computer, self.molsys, self.history, protocol, requires)
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

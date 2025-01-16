"""Helpers to provide high-level interfaces for OptKing. The Helpers allow individual steps to be taken easily
from a variety of sources. EngineHelper runs calculations through
`QCEngine <https://molssi.github.io/QCEngine/>`__. Optimizations can also be run through the QCEngine
procedure for OptKing :ref:`example <qcengine_running>`. CustomHelper adds the abilility to directly input energies, gradients,
hessians, etc...
"""

import logging
import json
import pathlib
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from optking.IRCfollowing import IntrinsicReactionCoordinate
import qcelemental as qcel

from . import compute_wrappers, hessian, history, molsys, optwrapper
from .convcheck import conv_check
from .exceptions import OptError, AlgError
from .optimize import (
    get_pes_info,
    make_internal_coords,
    optimize,
    prepare_opt_output,
    OptimizationManager,
)
from .misc import import_psi4
from .printTools import print_geom_grad, welcome
from . import log_name
from . import op

logger = logging.getLogger(f"{log_name}{__name__}")


class Helper(ABC):
    """
    Base class for CustomHelper (accepts user provided gradients) and EngineHelper (uses MolSSI's QCEngine for gradients)

    A step may be taken by setting the class attributes ``gX`` and ``E``, then calling the
    ``compute()`` and ``take_step()`` methods. The class attribute ``HX`` may also be set at any time as
    desired, if this is not set then OptKing will perform its normal update/guess procedure.

    If ``full_hess_every`` has been set, the optimizer will require that hessians be provided every n steps.
    The properties required by the optimizer can be queried by calling ``get_requirements()``
    test_convergence may be used to determine compliance with optking's convergence criteria

    OptKing will create a ``OptimizationResult`` as output in this process. This will be written upon
    calling close()

    """

    def __init__(self, params={}, **kwargs):
        """Initialize options. Still need a molecule to create molsys and computer"""

        optwrapper.initialize_options(params, silent=kwargs.get("silent", False))
        self.params: op.OptParams = op.Params

        self.computer: compute_wrappers.ComputeWrapper
        self.step_num = 0
        # The following are not used before being computed:

        self._Hq: Union[np.ndarray, None] = None
        self.HX: Union[np.ndarray, None] = None
        self.gX: Union[np.ndarray, None] = None
        self.fq: Union[np.ndarray, None] = None
        self.dq: Union[np.ndarray, None] = None
        self.new_geom: Union[np.ndarray, None] = None
        self.E: Union[float, None] = None
        self.step_str = None

        self.opt_input: Union[dict, None]
        self.molsys: Union[molsys.Molsys, None] = None
        self.history = history.History(self.params)
        self.opt_manager: OptimizationManager

    def pre_step_str(self):
        """Returns a formatted string to summarize molecular system before taking a step or calling compute"""

        string = ""
        string += welcome()
        string += str(self.molsys)
        return string

    def post_step_str(self):
        """Returns a formatted string to summarize the step taken after calling ``take_step()``"""

        string = ""
        string += self.step_str if self.step_str is not None else ""
        energies = [step.E for step in self.history.steps]
        status = self.status(str_mode="both")
        step_type = "irc" if self.params.opt_type == "IRC" else "standard"
        conv_info = {
            "step_type": step_type,
            "energies": energies,
            "dq": self.dq,
            "fq": self.fq,
            "iternum": self.step_num,
        }

        if status == "CONVERGED" and len(energies) > 0:
            if self.params.opt_type != "IRC":
                conv_table, criteria_table = conv_check(
                    conv_info, self.params.__dict__, str_mode="both"
                )
                string += conv_table
                string += criteria_table
                string += self.history.summary_string()
            else:
                string += self.opt_manager.opt_method.irc_history.progress_report(return_str=True)
        elif "FAILED" not in status and len(energies) > 0:
            if self.params.opt_type == "IRC":
                irc_object: IntrinsicReactionCoordinate = self.opt_manager.opt_method
                conv_info["sub_step_num"] = irc_object.sub_step_number
                conv_info["iternum"] = irc_object.irc_step_number
                conv_info["fq"] = irc_object.irc_history._project_forces(
                    self.fq, self.molsys, self.params.linear_algebra_tol
                )

            string += conv_check(conv_info, self.params, str_mode="table")

        string += "Next Geometry in Ang \n"
        string += self.molsys.show_geom()
        return string

    def to_dict(self):
        d = {
            "step_num": self.step_num,
            "params": self.params.to_dict(by_alias=False),
            "molsys": self.molsys.to_dict(),
            "history": self.history.to_dict(),
            "computer": self.computer.__dict__,
            "hessian": self._Hq,
            "opt_input": self.opt_input,
            "opt_manager": self.opt_manager.to_dict(),
        }
        return d

    @classmethod
    def from_dict(cls, d):
        """Construct as far as possible the helper. Child class will need to update computer"""
        # creates the initial configuration of the OptHelper. Some options might
        # have changed over the course of the optimization (eg trust radius)

        helper = cls(d.get("opt_input"), params={}, silent=True)

        # We need to make sure that the new params EXACTLY matches what was exported.
        # No validation should occur after export as validation will interpret each exported
        # options as having been set explicitly by the user.
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
        """Create coordinates for optimization. print to logger"""

        make_internal_coords(self.molsys, self.params)
        self.show()

    def show(self):
        logger.info("Molsys:\n" + str(self.molsys))
        return

    @abstractmethod
    def _compute(self):
        """get energy gradient and hessian"""

    def compute(self):
        """Get the energy, gradient, and hessian. Project redundancies and apply constraints / forces"""

        try:
            if not self.molsys.intcos_present:
                # opt_manager.molsys is the same object as this molsys
                make_internal_coords(self.molsys, self.params)
                logger.debug("Molecular system after make_internal_coords:")
                logger.info(str(self.molsys))

            self._compute()
            logger.info("\n\t%s", print_geom_grad(self.geom, self.gX))
        except OptError as e:
            logger.critical("A critical error has occured: %s - %s", type(e), e, exc_info=True)
            raise e

    def take_step(self):
        """Must call compute before calling this method. Takes the next step."""
        self.opt_manager.error = None
        try:
            self.dq, self.step_str = self.opt_manager.take_step(
                self.fq, self._Hq, self.E, return_str=True
            )
        except AlgError as e:
            self.opt_manager.alg_error_handler(self._Hq, self.fq, e)
        except OptError as e:
            logger.critical("A critical error has occured: %s - %s", type(e), e, exc_info=True)
            raise e

        self.new_geom = self.molsys.geom
        self.step_num += 1
        self.opt_manager.step_number += 1

        self.show()
        return self.dq

    def test_convergence(self, str_mode=None):
        """Check the final two steps for convergence. If the algorithm uses linesearching, linesearches are not considered

        Returns
        -------
        bool

        """

        return self.opt_manager.converged(
            self.E, self.fq, self.dq, self.step_num, str_mode=str_mode
        )

    def close(self):
        del self._Hq
        del self.params
        return self.opt_manager.finish()

    @property
    def gX(self):
        return self._gX

    @gX.setter
    def gX(self, val):
        """gX must be set in order to perform an optimization. Cartesian only"""

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
            elif val.ndim == 1 and len(val) == dim**2:
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
        """Return final energy and geometry"""
        last_step = self.history.steps[-1]
        return last_step.E, last_step.geom

    def status(self, str_mode=None):
        """get string message describing state of optimizer

        Returns
        -------
        str :
            'FAILED' if unrecoverable
            'UNFINISHED-FAILED' if recoverable error
            'UNFINISHED' optimization has not converged
            'FINISHED' optimization converged
        """

        if self.opt_manager.error == "OptError":
            return "FAILED"

        if self.opt_manager.error == "AlgError":
            # self.HX = None
            self._Hq = None
            self.step_num = 0
            return "UNFINISHED-FAILED"

        if self.test_convergence(str_mode) is True:
            return "CONVERGED"

        return "UNFINISHED"


class CustomHelper(Helper):
    """Class allows for easy setup of OptKing. Accepts custom forces, energies,
    and hessians from user. User will need to write a loop to perform optimization.

    Examples
    --------

    >>> import qcengine as qcng
    >>> from qcelemental.models import Molecule, OptimizationInput
    >>> from qcelemental.models.common_models import Model
    >>> from qcelemental.models.procedures import QCInputSpecification
    >>> opt_input = {
    ...     "initial_molecule": {
    ...         "symbols": ["O", "O", "H", "H"],
    ...         "geometry": [
    ...             0.0000000000,
    ...             0.0000000000,
    ...             0.0000000000,
    ...             -0.0000000000,
    ...             -0.0000000000,
    ...             2.7463569188,
    ...             1.3013018774,
    ...             -1.2902977124,
    ...             2.9574871774,
    ...             -1.3013018774,
    ...             1.2902977124,
    ...             -0.2111302586,
    ...         ],
    ...         "fix_com": True,
    ...         "fix_orientation": True,
    ...     },
    ...     "input_specification": {
    ...         "model": {"method": "hf", "basis": "sto-3g"},
    ...         "driver": "gradient",
    ...         "keywords": {"d_convergence": "1e-7"},
    ...     },
    ...     "keywords": {"g_convergence": "GAU_TIGHT", "program": "psi4"},
    ... }
    >>> for step in range(30):
    ... # Compute one's own energy and gradient
    ... E, gX = optking.lj_functions.calc_energy_and_gradient(opt.geom, 2.5, 0.01, True)
    ... # Insert these values into the 'user' computer.
    ... opt.E = E
    ... opt.gX = gX
    ... opt.compute() # process input. Get ready to take a step
    ... opt.take_step()
    ... conv = opt.test_convergence()
    ... if conv is True:
    ...     print("Optimization SUCCESS:")
    ...     break
    ... else:
    ...     print("Optimization FAILURE:")
    >>> json_output = opt.close() # create an unvalidated OptimizationOutput like object
    >>> E = json_output["energies"][-1]

    Notes
    -----
    Overrides. ``gX``, ``HX``, and ``E`` to allow for user input.

    """

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

            if isinstance(mol_src, (psi4.qcdb.Molecule, psi4.core.Molecule)):
                self.molsys, self.opt_input = molsys.Molsys.from_psi4(mol_src)
            else:
                try:
                    self.molsys, self.opt_input = molsys.Molsys.from_psi4(
                        psi4.core.get_active_molecule()
                    )
                except Exception as error:
                    raise OptError("Failed to grab psi4 molecule as last resort") from error

        self.computer.molecule = self.opt_input
        self.build_coordinates()
        self.opt_manager = OptimizationManager(
            self.molsys, self.history, self.params, self.computer
        )
        self.opt_manager.step_number = 1

    @classmethod
    def from_dict(cls, d):
        helper = cls(d.get("opt_input"), params={}, silent=True)

        # We need to make sure that the new params EXACTLY matches what was exported.
        # No validation should occur after export as validation will interpret each exported
        # options as having been set explicitly by the user.
        helper.params = op.OptParams.from_internal_dict(d.get("params"))
        op.Params = helper.params
        # update with current information
        helper.molsys = molsys.Molsys.from_dict(d.get("molsys"))
        helper.history = history.History.from_dict(d.get("history"))
        helper.step_num = d.get("step_num")
        helper.irc_step_num = d.get("irc_step_num")
        helper._Hq = d.get("hessian")
        helper.computer = compute_wrappers.make_computer_from_dict("user", d.get("computer"))
        helper.opt_manager = OptimizationManager.from_dict(
            d["opt_manager"], helper.molsys, helper.history, helper.params, helper.computer
        )
        return helper

    def _compute(self):
        """The call to computer in this class is essentially a lookup for the value provided by
        the User."""

        if self.HX is None:
            if "hessian" in self.calculations_needed():
                if self.params.cart_hess_read:
                    self.HX = hessian.from_file(self.params.hessian_file)  # set ourselves if file
                    _ = self.computer.compute(self.geom, driver="hessian")
                    self.gX = self.computer.external_gradient
                    self.fq = self.molsys.gradient_to_internals(self.gX, -1.0)
                    self._Hq = self.molsys.hessian_to_internals(self.HX)
                    self.fq, self._Hq = self.molsys.apply_external_forces(self.fq, self._Hq)
                    self.fq, self._Hq = self.molsys.project_redundancies_and_constraints(
                        self.fq, self._Hq
                    )
                    self.HX = None
                    self.params.cart_hess_read = False
                    self.params.hessian_file = pathlib.Path("")
                else:
                    raise RuntimeError(
                        "Optking requested a hessian but was not provided one. "
                        "This could be a driver issue"
                    )
            elif self.step_num == 0:
                logger.debug("Guessing hessian")
                self._Hq = hessian.guess(self.molsys, guessType=self.params.intrafrag_hess)
                self.gX = self.computer.compute(self.geom, driver="gradient", return_full=False)
                self.fq = self.molsys.gradient_to_internals(self.gX, -1.0)
                self.fq, self._Hq = self.molsys.apply_external_forces(self.fq, self._Hq)
                self.fq, self._Hq = self.molsys.project_redundancies_and_constraints(
                    self.fq, self._Hq
                )

            else:
                logger.debug("Updating hessian")
                self.gX = self.computer.compute(self.geom, driver="gradient", return_full=False)
                self.fq = self.molsys.gradient_to_internals(self.gX, -1.0)
                self.fq, self._Hq = self.molsys.apply_external_forces(self.fq, self._Hq)
                self._Hq = self.history.hessian_update(self._Hq, self.fq, self.molsys)
                self.fq, self._Hq = self.molsys.project_redundancies_and_constraints(
                    self.fq, self._Hq
                )
        else:
            result = self.computer.compute(self.geom, driver="hessian")
            self.HX = self.computer.external_hessian
            self.gX = self.computer.external_gradient
            self.fq = self.molsys.gradient_to_internals(self.gX, -1.0)
            self._Hq = self.molsys.hessian_to_internals(self.HX)
            self.HX = None  # set back to None
            self.params.cart_hess_read = False
            self.fq, self._Hq = self.molsys.apply_external_forces(self.fq, self._Hq)
            self.fq, self._Hq = self.molsys.project_redundancies_and_constraints(self.fq, self._Hq)

    def calculations_needed(self):
        """Assume gradient is always needed. Provide tuple with keys for required properties"""

        # TODO revist once multiple opt_managers have been finalized. For now assume opt_helper
        # is correct since it knows the behavior of _compute().
        if self.params.cart_hess_read != self.opt_manager.params.cart_hess_read:
            self.opt_manager.params.cart_hess_read = self.params.cart_hess_read

        hessian_protocol = self.opt_manager.get_hessian_protocol(self.step_num)
        protocol = hessian_protocol.get("protocol")

        if protocol == "compute":
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
        """Set energy in self and computer"""
        # call parent classes setter. Weird python syntax. Class will always be CustomHelper
        # self.__class__ could be a child class type. (No child class currently)
        # super() and super(__class__, self.__class__) should be equivalent but the latter is required?
        super(__class__, self.__class__).E.__set__(self, val)
        self.computer.external_energy = val

    @HX.setter
    def HX(self, val):
        """Set hessian in self and computer"""
        super(__class__, self.__class__).HX.__set__(self, val)
        self.computer.external_hessian = val

    @gX.setter
    def gX(self, val):
        """Set gradient in self and computer"""
        super(__class__, self.__class__).gX.__set__(self, val)
        self.computer.external_gradient = val


class EngineHelper(Helper):
    """Perform an optimization using qcengine to compute properties. Use OptimizationInput to setup
    a molecular system

    Examples
    --------
    >>> import optking
    >>> import qcengine as qcng
    >>> from qcelemental.models import Molecule, OptimizationInput
    >>> from qcelemental.models.common_models import Model
    >>> from qcelemental.models.procedures import QCInputSpecification
    >>> molecule = Molecule.from_data(
    ...     symbols = ["O", "O", "H", "H"],
    ...     geometry = [
    ...         0.0000000000,
    ...         0.0000000000,
    ...         0.0000000000,
    ...         -0.0000000000,
    ...         -0.0000000000,
    ...         2.7463569188,
    ...         1.3013018774,
    ...         -1.2902977124,
    ...         2.9574871774,
    ...         -1.3013018774,
    ...         1.2902977124,
    ...         -0.2111302586,
    ...     ],
    ...     fix_com=True,
    ...     fix_orientation=True,
    ... )
    >>> model = Model(method="hf", basis="sto-3g")
    >>> input_spec = QCInputSpecification(
    ...     driver="gradient", model=model, keywords={"d_convergence": 1e-7}  # QC program options
    ... )
    >>> opt_input = OptimizationInput(
    ...     initial_molecule=molecule,
    ...     input_specification=input_spec,
    ...     keywords={"g_convergence": "GAU_TIGHT", "program": "psi4"},  # optimizer options
    ... )
    >>> opt = optking.EngineHelper(opt_input)
    >>> # opt_result = opt.optimize()  # optimize geometry - no interaction
    >>> for step in range(30):
    ...    # Compute one's own energy and gradient
    ...    opt.compute() # process input. Get ready to take a step
    ...    opt.take_step()
    ...    conv = opt.test_convergence()
    ...    if conv is True:
    ...        print("Optimization SUCCESS:")
    ...        break
    >>> else:
    ...     print("Optimization FAILURE:")
    >>> json_output = opt.close() # create an unvalidated OptimizationOutput like object
    >>> E = json_output["energies"][-1]

    """

    def __init__(self, optimization_input, **kwargs):
        """Create an EngineHelper. Can optimize interactivly or non interactively.

        Parameters
        ----------
        optimization_input: Union[qcelemental.procedures.OptimizationInput, dict]

        """
        if isinstance(optimization_input, qcel.models.OptimizationInput):
            self.opt_input = optimization_input.dict()
        elif isinstance(optimization_input, dict):
            tmp = qcel.models.OptimizationInput(**optimization_input).dict()
            self.opt_input = json.loads(qcel.util.serialization.json_dumps(tmp))
        # self.calc_name = self.opt_input['input_specification']['model']['method']

        super().__init__(self.opt_input["keywords"], **kwargs)
        self.molsys = molsys.Molsys.from_schema(self.opt_input["initial_molecule"])
        self.computer = optwrapper.make_computer(self.opt_input, "qc")
        self.computer_type = "qc"
        self.build_coordinates()
        self.opt_manager = OptimizationManager(
            self.molsys, self.history, self.params, self.computer
        )

    @classmethod
    def from_dict(cls, d):

        helper = cls(d.get("opt_input"), params={}, silent=True)

        # We need to make sure that the new params EXACTLY matches what was exported.
        # No validation should occur after export as validation will interpret each exported
        # options as having been set explicitly by the user.
        helper.params = op.OptParams.from_internal_dict(d.get("params"))
        op.Params = helper.params
        # update with current information
        helper.molsys = molsys.Molsys.from_dict(d.get("molsys"))
        helper.history = history.History.from_dict(d.get("history"))
        helper.step_num = d.get("step_num")
        helper.irc_step_num = d.get("irc_step_num")
        helper._Hq = d.get("hessian")
        helper.computer = compute_wrappers.make_computer_from_dict("qc", d.get("computer"))
        helper.opt_manager = OptimizationManager.from_dict(
            d["opt_manager"], helper.molsys, helper.history, helper.params, helper.computer
        )
        return helper

    def _compute(self):
        hessian_protocol = self.opt_manager.get_hessian_protocol(self.step_num)
        protocol = hessian_protocol["protocol"]
        requires = self.opt_manager.opt_method.requires()

        self._Hq, _, self.gX, self.E = get_pes_info(
            self._Hq, self.computer, self.molsys, self.history, self.params, protocol, requires
        )

        self.fq = self.molsys.gradient_to_internals(self.gX, -1.0)

    def optimize(self):
        """Creating an EngineHelper and calling optimize() is equivalent to calling the deprecated
        optimize_qcengine() with an OptimizationInput. However, EngineHelper will maintain
        its state."""
        self.opt_input = optimize(self.molsys, self.computer)
        # update molecular system
        # set E, g_x, and hessian to have their last values
        # set step_number
        # set self.history to match history.

""" Provides some of the high level functions and classes to run optimizations. This is a good starting place for anyone
looking to add features to the code to familarize themselves with the overall workings of optking.
Functions may be useful to users seeking greater control over the inner workings of optking than provided by the
OptHelpers. For instance if manually creating a molecular system or manually controlling / switching algorithms
on the fly.

See also `OptimizationAlgorithm <stepalgorithms.OptimizationAlgorithm>` and
`OptimizationInterface <stepalgorithms.OptimizationInterface>`for core functionality common to all the optimization
algorithms like backtransformation and displacement.

"""

import logging
from typing import Union

import numpy as np
from optking.compute_wrappers import ComputeWrapper
from optking.molsys import Molsys

from . import IRCfollowing, addIntcos, hessian, history, intcosMisc
from . import optparams as op
from . import stepAlgorithms
from . import testB, linesearch
from .exceptions import AlgError, OptError
from .printTools import print_array_string, print_geom_grad, print_mat_string
from . import log_name

logger = logging.getLogger(f"{log_name}{__name__}")


def optimize(o_molsys, computer):
    """Driver for OptKing's optimization procedure. Suggested that users use EngineHelper, CustomHelper, or
    one of the external interfaces (psi4 and qcengine) to perform a normal (full) optimization

    Parameters
    ----------
    o_molsys : cls
        optking molecular system
    computer : compute_wrappers.ComputeWrapper

    Returns
    -------
    float, float or dict
        energy and nuclear repulsion energy or MolSSI qc_schema_output as dict

    """

    logger = logging.getLogger(__name__)

    H = 0  # hessian in internals

    opt_history = history.History(op.Params)

    # Try to optimize one structure OR set of IRC points. OptError and all Exceptions caught below.
    try:
        converged = False
        if not o_molsys.intcos_present:
            make_internal_coords(o_molsys, op.Params)
            logger.debug("Molecular system after make_internal_coords:")
            logger.debug(str(o_molsys))

        opt_object = OptimizationManager(o_molsys, opt_history, op.Params, computer)
        # following loop may repeat over multiple algorithms OR over IRC points
        logger.info("\tStarting optimization algorithm.\n")
        while converged is not True:
            try:
                H, fq, energy = opt_object.start_step(H)
                dq = opt_object.take_step(fq, H, energy, return_str=False)
                converged = opt_object.converged(energy, fq, dq)
                opt_object.check_maxiter()  # raise error otherwise continue

            except AlgError as AF:
                opt_object.alg_error_handler(AF)
        qc_output = opt_object.finish(error=None)
        return qc_output

    except OptError as error:
        logger.error(error)
        return opt_object.opt_error_handler(error)

    except Exception as error:
        logger.error(error)
        return opt_object.unknown_error_handler(error)


class OptimizationManager(stepAlgorithms.OptimizationInterface):
    """Recommended use of Optking's Optimization Algorithms is to create this class and then loop
    over take_step. OptimizationFactory will either return the appropriate OptimizationAlgorithm or
    return itself if management of multiple algorithms is required (linesearching).
    Currently only 1 linesearch method is implemented so changing linesearch_method to anything that does
    not evaluate to None will turn linesearch on.

    This class' primary purpose is to abstract the interface for an OptimizationAlgorithm, Linesearch, or
    IRC so no special handling is needed for an IRC optimization as opposed to a NR optimization.

    # TODO add dynamic_level management here
    """

    _LINESEARCHES = {"ENERGY": linesearch.ThreePointEnergy}

    def __init__(self, molsys: Molsys, history_object: history.History, params: op.OptParams, computer: ComputeWrapper):
        super().__init__(molsys, history_object, params)
        self.direction: Union[np.ndarray, None] = None

        method = "IRC" if params.opt_type == "IRC" else params.step_type
        self.opt_method = optimization_factory(method, molsys, self.history, params)
        self.step_number = 0
        self.computer = computer
        self.linesearch_method = None
        self.stashed_hessian = None

        if params.linesearch:
            self.linesearch_method = OptimizationManager._LINESEARCHES["ENERGY"](molsys, self.history, params)
            self.opt_method.trust_radius_on = False
        self.requires = self.update_requirements()
        self.current_requirements = self.update_requirements()
        self.params = params
        self.erase_hessian = False
        self.check_linesearch = True
        self.error = None

    def to_dict(self):
        """Convert attributes to serializable form."""
        d = {
            "direction": self.direction,
            "step_number": self.step_number,
            "stashed_hessian": self.stashed_hessian,
            "requires": self.requires,
            "current_requirements": self.current_requirements,
            "erase_hessian": self.erase_hessian,
            "check_linesearch": self.check_linesearch,
            "error": self.error,
        }
        if self.linesearch_method:
            d["linesearch_method"] = self.linesearch_method.to_dict()

        if self.params.opt_type == "IRC":
            d["irc_object"] = self.opt_method.to_dict()

        return d

    @classmethod
    def from_dict(cls, d, molsys, history, params, computer):
        """Reload attributes from the provided dictionary. Create all necessary classes.

        To prevent duplication, OptHelper handles converting the molsys, history, params, and computer to/from dict
        """

        manager = cls(molsys, history, params, computer)
        manager.direction = d["direction"]
        method = "IRC" if params.opt_type == "IRC" else params.step_type
        manager.step_number = d["step_number"]
        manager.stashed_hessian = d["stashed_hessian"]
        manager.requires = d["requires"]
        manager.current_requirements = d["current_requirements"]
        manager.erase_hessian = d["erase_hessian"]
        manager.check_linesearch = d["check_linesearch"]
        manager.error = d["error"]

        if params.opt_type == "IRC":
            manager.opt_method = IRCfollowing.IntrinsicReactionCoordinate.from_dict(
                d["irc_object"], molsys, history, params
            )
        else:
            manager.opt_method = optimization_factory(
                method, molsys, history, params
            )  # Can just recreate with current history and params

        if d.get("linesearch_method"):
            manager.linesearch_method = OptimizationManager._LINESEARCHES["ENERGY"].from_dict(
                d["linesearch_method"], molsys, history, params
            )

        return manager

    def start_step(self, H: np.ndarray):
        """Initialize coordinates and perform any coordinate transformations of gradients and hessians.

        Returns
        -------
        H: np.ndarray
            2D. Hessian in appropriate coordinates
        f_q: np.ndarray
            1D. forces in appropriate coordinates
        E: float
            energy

        """

        # if optimization coordinates are absent, choose them. Could be erased after AlgError
        if not self.molsys.intcos_present:
            make_internal_coords(self.molsys, self.params)
            logger.debug("Molecular system after make_internal_coords:")
            logger.info(str(self.molsys))

        self.step_number += 1
        header = f"{'----------------------------':^74}"
        header += f"\n{'Taking A Step: Step Number %d' % self.step_number:^90}"
        header += f"\n{'----------------------------':^90}"
        logger.info(header)

        requirements = self.opt_method.requires()
        protocol = self.get_hessian_protocol()
        H, g_q, g_x, E = get_pes_info(H, self.computer, self.molsys, self.history, self.params, protocol, requirements)

        logger.info("%s", print_geom_grad(self.molsys.geom, g_x))

        f_q = -g_q
        f_q, H = self.molsys.project_redundancies_and_constraints(f_q, H)
        self.molsys.q_show()

        if self.params.test_B:
            testB.test_b(self.molsys)
        if self.params.test_derivative_B:
            testB.test_derivative_b(self.molsys)

        logger.info(print_array_string(f_q, title="Internal forces in au:"))
        return H, f_q, E

    def take_step(self, fq=None, H=None, energy=None, return_str=False, **kwargs):
        """Take whatever step (normal, linesearch, IRC, constrained IRC) is next.

        fq: Union[np.ndarray, None]
            forces
        H: Union[np.ndarray, None]
            hessian
        energy: Union[np.ndarray, None]
        return_str: bool
            if True return string with information about step (information is logged regardless)

        """

        self.current_requirements = self.update_requirements()

        if not self.params.linesearch:
            achieved_dq, returned_str = self.opt_method.take_step(fq, H, energy, return_str=True)
        else:
            if self.direction is None:
                self.direction = self.opt_method.step(fq, H, energy)

                self.linesearch_method.start(self.direction)

                if self.check_linesearch:
                    self.history.append(self.molsys.geom, energy, fq, self.molsys.gradient_to_cartesians(-1 * fq))
                    ls_energy = self.linesearch_method.expected_energy
                    dq_norm, unit_dq, grad, hess = self.opt_method.step_metrics(self.direction, fq, H)
                    self.history.append_record(ls_energy, self.direction, unit_dq, grad, hess)

                self.stashed_hessian = H
                self.check_linesearch = False

            achieved_dq, returned_str = self.linesearch_method.take_step(fq, H, energy, return_str=True)

            if self.linesearch_method.minimized:
                logger.info("Linesearch complete. Next step will compute a new direction")
                # cleanup
                self.linesearch_method.reset()
                self.direction = None
                self.check_linesearch = True

        self.requires = self.update_requirements()
        if return_str:
            return achieved_dq, returned_str
        return achieved_dq

    def update_requirements(self):
        """Get the current requirements for the next step.

        Notes
        -----
        If linesearching requirements can change. Always safe to provide a gradient regardless."""

        if self.direction is None:
            return self.opt_method.requires()
        else:
            return self.linesearch_method.requires()

    def converged(self, E, fq, dq, step_number=None, str_mode=None):
        """Test whether the optimization has finished. An optimization can only be declared converged
        If a gradient has been provided (linesearching cannot terminate an optimization)"""

        if step_number is None:
            step_number = self.step_number
        converged = False
        if not self.linesearch_method or self.check_linesearch:
            converged = self.opt_method.converged(dq, fq, step_number, str_mode=str_mode)
            if str_mode:
                return converged
            if converged is True:
                logger.info("\tConverged in %d steps!" % step_number)
                logger.info("\tFinal energy is %20.13f" % E)
                logger.info("\tFinal structure (Angstroms): \n" + self.molsys.show_geom())
        return converged

    def check_maxiter(self):
        """Check iterations < geom_maxiter. For IRC's check `total_steps_taken`."""

        if self.params.opt_type == "IRC":
            iterations = self.opt_method.total_steps_taken
        else:
            iterations = self.step_number

        # Hard quit if too many total steps taken (inc. all IRC points and algorithms).
        if iterations >= self.params.geom_maxiter:
            logger.error(
                "\tTotal number of steps (%d) exceeds maximum allowed (%d).\n" % (iterations, self.params.geom_maxiter)
            )
            raise OptError(
                "Maximum number of steps exceeded: {}.".format(self.params.geom_maxiter),
                "OptError",
            )

    def get_hessian_protocol(self):
        """Determine action to take for how to compute a hessian. Handles alternate IRC behavior

        Returns
        -------
        str: one of ('compute', 'update', 'guess', 'unneeded')

        """

        if "hessian" not in self.opt_method.requires():
            return "unneeded"

        if self.erase_hessian is True:
            self.erase_hessian = False
            return "compute" if self.params.full_hess_every > 0 else "guess"

        if self.params.cart_hess_read:
            return "compute"

        if self.step_number <= 1:
            if self.params.opt_type != "IRC":
                if self.params.full_hess_every > -1:  # compute hessian at least once.
                    protocol = "compute"
                else:
                    protocol = "guess"
            else:  # IRC
                protocol = "compute"
        else:
            if self.params.full_hess_every < 1:
                protocol = "update"
            elif (self.step_number - 1) % self.params.full_hess_every == 0:
                protocol = "compute"
            else:
                protocol = "update"

        return protocol

    def clear(self):
        """Reset history (inculding all steps) and molecule"""
        self.history.steps = []
        self.molsys.intcos = []
        self.step_number = 0
        self.history.steps_since_last_hessian = 0
        self.history.consecutive_backsteps = 0

    def alg_error_handler(self, error):
        """consumes an AlgError. Takes appropriate action"""
        logger.error(" Caught AlgError exception\n")
        eraseIntcos = False

        if error.linearBends:
            # New linear bends detected; Add them, and continue at current level.
            # from . import bend # import not currently being used according to IDE
            for l in error.linearBends:
                if l.bend_type == "LINEAR":  # no need to repeat this code for "COMPLEMENT"
                    iF = addIntcos.check_fragment(l.atoms, self.molsys)
                    F = self.molsys.fragments[iF]
                    intcosMisc.remove_old_now_linear_bend(l.atoms, F.intcos)
                    F.add_intcos_from_connectivity()
            eraseHistory = True
        elif self.params.dynamic_level == self.params.dynamic_level_max:
            logger.critical("\n\t Current algorithm/dynamic_level is %d.\n" % self.params.dynamic_level)
            logger.critical("\n\t Alternative approaches are not available or turned on.\n")
            raise OptError("Maximum dynamic_level reached.")
        else:
            self.params.dynamic_level += 1
            logger.warning("\n\t Increasing dynamic_level algorithm to %d.\n" % self.params.dynamic_level)
            logger.warning("\n\t Erasing old history, hessian, intcos.\n")
            eraseIntcos = True
            eraseHistory = True
            self.params.update_dynamic_level_params(self.params.dynamic_level)

        logger.info("Printing the parameters %s", self.params)

        if eraseIntcos:
            logger.warning(" Erasing coordinates.\n")
            for f in self.molsys.fragments:
                del f.intcos[:]
            self.molsys._dimer_intcos = []

        if eraseHistory:
            logger.warning(" Erasing history.\n")
            self.clear()
            self.erase_hessian = True

        self.error = "AlgError"

    def opt_error_handler(self, error):
        """OptError indicates an unrecoverable error. Print information and trigger cleanup."""
        logger.critical("\tA critical optimization-specific error has occured.")
        logger.critical("\tResetting all optimization options for potential queued jobs.\n")
        logger.exception("Error caught:" + str(error))
        # Dump histories if possible

        return self._exception_cleanup(error)

    def unknown_error_handler(self, error):
        """Unknown errors are not recoverable error. Print information and trigger cleanup."""
        logger.critical("\tA non-optimization-specific error has occurred.\n")
        logger.critical("\tResetting all optimization options for potential queued jobs.\n")
        logger.exception("Error Type:  " + str(type(error)))
        logger.exception("Error caught:" + str(error))

        return self._exception_cleanup(error)

    def _exception_cleanup(self, error):

        logger.info("\tDumping history: Warning last point not converged.\n" + self.history.summary_string())

        if self.params.opt_type == "IRC":
            logging.debug("\tDumping IRC points completed")
            self.opt_method.irc_history.progress_report()
        self.error = "OptError"

        return self.finish(error)

    def finish(self, error=None):

        rxnpath = None
        if self.params.opt_type == "IRC":
            self.opt_method.irc_history.progress_report()
            rxnpath = self.opt_method.irc_history.rxnpath_dict()
        else:
            logger.info("\tOptimization Finished\n" + self.history.summary_string())

        qc_output = prepare_opt_output(self.molsys, self.computer, rxnpath=rxnpath, error=error)
        self.clear()
        return qc_output


def optimization_factory(method, molsys, history_object, params=None):
    """create optimization algorithms. method may be redundant however params is allowed
     to be none so method is required explicitly

    Returns
    -------
    OptimizationAlgorithm"""
    ALGORITHMS = {
        "RFO": stepAlgorithms.RestrictedStepRFO,
        "P_RFO": stepAlgorithms.PartitionedRFO,
        "NR": stepAlgorithms.QuasiNewtonRaphson,
        "SD": stepAlgorithms.SteepestDescent,
        "IRC": IRCfollowing.IntrinsicReactionCoordinate,
    }

    return ALGORITHMS.get(method, stepAlgorithms.RestrictedStepRFO)(molsys, history_object, params)


def get_pes_info(
    H: np.ndarray,
    computer: ComputeWrapper,
    o_molsys: Molsys,
    opt_history: history.History,
    params: op.OptParams,
    hessian_protocol="update",
    requires=("energy", "gradient"),
):
    """Calculate, update, or guess hessian as appropriate. Calculate gradient and transform to the
    current coordinate system, pulls the gradient from hessian output if possible.

    Parameters
    ----------
    H: np.ndarray
        current Hessian
    computer : compute_wrappers.ComputeWrapper
    o_molsys : molsys.Molsys
    opt_history: history.History
    params : op.OptParams
    requires : list
        ("energy", "gradient", "hessian")
    hessian_protocol : str
        one of ("unneeded", "compute", "guess", "update")

    Returns
    -------
    np.ndarray
        hessian matrix (inteneral coordinates)
    np.ndarray:
        gradient vector (internal coordinates)
    np.ndarrray
        gradient vector (cartesian coordinates)
    float
        energy

    """

    if "gradient" in requires:
        driver = "gradient"
    else:
        driver = "energy"

    if hessian_protocol == "update":
        logger.info(f"Updating Hessian with {str(op.Params.hess_update)}")
        result = computer.compute(o_molsys.geom, driver=driver, return_full=False)
        g_x = np.asarray(result) if driver == "gradient" else None
        f_q = o_molsys.gradient_to_internals(g_x, -1.0)
        f_q, H = o_molsys.apply_external_forces(f_q, H)
        H = opt_history.hessian_update(H, f_q, o_molsys)
        g_q = -f_q
    else:
        if hessian_protocol == "compute" and not params.cart_hess_read:
            H, g_x = get_hess_grad(computer, o_molsys)

        elif hessian_protocol == "guess" or isinstance(H, int):
            logger.info(f"Guessing Hessian with {str(params.intrafrag_hess)}")
            H = hessian.guess(o_molsys, guessType=params.intrafrag_hess)
            result = computer.compute(o_molsys.geom, driver=driver, return_full=False)
            g_x = np.asarray(result) if driver == "gradient" else None
        elif hessian_protocol in ["unneeded"]:
            result = computer.compute(o_molsys.geom, driver=driver, return_full=False)
            g_x = np.asarray(result) if driver == "gradient" else None
        elif params.cart_hess_read:
            H = hessian.from_file(params.hessian_file)
            params.cart_hess_read = False
            params.hessian_file = None
        else:
            raise OptError("Encountered unknown value from get_hessian_protocol()")

        f_q = o_molsys.gradient_to_internals(g_x, -1)
        f_q, H = o_molsys.apply_external_forces(f_q, H)
        g_q = -f_q

    logger.info(print_mat_string(H, title="Hessian matrix"))
    return H, g_q, g_x, computer.energies[-1]


def get_hess_grad(computer, o_molsys):
    """Compute hessian and fetch gradient from output if possible. Perform separate gradient
    calculation if needed
    Parameters
    ----------
    computer: compute_wrappers.ComputeWrapper
    o_molsys: molsys.Molsys
    Returns
    -------
    tuple(np.ndarray, np.ndarray)
    Notes
    -----
    Hessian is in internals gradient is in cartesian
    """
    ret = computer.compute(o_molsys.geom, driver="hessian", return_full=True, print_result=False)
    h_cart = np.asarray(ret["return_result"]).reshape(o_molsys.geom.size, o_molsys.geom.size)
    try:
        logger.debug("Looking for gradient in hessian output")
        g_cart = ret["extras"]["qcvars"]["CURRENT GRADIENT"]
    except KeyError:
        logger.error("Could not find the gradient in qcschema")
        grad = computer.compute(o_molsys.geom, driver="gradient", return_full=False)
        g_cart = np.asarray(grad)
    # Likely not at stationary point. Include forces
    # ADDENDUM currently neglects forces term for all points - including non-stationary
    H = o_molsys.hessian_to_internals(h_cart)

    return H, g_cart


def make_internal_coords(o_molsys: Molsys, params: op.OptParams):
    """
    Add optimization coordinates to molecule system.
    May be called if coordinates have not been added yet, or have been removed due to an
    algorithm error (bend going linear, or energy increasing, etc.).

    Parameters
    ----------
    o_molsys: Molsys
        current molecular system.
    params: op.OptParams

    Returns
    -------
    o_molsys: Molsys
        The molecular system updated with internal coordinates.
    """
    # if params is None:
    #     params = op.Params
    logger.debug("\t Adding internal coordinates to molecular system")

    # Use covalent radii to determine bond connectivity.
    connectivity = addIntcos.connectivity_from_distances(o_molsys.geom, o_molsys.Z)
    logger.debug("Connectivity Matrix\n" + print_mat_string(connectivity))

    if params.frag_mode == "SINGLE":
        # Make a single, supermolecule.
        o_molsys.consolidate_fragments()  # collapse into one frag (if > 1)
        o_molsys.split_fragments_by_connectivity()  # separate by connectivity
        # increase connectivity until all atoms are connected
        o_molsys.augment_connectivity_to_single_fragment(connectivity)
        o_molsys.consolidate_fragments()  # collapse into one frag

        if params.opt_coordinates in ["INTERNAL", "REDUNDANT", "BOTH"]:
            o_molsys.fragments[0].add_intcos_from_connectivity(connectivity)

        if params.opt_coordinates in ["CARTESIAN", "BOTH"]:
            o_molsys.fragments[0].add_cartesian_intcos()

    elif params.frag_mode == "MULTI":
        # if provided multiple frags, then we use these.
        # if not, then split them (if not connected).
        if o_molsys.nfragments == 1:
            o_molsys.split_fragments_by_connectivity()

        if o_molsys.nfragments > 1:
            addIntcos.add_dimer_frag_intcos(o_molsys)
            # remove connectivity so that we don't add redundant coordinates
            # between fragments
            o_molsys.purge_interfragment_connectivity(connectivity)

        if params.opt_coordinates in ["INTERNAL", "REDUNDANT", "BOTH"]:
            for iF, F in enumerate(o_molsys.fragments):
                C = np.ndarray((F.natom, F.natom))
                C[:] = connectivity[o_molsys.frag_atom_slice(iF), o_molsys.frag_atom_slice(iF)]
                F.add_intcos_from_connectivity(C)

        if params.opt_coordinates in ["CARTESIAN", "BOTH"]:
            for F in o_molsys.fragments:
                F.add_cartesian_intcos()
    addIntcos.add_constrained_intcos(o_molsys)  # make sure these are in the set
    return


def prepare_opt_output(o_molsys, computer, rxnpath=False, error=None):
    logger.info("Preparing OptimizationResult")
    # Get molecule from most recent step. Add provenance and fill in non-required fills.
    # Turn back to dict
    computer.update_geometry(o_molsys.geom)
    final_molecule = computer.molecule

    qc_output = {
        "schema_name": "qcschema_optimization_output",
        "trajectory": computer.trajectory,
        "energies": computer.energies,
        "final_molecule": final_molecule,
        "extras": {},
        "success": True,
    }

    if error:
        qc_output.update({"success": False, "error": {"error_type": error.err_type, "error_message": error.mesg}})

    if rxnpath:
        print(rxnpath)
        qc_output["extras"]["irc_rxn_path"] = rxnpath
        qc_output["final_geometry"] = rxnpath[-2]["x"]
        qc_output["extras"]["final_irc_energy"] = rxnpath[-2]["energy"]

    return qc_output

"""Defines classes for minimization and transition state optimizations

Classes
-------
OptimizationInterface
    An abstract class that all optimization classes inherit from. Requires that all subclasses
    implement a ``take_step()`` method
OptimizationAlgorithm
    An abstract class that extends ``OptimizationInterface`` most algorithms (minimization or ts)
    inherit from this class.
QuasiNewtonOptimization
    Abstract class for performing Quasi-Newton optimizations, these methods utilize an approximate
    updated hessian to approximate the potential energy surface.
RFO
    An abstract class for performing various Rational Function Optimization. RFO itself is an
    extension of a QuasiNewtonOptimization.
NewtonRaphson
    Extension of QuasiNewtonOptimization
RestrictedStepRFO
    Default minimization algorithm.
ImageRFO
    Extension of RestrictedStepRFO for transition state finding
PartitionedRFO
    Default TS algorithm before OptKing ``v0.3.0``
SteepestDescent
    Basic gradient descent
ConjugateGradient
    Three varieties of CG are implemented Fletcher, Descent, and Polak. Fletcher is default.
Linesearch
    1-D linesearch by quadratic fit of energies

Functions
---------
step_matches_forces
    A basic attempt to determine whether the step taken matches the symmetry of the gradient.

See Also
--------
:py:class:`optking.optimize.OptimizationManager`
:py:class:`optking.opt_helper.CustomHelper`

Notes
-----
The optimization classes above may be created using the factory pattern through
:py:func:`optking.optimize.optimization_factory`. If linesearching or more advanced management of the optimization
process is desired an OptimizationManager should be created.
(More features are coming to the OptimizationManager)
"""

import logging
from math import fabs, sqrt
from abc import ABC, abstractmethod
from typing import Union

import numpy as np

from . import convcheck
from .displace import displace_molsys
from .exceptions import AlgError, OptError
from .history import History
from .linearAlgebra import symm_mat_eig
from .misc import is_dq_symmetric
from .molsys import Molsys
from .printTools import print_array_string, print_mat_string
from . import log_name
from . import op

logger = logging.getLogger(f"{log_name}{__name__}")


class OptimizationInterface(ABC):
    """Declares that ALL OptKing optimization methods/algorithms will have
    a self.take_step() method. All methods must be able to determine what the next step
    to take should be given a history. See take_step() docstring for details."""

    def __init__(self, molsys: Molsys, history: History, params: op.OptParams):
        """set history and molsys. Create a default params object if required.
        Individual params will be set as instance attributes by the child classes as needed

        Parameters
        ----------
        molsys: molsys.Molsys
        history: history.History
        params: op.OptParams"""

        self.molsys: Molsys = molsys
        self.history: History = history

        if not params:
            params = op.OptParams()
        self.print_lvl = params.print_lvl

    @abstractmethod
    def take_step(self, fq=None, H=None, energy=None, **kwargs):
        """Method skeleton (for example see OptimizationAlgorithm)
        1. Choose what kind of step should take place next
        2. take step
        3. displace molsys
        4. update history (trim history as applicable)
        5. return step taken
        """
        pass

    def step_metrics(self, dq, fq, H):
        # get norm |q| and unit vector in the step direction
        dq_norm = sqrt(np.dot(dq, dq))
        unit_dq = dq / dq_norm

        # get gradient and hessian in step direction
        grad = -1 * np.dot(fq, unit_dq)  # gradient, not force
        hess = np.dot(unit_dq, np.dot(H, unit_dq))

        logger.info("\t|target step| : %15.10f" % dq_norm)
        logger.info("\tgradient     : %15.10f" % grad)
        logger.info("\thessian      : %15.10f" % hess)

        return dq_norm, unit_dq, grad, hess


class OptimizationAlgorithm(OptimizationInterface):
    """The standard minimization and transition state algorithms inherit from here. Defines
    the take_step for those algorithms. Backstep and trust radius management are performed here.

    All child classes implement a ``step()`` method using the forces and Hessian to compute
    a step direction and possibly a step_length. ``trust_radius_on = False`` allows a child class
    to override a basic trust radius enforcement."""

    def __init__(self, molsys, history, params):
        super().__init__(molsys, history, params)
        self.trust_radius_on = True
        self.params = params
        # self.intrafrag_trust = params.intrafrag_trust
        # self.intrafrag_trust_max = params.intrafrag_trust_max
        # self.intrafrag_trust_min = params.intrafrag_trust_min
        # self.consecutive_backsteps_allowed = params.consecutive_backsteps_allowed
        # self.ensure_bt_convergence = params.ensure_bt_convergence
        # self.dynamic_level = params.dynamic_level
        # self.opt_type = params.opt_type
        # self.linesearch = params.linesearch

    @abstractmethod
    def requires(self, **kwargs):
        """Returns tuple with strings ('energy', 'gradient', 'hessian') for what the algorithm
        needs to compute a new point"""
        pass

    @abstractmethod
    def supports_trust_region(self):
        """Returns boolean for whether a trust region should be used with this method"""
        pass

    def expected_energy(self, dq, fq, H):
        """Compute the expected energy given the model for the step

        Parameters
        ----------
        step: float
            normalized step (unit length)
        grad: np.ndarray
            projection of gradient onto step
        hess: np.ndarray
            projection of hessian onto step
        """
        pass

    @abstractmethod
    def step(self, fq: np.ndarray, H: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Basic form of the algorithm"""
        pass

    def take_step(self, fq=None, H=None, energy=None, return_str=False, **kwargs):
        """Compute step and take step"""

        if len(fq) == 0:
            logger.warning("Forces are missing. Step is 0")
            return np.zeros(0)

        self.history.append(
            self.molsys.geom, energy, fq, self.molsys.gradient_to_cartesians(-1 * fq)
        )

        if self.backstep_needed(fq):
            dq = self.backstep()
        else:
            dq = self.step(fq, H)

        if self.trust_radius_on:
            dq = self.apply_intrafrag_step_scaling(dq)
            dq = self.apply_interfrag_step_scaling(dq)

        self.molsys.interfrag_dq_discontinuity_correction(dq)
        achieved_dq, achieved_dx, return_str = displace_molsys(
            self.molsys,
            dq,
            fq,
            ensure_convergence=self.params.ensure_bt_convergence,
            return_str=return_str,
            **self.params.__dict__,
        )
        dq_norm, unit_dq, projected_fq, projected_hess = self.step_metrics(achieved_dq, fq, H)
        delta_energy = self.expected_energy(dq, fq, H)
        logger.debug("\tProjected energy change: %10.10lf\n" % delta_energy)
        self.history.append_record(delta_energy, achieved_dq, unit_dq, projected_fq, projected_hess)

        dq_norm = np.linalg.norm(achieved_dq)
        logger.info("\tNorm of achieved step-size %15.10f" % dq_norm)
        dx_norm = np.linalg.norm(achieved_dx)
        logger.info("\tNorm of achieved step-size (cart): %15.10f" % dx_norm)

        # Before quitting, make sure step is reasonable.  It should only be
        # screwball if we are using the "First Guess" after the back-transformation failed.
        dq_norm = np.linalg.norm(achieved_dq[0 : self.molsys.num_intrafrag_intcos])
        if dq_norm > 5 * self.params.intrafrag_trust:
            raise AlgError("opt.py: Step is far too large.")

        if return_str:
            return achieved_dq, return_str
        return achieved_dq

    def apply_intrafrag_step_scaling(self, dq):
        """Apply maximum step limit by scaling."""
        trust = self.params.intrafrag_trust
        if sqrt(np.dot(dq, dq)) > trust:
            scale = trust / sqrt(np.dot(dq, dq))
            logger.info("\tStep length exceeds trust radius of %10.5f." % trust)
            logger.info("\tScaling displacements by %10.5f" % scale)
            dq *= scale
        return dq

    def apply_interfrag_step_scaling(self, dq):
        """Check the size of the interfragment modes.  They can inadvertently represent
        very large motions.

        Returns
        -------
        dq : scaled step according to trust radius
        """
        # loop over dimers with interfrag intcos
        for i, dimer_coords in enumerate(self.molsys.dimer_intcos):
            start = self.molsys.dimerfrag_1st_intco(i)
            # loop over individual intcos
            for j, intco in enumerate(dimer_coords.pseudo_frag.intcos):
                val = dq[start + j]
                if abs(val) > self.params.interfrag_trust:
                    logger.info(
                        f"Reducing step for Dimer({dimer_coords.A_idx + 1},{dimer_coords.B_idx + 1}), {intco}, {start + j}"
                    )
                    if val > 0:
                        dq[start + j] = self.params.interfrag_trust
                    else:
                        dq[start + j] = -1.0 * self.params.interfrag_trust
        return dq

    def backstep(self):
        """takes a partial step backwards. fq and H should correspond to the previous not current point

        Notes
        -----
        Take partial backward step.  Update current step in history.
        Divide the last step size by 1/2 and displace from old geometry.
        """

        logger.warning("\tRe-doing last optimization step - smaller this time.\n")
        self.history.consecutive_backsteps += 1

        # Calling function shouldn't let this happen; this is a check for developer
        if len(self.history.steps) < 2:
            raise OptError("Backstep called, but no history is available.")

        # Erase last, partial step data for current step.
        del self.history.steps[-1]

        # Get data from previous step.
        dq = self.history.steps[-1].Dq

        # Copy old geometry so displace doesn't change history
        geom = self.history.steps[-1].geom.copy()
        self.molsys.geom = geom

        # Compute new Dq and energy step projection.
        dq /= 2
        return dq

    def converged(self, dq, fq, step_number, str_mode=""):
        energies = [step.E for step in self.history.steps]
        conv_info = {
            "step_type": "standard",
            "energies": energies,
            "dq": dq,
            "fq": fq,
            "iternum": step_number,
        }
        converged = convcheck.conv_check(conv_info, self.params, str_mode=str_mode)
        if str_mode:
            return converged
        logger.info("\tConvergence check returned %s" % converged)

        return converged

    def backstep_needed(self, fq):
        """Simple logic for whether a backstep is advisable (or too many have been taken).

        Returns
        -------
        bool : True if a backstep should be taken
        """

        previous_is_decent = self.assess_previous_step()

        if previous_is_decent:
            self.history.consecutive_backsteps = 0
            return False

        backstep_remaining = (
            self.history.consecutive_backsteps < self.params.consecutive_backsteps_allowed
        )

        # This worked really well for a handful of tests and significantly worsened a
        # handful of tests needs more investigation before adopting
        # ALlow backstepping in early steps if it looks like we just overstepped.
        # if len(self.history.steps) > 2:
        #     unit_fq = fq / np.linalg.norm(fq)
        #     prev_fq = self.history.steps[-2].forces
        #     prev_unit_fq = prev_fq / np.linalg.norm(prev_fq)
        #     logger.debug(f"Overlap of forces with previous {np.dot(unit_fq, prev_unit_fq)}")
        #     if np.dot(unit_fq, prev_unit_fq) < -0.7:
        #         logger.info(
        #             "Force overlap indicates that the minimum has been overstepped. "
        #             "Performing backstep"
        #         )
        #         self.history.consecutive_backsteps += 1
        #         return True

        if len(self.history.steps) < 5:
            # ignore early issues.
            logger.info("\tNear start of optimization, so ignoring bad step.\n")
            return False

        if backstep_remaining:
            self.history.consecutive_backsteps += 1
            logger.info(
                "\tThis is consecutive backstep %d.\n",
                self.history.consecutive_backsteps,
            )
            return True  # Assumes that the client follows instructions

        if self.params.dynamic_level == 0:
            logger.info("Continuing despite bad step. dynamic level is 0")
            return False
        raise AlgError("Bad Step. Maximum number of backsteps taken")

    def assess_previous_step(self):
        """Determine whether the last step was acceptable, prints summary and change trust radius"""

        decent = True
        if len(self.history.steps) < 2:
            self.history.steps[-1].decent = decent
            return decent

        energy_change = self.history.steps[-1].E - self.history.steps[-2].E
        projected_change = self.history.steps[-2].projectedDE

        opt_step_report = "\n\tCurrent energy: %20.10lf\n" % self.history.steps[-1].E
        opt_step_report += "\tEnergy change for the previous step:\n"
        opt_step_report += "\t\tActual       : %20.10lf\n" % energy_change
        opt_step_report += "\t\tProjected    : %20.10lf\n" % projected_change

        logger.info("\tCurrent Step Report \n %s" % opt_step_report)

        energy_ratio = energy_change / projected_change
        logger.info("\tEnergy ratio = %10.5lf" % energy_ratio)

        if self.supports_trust_region() and not self.params.linesearch:
            # Predicted up. Actual down.  OK.  Do nothing.
            if projected_change > 0 and energy_ratio < 0.0:
                decent = True
            # Actual step is  up.
            elif energy_change > 0:
                logger.warning("\tEnergy has increased in a minimization.")
                self.decrease_trust_radius()
                decent = False
            # Predicted down.  Actual down.
            elif energy_ratio < 0.25:
                self.decrease_trust_radius()
                decent = True
            elif energy_ratio > 0.75:
                self.increase_trust_radius()
                decent = True

        self.history.steps[-2].decent = decent
        return decent

    def increase_trust_radius(self):
        """Increase trust radius by factor of 3"""
        maximum = self.params.intrafrag_trust_max
        if self.params.intrafrag_trust != maximum:
            new_val = self.params.intrafrag_trust * 3
            new_val = maximum if new_val > self.params.intrafrag_trust_max else new_val
            logger.info("\tEnergy ratio indicates good step.")
            logger.info("\tIntrafrag trust radius increased to %6.3g.", new_val)
            self.params.intrafrag_trust = new_val

        if self.params.frag_mode == "MULTI":
            maximum = self.params.interfrag_trust_max
            if self.params.interfrag_trust != maximum:
                new_val = self.params.interfrag_trust * 3
                new_val = maximum if new_val > self.params.interfrag_trust_max else new_val
                logger.info("\tEnergy ratio indicates good step.")
                logger.info("\tInterfrag trust radius increased to %6.3g.", new_val)
                self.params.interfrag_trust = new_val

    def decrease_trust_radius(self):
        """Scale trust radius by 0.25"""
        minimum = self.params.intrafrag_trust_min
        if self.params.intrafrag_trust != minimum:
            new_val = self.params.intrafrag_trust / 4
            new_val = minimum if new_val < minimum else new_val
            logger.warning("\tEnergy ratio indicates iffy step.")
            logger.warning("\tIntrafrag trust radius decreased to %6.3g.", new_val)
            self.params.intrafrag_trust = new_val

        if self.params.frag_mode == "MULTI":
            minimum = self.params.interfrag_trust_min
            if self.params.interfrag_trust != minimum:
                new_val = self.params.interfrag_trust / 4
                new_val = minimum if new_val < minimum else new_val
                logger.warning("\tEnergy ratio indicates iffy step.")
                logger.warning("\tInterfrag trust radius decreased to %6.3g.", new_val)
                self.params.interfrag_trust = new_val

    def update_history(self, delta_e, achieved_dq, unit_dq, projected_f, projected_hess):
        """Basic history update method. This should be expanded here and in child classes in
        future"""
        pass

    # def converged(self, step_number, dq, fq):
    #     energies = (self.history.steps[-1].E, self.history.steps[-2].E)  # grab last two energies
    #     converged = convcheck.conv_check(step_number, self.molsys, dq, energies)
    #     logger.info("\tConvergence check returned %s" % converged)
    #     return converged


class QuasiNewtonOptimization(OptimizationAlgorithm, ABC):
    def requires(self):
        return "energy", "gradient", "hessian"

    def supports_trust_region(self):
        return True

    def expected_energy(self, dq, fq, H):
        """Quadratic energy model"""
        dq_norm, unit_dq, proj_grad, proj_hess = self.step_metrics(dq, fq, H)
        return dq_norm * proj_grad + 0.5 * dq_norm**2 * proj_hess

    def take_step(self, fq=None, H=None, energy=None, return_str=False, **kwargs):
        if H is None or len(H) == 0:
            logger.warning("Missing Hessian. Step is 0")
            return np.zeros(0)
        return super().take_step(fq, H, energy, return_str)


class SteepestDescent(OptimizationAlgorithm):
    """Steepest descent with step size adjustment

    Notes
    -----
    dq = c * fq
    """

    def __init__(self, molsys, history, params):
        super().__init__(molsys, history, params)
        self.method = params.steepest_descent_type

    def requires(self):
        return "energy", "gradient"

    def supports_trust_region(self):
        return True

    def step_size_scalar(self, fq):
        """Perform two point calculation of steepest descent step_size"""

        methods = {
            "OVERLAP": self._force_overlap,
            "BARZILAI_BORWEIN": self._barzilai_borwein,
        }

        if len(self.history.steps) < 2:
            return 1
        return methods.get(self.method, self._force_overlap)(fq)

    def _force_overlap(self, fq):
        """
        Notes
        -----
        c = ((fq_k-1.fq)/||fq|| - ||fq||) / ||x_k - x_k-1||
        """

        sd_h = 1

        if len(self.history.steps) < 2:
            return sd_h

        old_fq = self.history.steps[-2].forces
        old_dq = self.history.steps[-2].Dq

        fq_norm = np.linalg.norm(fq)

        # Compute overlap of previous forces with current forces.
        old_unit_fq = old_fq / np.linalg.norm(old_fq)
        unit_fq = fq / fq_norm
        overlap = np.dot(old_unit_fq, unit_fq)
        logger.debug("\tOverlap of current forces with previous forces %8.4lf" % overlap)

        if overlap > 0.50:
            old_dq_norm = np.linalg.norm(old_dq)
            # Magnitude of previous force in step direction
            old_fq_norm = np.dot(old_fq, fq) / fq_norm
            sd_h = abs(old_fq_norm - fq_norm) / old_dq_norm

        return 1 / sd_h

    def _barzilai_borwein(self, fq):
        """Alternative step size choice for steepest descent
        Notes
        -----
        https://doi.org/10.1093/imanum/8.1.141"""
        old_fq = self.history.steps[-2].forces
        old_dq = self.history.steps[-2].Dq
        # delta gradient is negative delta forces. gk - gk-1 = fk-1 - fk
        delta_fq = old_fq - fq
        c = np.dot(old_dq, old_dq) / np.dot(old_dq, delta_fq)
        return c

    def step(self, fq, *args, **kwargs):
        logger.info("Taking Steepest Descent Step")
        sd_h = self.step_size_scalar(fq)
        dq = fq * sd_h
        return dq

    def expected_energy(self, dq, fq, H):
        """Quadratic energy model"""
        dq_norm, unit_dq, proj_grad, proj_hess = self.step_metrics(dq, fq, H)
        return dq_norm * proj_grad + 0.5 * dq_norm**2 * proj_hess


class ConjugateGradient(OptimizationAlgorithm):
    """ Implements the conjugate gradient algorithm.

    Notes
    -----
    The following varieties are implemented.

    #. Fletcher (from Fletcher's "Pratical Methods of Optimization, Vol. 1", Ch. 4, Pg. 63, Eqn. 4.1.4)
    #. descent (from Fletcher, Ch. 4, Pg. 66, Eqn. 4.1.11)
    #. Polak (from Fletcher, Ch. 4, Pg. 66, Eqn. 4.1.12)
    """

    def __init__(self, molsys, history, params):
        super().__init__(molsys, history, params)
        self.method = params.conjugate_gradient_type

    def requires(self):
        return "energy", "gradient"

    def step(self, fq, *args, **kwargs):
        logger.info("Taking Conjugate Gradient Step")

        if len(self.history.steps) < 2:
            return fq

        # Previous step
        prev_dq = self.history.steps[-2].Dq
        # Previous gradient
        prev_fq = self.history.steps[-2].forces

        # Default method
        if self.method == "FLETCHER":  # Fletcher-Reeves
            beta_numerator = np.dot(fq, fq)
            beta_denominator = np.dot(prev_fq, prev_fq)

        elif self.method == "POLAK":  # Polak-Ribiere
            beta_numerator = np.dot(fq, fq - prev_fq)
            beta_denominator = np.dot(prev_fq, prev_fq)

        elif self.method == "DESCENT":
            beta_numerator = np.dot(fq, fq)
            beta_denominator = np.dot(prev_fq, prev_dq)

        beta_fq = beta_numerator / beta_denominator
        # logger.info("\tfq:\n\t" + print_array_string(fq))
        dq = fq + beta_fq * prev_dq
        # logger.info("\tdq:\n\t" + print_array_string(dq))
        return dq

    def expected_energy(self, dq, fq, H):
        """Quadratic energy model"""
        dq_norm, unit_dq, proj_grad, proj_hess = self.step_metrics(dq, fq, H)
        return dq_norm * proj_grad + 0.5 * dq_norm**2 * proj_hess

    def supports_trust_region(self):
        return False


class QuasiNewtonRaphson(QuasiNewtonOptimization):
    def step(self, fq=None, H=None, *args, **kwargs):
        """Basic NR step. Hinv fq = dq
        Parameters
        ----------
        fq: np.ndarray
        H: np.ndarray"""
        Hinv = np.linalg.pinv(H, hermitian=True)
        dq = np.dot(Hinv, fq)
        return dq


class RFO(QuasiNewtonOptimization, ABC):
    """Standard RFO and base class for RS_RFO, P_RFO, RS_PRFO #TODO"""

    def __init__(self, molsys, history, params):
        super().__init__(molsys, history, params)
        # self.simple_step_scaling = params.simple_step_scaling
        # self.rfo_follow_root = params.rfo_follow_root
        self.rfo_root = self.params.rfo_root
        self.old_root = self.rfo_root
        # self.rfo_normalization_max = params.rfo_normalization_max

    def expected_energy(self, dq, fq, H):
        """RFO model - 2x2 Pade Approximation"""
        dq_norm, unit_dq, proj_grad, proj_hess = self.step_metrics(dq, fq, H)
        return (dq_norm * proj_grad + 0.5 * dq_norm**2 * proj_hess) / (1 + dq_norm**2)

    @staticmethod
    def build_rfo_matrix(lower, upper, fq, H):
        dim = upper - lower
        matrix = np.zeros((dim + 1, dim + 1))  # extra row and column for augmenting with gradient
        matrix[:dim, :dim] = H[lower:upper, lower:upper]
        matrix[:dim, -1] = matrix[-1, :dim] = -fq[lower:upper]
        return matrix

    def _intermediate_normalize(self, eigenvectors):
        """Intermediate normalize the eigenvectors to have a final element of 1.
        One of these eigenvectors will be the step direction.
        Any eigenvector that cannot be normalized (due to small divisors) is left untouched but
        should not be used to take a step.
        """

        # normalization constants and max values are reshaped to column vectors so that they can be applied
        # element wise across the eigenvectors (rows)
        i_norm = np.where(
            np.abs(eigenvectors[:, -1]) > 1.0e-10, eigenvectors[:, -1], 1
        )  # last element or 1
        i_norm = i_norm.reshape(-1, 1)
        tmp = eigenvectors / i_norm
        max_values = (np.amax(np.abs(tmp), axis=1)).reshape(-1, 1)
        normalized_evects = np.where(
            max_values < self.params.rfo_normalization_max, tmp, eigenvectors
        )
        return normalized_evects


class RestrictedStepRFO(RFO):
    """Rational Function approximation (or 2x2 Pade approximation) for step.
    Uses Scaling Parameter to adjust step size analagous to the NR hessian shift parameter."""

    def __init__(self, molsys, history, params):
        super().__init__(molsys, history, params)
        # self.rsrfo_alpha_max = params.rsrfo_alpha_max
        # self.accept_symmetry_breaking = params.accept_symmetry_breaking
        self.alpha = 1.0
        self.unnormalized = np.zeros(molsys.num_intcos)
        self.simple_step_scaling = params.simple_step_scaling

    def step(self, fq, H, *args, **kawrgs):
        """The step is an eigenvector of the gradient augmented Hessian. Looks for the
        lowest eigenvalue / eigenvector that is normalizable. If rfo_follow_root the previous root/step
        is considered first"""

        logger.debug("\tTaking RFO optimization step.")

        # Build the original, unscaled RFO matrix.
        RFOmat = RFO.build_rfo_matrix(0, len(H), fq, H)  # use entire hessian for RFO matrix

        if self.params.simple_step_scaling:
            e_vectors, e_values = self._intermediate_normalize(RFOmat)
            rfo_root = self._select_rfo_root(
                self.history.steps[-2].followedUnitVector,
                e_vectors,
                e_values,
                fq,
                alpha_iter=0,
            )
            dq = e_vectors[rfo_root, :-1]  # remove normalization constant
            converged = False
        else:
            # converge alpha to select step length. Same procedure as above.
            converged, dq = self._solve_rs_rfo(RFOmat, H, fq)

        # if converged, trust radius has already been applied through alpha
        self.trust_radius_on = not converged
        logger.debug("\tFinal scaled step dq:\n\n\t" + print_array_string(dq))
        return dq

    def _solve_rs_rfo(self, RFOmat, H, fq):
        """Performs an iterative process to determine alpha step scaling parameter and"""

        converged = False
        alpha = 1.0  # scaling factor for RS-RFO, scaling matrix is sI
        alpha_iter = -1
        dim = len(fq)
        max_rfo_iter = 25  # max. # of iterations to try to converge RS-RFO
        Hevals, Hevects = symm_mat_eig(H)  # Need for computing alpha at end of loop
        follow_root = self.params.rfo_follow_root
        trust = self.params.intrafrag_trust

        last_evect = np.zeros(dim)
        if self.params.rfo_follow_root and len(self.history.steps) > 1:
            # RFO vector from previous geometry step
            last_evect[:] = self.history.steps[-2].followedUnitVector
        rfo_step_report = ""

        # initialize to last step. Will be initialized to meaningful step or OptError will be
        # raised
        dq = last_evect
        best_alpha = {"alpha": 1.0, "steplen": 1e10, "dq": np.zeros(dim)}

        while not converged and alpha_iter < max_rfo_iter:
            alpha_iter += 1

            # If we exhaust iterations without convergence, then bail on the
            # restricted-step algorithm. Use the best estimate for alpha and apply crude scaling
            # on top of that.
            if alpha_iter == max_rfo_iter:
                logger.warning("\tFailed to converge alpha. Doing simple step-scaling instead.")
                alpha = best_alpha["alpha"]
                dq = best_alpha["dq"]
                break

            try:
                SRFOevals, SRFOevects = self._scale_and_normalize(RFOmat, alpha)
            except OptError as e:
                alpha = 1.0
                logger.warning(
                    "Could not converge alpha due to a linear algebra error. Continuing with simple step scaling"
                )
                break

            # Determine best (lowest eigenvalue), acceptable root and take as step
            rfo_root = self._select_rfo_root(last_evect, SRFOevects, SRFOevals, fq, alpha_iter)
            dq = SRFOevects[rfo_root][:-1]  # omit last column
            step_len = np.linalg.norm(dq)
            # If alpha explodes, give up on iterative scheme

            if fabs(alpha) > self.params.rsrfo_alpha_max:
                logger.debug("Scaling parameter has exploded. Aborting")
                converged = False
                alpha_iter = max_rfo_iter - 1
            elif step_len < (trust + 1e-5):
                converged = True

            # When alpha blows up, the proposed, tiny step if often closer to the trust radius
            # than the previous, reasonable step that is larger than the trust radius
            # Don't store these alphas as "best"
            step_closer = np.abs(step_len - trust) < np.abs(best_alpha["steplen"] - trust)
            alpha_suitable = fabs(alpha) < self.params.rsrfo_alpha_max
            if step_closer and alpha_suitable:
                best_alpha["alpha"] = alpha
                best_alpha["dq"] = dq
                best_alpha["steplen"] = step_len

            alpha, print_out = self._update_alpha(
                alpha, step_len, alpha_iter, dq, fq, Hevects, Hevals
            )
            rfo_step_report += print_out

        # end alpha RS-RFO iterations
        self.alpha = alpha
        logger.debug(rfo_step_report)
        self.params.rfo_follow_root = follow_root
        return converged, dq

    def _update_alpha(self, alpha, step_len, alpha_iter, dq, fq, Hevects, Hevals):
        rfo_step_report = ""

        if alpha_iter == 0 and not self.params.simple_step_scaling:
            logger.debug("\tDetermining step-restricting scale parameter (alpha) for RS-RFO.")

        def print_alpha_update(alpha_iter, step_len, alpha, rfo_root, deriv, _lambda):
            # Standard DEBUG printing
            header = "\n\t\t{:^6s}{:^12s}{:^15s}{:^11s}"
            header_args = ["Iter", "|step|", "alpha", "rfo_root"]
            header_len = 52
            args = [alpha_iter + 1, step_len, alpha, self.rfo_root + 1]
            table = "\n\t\t{:^6d}{:^12.5g}{:^15.5g}{:^11d}"

            if self.print_lvl >= 2:
                # Additional debug printing
                args += [deriv, _lambda]
                header += "{:^18s}{:^12s}"
                header_args += "d(|step|)/d(alpha)", "lambda"
                header_len = 84
                args += [deriv, _lambda]
                table += "{:^20.5g}{:^12.5g}"

            report = ""

            if alpha_iter == 0:
                report = (
                    header.format(*header_args) + "\n\t" + "-" * header_len + table.format(*args)
                )

            if alpha_iter > 0 and not self.params.simple_step_scaling:
                report = table.format(*args)

            return report

        _lambda = -1 * fq @ dq
        # Calculate derivative of step size wrt alpha.
        tval = np.einsum("ij, j -> i", Hevects, fq) ** 2 / (Hevals - _lambda * alpha) ** 3
        tval = np.sum(tval)
        deriv = 2 * _lambda / (1 + alpha * step_len**2) * tval

        rfo_step_report += print_alpha_update(
            alpha_iter, step_len, alpha, self.rfo_root, deriv, _lambda
        )

        # Calculate new scaling alpha value.
        # Equation 20, Besalu and Bofill, Theor. Chem. Acc., 1998, 100:265-274
        alpha += 2 * (self.params.intrafrag_trust * step_len - step_len**2) / deriv

        return alpha, rfo_step_report

    def _scale_and_normalize(self, RFOmat, alpha=1.0):
        """Scale the RFO matrix given alpha. Compute eigenvectors and eigenvalaues. Peform normalization
        and report values

        Parameters
        ----------
        RFOmat : np.ndarray
        alpha: float
        """

        dim1, dim2 = RFOmat.shape
        SRFOmat = np.zeros((dim1, dim2))  # For scaled RFO matrix.

        # scale RFO matrix leaving the last row unchanged, compute eigenvectors
        SRFOmat[:-1, :-1] = RFOmat[:-1, :-1] / alpha
        # in case alpha goes negative, this prevents warnings
        rootAlpha = np.sign(alpha) * (np.abs(alpha)) ** 0.5
        SRFOmat[-1, :-1] = RFOmat[-1, :-1] / rootAlpha
        SRFOmat[:-1, -1] = RFOmat[:-1, -1] / rootAlpha
        SRFOevals, SRFOevects = symm_mat_eig(SRFOmat)

        self.prenormalized = SRFOevects[:, :]
        SRFOevects = self._intermediate_normalize(SRFOevects)

        # transform step back.
        scale_mat = np.diag(np.repeat(1 / rootAlpha, dim1))
        scale_mat[-1, -1] = 1
        # SRFOevects = np.einsum("ij, jk -> ik", scale_mat, SRFOevects)
        SRFOevects = np.transpose(scale_mat @ np.transpose(SRFOevects))

        if self.print_lvl >= 4:
            logger.debug("\tScaled RFO matrix.\n\n" + print_mat_string(SRFOmat))
            logger.debug("\tEigenvectors of scaled RFO matrix.\n\n" + print_mat_string(SRFOevects))
            logger.debug(
                "\tEigenvalues of scaled RFO matrix.\n\n\t" + print_array_string(SRFOevals)
            )
            logger.debug(
                "\tFirst eigenvector (unnormalized) of scaled RFO matrix.\n\n\t"
                + print_array_string(SRFOevects[0])
            )
            logger.debug(
                "\tAll intermediate normalized eigenvectors (rows).\n\n"
                + print_mat_string(SRFOevects)
            )

        return SRFOevals, SRFOevects

    def _select_rfo_root(self, last_evect, SRFOevects, SRFOevals, fq, alpha_iter=0):
        """If root-following is turned off (default for first alpha iteration), then take the eigenvector with the
        lowest eigenvalue beginning at self.rfo_root.
        If it is the first iteration, then do the same (lowest eigenvalue).
        In subsequent steps (alpha iterations), overlaps will be checked.

        Sets self.rfo_follow_root = True the first time _select_rfo_root is called
        rfo_follow_root is reset to the original value at the end of a RFO step.

        Parameters
        ----------
        last_evect: np.ndarray
        SRFOevects: np.ndarray
        SRFOevals: np.ndarray
        alpha_iter: int
        """

        rfo_root = self.old_root
        if not self.params.rfo_follow_root or np.array_equal(last_evect, np.zeros(len(last_evect))):
            # Determine root only once at beginning. This root will be followed in subsequent alpha iterations
            if alpha_iter == 0:
                logger.debug("\tChecking RFO solution %d." % 1)

                for i in range(self.rfo_root, len(SRFOevals)):
                    acceptable = self._check_rfo_eigenvector(SRFOevects[i], fq, i)
                    if acceptable is False:
                        continue

                    rfo_root = i
                    break

                else:
                    # no good root found, using the default
                    rfo_root = self.rfo_root

            # Save initial root. 'Follow' during the RS-RFO iterations.
            self.params.rfo_follow_root = True

        else:  # Do root following.
            # Find maximum overlap. Dot only within H block.
            overlaps = np.einsum("ij, j -> i", SRFOevects[:-1, :-1], last_evect)
            logger.info("Best Fits for overlap:\n%s", overlaps)
            bestfit = np.argmax(overlaps)

            if bestfit != self.old_root:
                logger.info("\tRoot-following has changed rfo_root value to %d." % (bestfit + 1))
                rfo_root = bestfit

        if alpha_iter == 0:
            logger.info("\tUsing RFO solution %d." % (rfo_root + 1))

        # Print only the lowest eigenvalues/eigenvectors
        if self.params.print_lvl >= 2:
            for i, eigval in enumerate(SRFOevals):
                if i >= self.rfo_root and eigval > -1e-6:
                    break

                template = "\n\tScaled RFO eigenvalue %d:\n\t%15.10lf (or 2*%-15.10lf)\n"
                print_out = template.format(*(i + 1, eigval, eigval / 2))
                print_out += "\n\teigenvector:\n\t"
                print_out += print_array_string(SRFOevects[i])
                logger.info(print_out)

        self.old_root = rfo_root
        return rfo_root

    def _check_rfo_eigenvector(self, vector, fq, index):
        """Check whether eigenvector of RFO matrix is numerically acceptable for following and of
        proper symmetry. Double checks max values #TODO add real symmetry check

        Parameters
        ----------
        vector: np.ndarray
            an eigenvector of the rfo matrix
        index: int
            sorted index for the vector (eigenvector)
        """

        def reject_root(mesg):
            logger.warning("\tRejecting RFO root %d because %s", index + 1, mesg)
            return False

        # Check symmetry of root. Leave True if unessecary. Not currently functioning
        symmetric = (
            True
            if not self.params.accept_symmetry_breaking
            else is_dq_symmetric(self.molsys, vector[:-1])
        )

        if not symmetric:
            return reject_root("it breaks the molecular point group")

        if not step_matches_forces(vector, fq):
            return reject_root("Step does not qualitatively match the forces")

        if np.abs(vector[-1]) < 1e-10:
            return reject_root(f"Normalization gives large value. denominator is {vector[-1]}")

        if np.amax(np.abs(vector)) > self.params.rfo_normalization_max:
            return reject_root(
                f"Normalization gives large value. largest value is {np.amax(np.abs(vector))}"
            )

        return True


class PartitionedRFO(RFO):
    """Partitions the gradient augmented Hessian into eigenvectors to maximize along (direction of the TS)
    and minimize along all other directions. Rational Function or (2x2 Pade approximation)"""

    def supports_trust_region(self):
        """Historically has not supported a trust region. TODO test turning on"""
        return False

    def step(self, fq, H, *args, **kwargs):
        hdim = len(fq)  # size of Hessian

        # Diagonalize H (technically only have to semi-diagonalize)
        h_eig_values, h_eig_vectors = symm_mat_eig(H)
        hess_diag = np.diag(h_eig_values)

        if self.print_lvl > 2:
            logger.info("\tEigenvalues of Hessian\n\n\t%s", print_array_string(h_eig_values))
            logger.info("\tEigenvectors of Hessian (rows)\n%s", print_mat_string(h_eig_vectors))
            logger.debug(
                "\tFor P-RFO, assuming rfo_root=1, maximizing along lowest eigenvalue of Hessian."
            )
            logger.debug("\tLarger values of rfo_root are not yet supported.")

        rfo_root = 0
        # self._select_rfo_root() TODO

        # number of degrees along which to maximize; assume 1 for now
        mu = 1
        fq_prime = np.dot(h_eig_vectors, fq)  # gradient transformation
        logger.info(
            "\tInternal forces in au, in Hevect basis:\n\n\t" + print_array_string(fq_prime)
        )

        # Build RFO max and Min. Augments each partition of Hessian with corresponding gradient values
        # The lowest mu eigenvalues / vectors will be maximized along. All others will be minimized
        maximize_rfo = RFO.build_rfo_matrix(rfo_root, mu, fq_prime, hess_diag)
        minimize_rfo = RFO.build_rfo_matrix(mu, hdim, fq_prime, hess_diag)
        logger.info("\tRFO max\n%s", print_mat_string(maximize_rfo))
        logger.info("\tRFO min\n%s", print_mat_string(minimize_rfo))

        rfo_max_evals, rfo_max_evects = symm_mat_eig(maximize_rfo)
        rfo_min_evals, rfo_min_evects = symm_mat_eig(minimize_rfo)
        rfo_max_evects = self._intermediate_normalize(rfo_max_evects)
        rfo_min_evects = self._intermediate_normalize(rfo_min_evects)
        logger.info("\tRFO min eigenvalues:\n\n\t%s" + print_array_string(rfo_min_evals))
        logger.info("\tRFO max eigenvalues:\n\n\t%s" + print_array_string(rfo_max_evals))
        logger.debug("\tRFO max eigenvectors (rows):\n%s", print_mat_string(rfo_max_evects))
        logger.debug("\tRFO min eigenvectors (rows):\n%s", print_mat_string(rfo_min_evects))

        p_vec = rfo_max_evects[mu, :mu]
        n_vec = rfo_min_evects[rfo_root, : hdim - mu]

        # Combines the eignvectors from RFO max and min
        prfo_evect = np.zeros(hdim)
        prfo_evect[: len(p_vec)] = p_vec
        prfo_evect[len(p_vec) :] = n_vec

        prfo_step = np.dot(h_eig_vectors.transpose(), prfo_evect)

        logger.info(
            "\tRFO step in Hessian Eigenvector Basis\n\n\t" + print_array_string(prfo_evect)
        )
        logger.info("\tRFO step in original Basis\n\n\t" + print_array_string(prfo_step))

        return prfo_step

    def _select_rfo_root(self):
        """TODO: use rfo_root to decide which eigenvectors are moved into the max/mu space.
        if not rfo_follow_root or len(oHistory.steps) < 2:
            rfo_root = op.Params.rfo_root
            printxopt("\tMaximizing along %d lowest eigenvalue of Hessian.\n" % (rfo_root+1) )
        else:
            last_iter_evect = history[-1].Dq
            dots = np.array([v3d.dot(h_eig_vectors[i],last_iter_evect,hdim) for i in range(hdim)], float)
            rfo_root = np.argmax(dots)
            printxopt("\tOverlaps with previous step checked for root-following.\n")
            printxopt("\tMaximizing along %d lowest eigenvalue of Hessian.\n" % (rfo_root+1) )
        """
        raise NotImplementedError(
            "Partitioned RFO only follows the lowest eigenvalue / vector currently"
        )


class ImageRFO(RestrictedStepRFO):
    def __init__(self, molsys, history, params):
        super().__init__(molsys, history, params)
        self.image_eval = 0  # transformed eigenvalue
        self.w_tv = np.zeros(
            molsys.num_intcos
        )  # hessian eigenvector corresponding to the transition vector vt
        self.h_tv = 0.0
        self.alpha = 1.0

    def supports_trust_region(self):
        return True

    def step(self, fq, H, *args, **kwargs):
        H_evals, H_evects = symm_mat_eig(H)

        logger.info("Hessian eigenvalues %s", print_array_string(H_evals))

        # Takes the smallest eigenvalue and the smallest eigenvector and transforms the gradient
        # and hessian so that the reaction mode is being minimized not maximimized
        logger.info("Transforming the PES with image function to search for a saddlepoint")

        # choose the vector more intelligently - smallest nonzero value (even if 0?)
        reduced_selection = H_evals[np.where(np.abs(H_evals) > 1e-7)]
        self.h_tv = reduced_selection[0]  # now we can find the smallest eigenvalue
        # Get eigenvector by searching Hevals for self.h_tv
        self.w_tv = H_evects[np.where(H_evals == self.h_tv), :]
        # self.h_tv = H_evals[0]
        # not a vector product. Matrix product with the two eigenvectors
        householder_op = np.eye(len(fq)) - 2 * self.w_tv.reshape(-1, 1) @ self.w_tv.reshape(1, -1)
        fq_image = householder_op @ fq
        H_image = householder_op @ H

        logger.debug("eigenvalue of inverted mode is %s", self.h_tv)
        logger.debug("Forces transformed with image function %s", print_array_string(fq_image))

        # Use entire matrix 0, len(H). No need to partition
        RFO_image_mat = RFO.build_rfo_matrix(0, len(H), fq_image, H_image)

        # same as RS-RFO above.
        if self.params.simple_step_scaling:
            e_vectors, e_values = self._intermediate_normalize(RFO_image_mat)
            self.rfo_root = self._select_rfo_root(
                self.history.steps[-2].followedUnitVector,
                e_vectors,
                e_values,
                fq_image,
            )
            dq = e_vectors[self.rfo_root, :-1]
            self.image_eval = e_values[self.rfo_root]
            converged = False
        else:
            # converge alpha to select step length. Same procedure as above.
            converged, dq = self._solve_rs_rfo(RFO_image_mat, H_image, fq_image)

        # if converged, trust radius has already been applied through alpha
        self.trust_radius_on = not converged
        logger.debug("\tFinal scaled step dq:\n\n\t" + print_array_string(dq))
        return dq

    # def expected_energy(self, dq, fq, H):
    #     This seems to just continually shrink the trust radius into oblivion.
    #     Never finish optimizations
    #     the eigenvalue of the RFO matrix is found by transforming the negative mode to positive
    #     transform back to predict the energy change for going uphill
    #     logger.info("%s", self.w_tv)
    #     logger.info("%s", fq)
    #     fq_tv = self.w_tv @ -fq  # B&B label with f but seems to be overlap with gradient
    #     # fq_tv_2 = self.w_tv @ fq
    #     dq_tv = self.w_tv @ dq
    #     eigval = self.image_eval + (2 * fq_tv * dq_tv + self.h_tv * dq_tv **2) / (1 + self.alpha * dq @ dq)
    #     # eigval_2 = self.image_eval + (2 * fq_tv_2 * dq_tv + self.h_tv * dq_tv **2) / (1 + self.alpha * dq @ dq)

    #     logger.info("Transformed lambda %f", self.image_eval)
    #     logger.info("untransformed lambda %f", eigval)
    #     logger.info("fq_tv %s", fq_tv)
    #     logger.info("dq_tv %s", dq_tv)
    #     # logger.info("untransformed lambda2 %f", eigval_2)

    #     v_tv = self.prenormalized[self.rfo_root, :-1]  # final value in rfo eigenvecor that gets normalized to 1.
    #     u_tv = self.prenormalized[self.rfo_root + 1, -1]
    #     norm_2 = u_tv**2 + v_tv @ (np.eye(len(v_tv)) * self.alpha) @ v_tv

    #     logger.info("intermediate normalization coefficient %f", u_tv)
    #     # logger.info("second normalization coefficient %f", norm_2)

    #     logger.info("Value: %s", 0.5 * eigval / u_tv**2)
    #     return (0.5 * eigval / (u_tv**2)) / (1 + dq @ (np.eye(len(v_tv)) * self.alpha) @ dq)

    def assess_previous_step(self):
        """Determine whether the last step was acceptable, prints summary and change trust radius"""

        decent = True
        if len(self.history.steps) < 2:
            self.history.steps[-1].decent = decent
            return decent

        energy_change = self.history.steps[-1].E - self.history.steps[-2].E
        projected_change = self.history.steps[-2].projectedDE

        opt_step_report = "\n\tCurrent energy: %20.10lf\n" % self.history.steps[-1].E
        opt_step_report += "\tEnergy change for the previous step:\n"
        opt_step_report += "\t\tActual       : %20.10lf\n" % energy_change
        opt_step_report += "\t\tProjected    : %20.10lf\n" % projected_change

        logger.info("\tCurrent Step Report \n %s" % opt_step_report)

        # names reflect those used by Besalu and Bofill
        lower_b = 0
        upper_b = 2
        de = 0.40
        di = 0.60

        # The de, di, upper_b and lower_b are recommended by B&B.
        # Require that LB < rle < rli < rui < rue < UB
        # original # current
        r_lower_e = lower_b + de  # 0.75       0.40
        r_lower_i = lower_b + di  # 0.80       0.60
        r_upper_e = upper_b - de  # 1.25       1.60
        r_upper_i = upper_b - di  # 1.20       1.40

        energy_ratio = energy_change / projected_change
        logger.info("\tEnergy ratio = %10.5lf" % energy_ratio)

        dq = self.history.steps[-1].Dq

        # is step within trust region (1e-5 is the error used in alpha convergence)
        in_trust_region = np.sqrt(dq @ dq) - self.params.intrafrag_trust < 1e-5

        if self.supports_trust_region() and not self.params.linesearch:
            if energy_ratio < r_lower_e or energy_ratio > r_upper_e:
                logger.debug("Decreasing trust radius")
                self.decrease_trust_radius()
                decent = False
            if r_lower_i < energy_ratio < r_upper_i and in_trust_region:
                logger.debug("Increasing trust radius")
                self.increase_trust_radius()
                decent = True

        self.history.steps[-2].decent = decent
        return decent


def step_matches_forces(dq: np.ndarray, fq: np.ndarray):
    """Check for eigenvectors from RFO matrix which don't qualitatively match
    the gradient. If the forces are zero along a given coordinate,
    the eigenvector which will be taken as a step should have a component of 0 along that coordinate
    """

    temp = dq.copy()
    # Silence close to zero values in the eigenvectors
    indices = np.argwhere(np.abs(temp) < 1e-10)
    temp[indices] = 0.0

    # Check for eigenvector values (dq) that are are non-zero where
    # the correspinding forces for that coordinate are zero
    indices = np.argwhere(np.abs(fq) < 1e-12)
    if not np.allclose(dq[indices], 0.0, rtol=0.0, atol=1e-4):
        logger.debug(
            "The step has a non-zero component along a coordinate with near-zero force %s",
            print_array_string(dq, form=":10.12f"),
        )
        return False

    return True

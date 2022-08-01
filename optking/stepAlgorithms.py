""" Defines classes for minimization and transition state optimizations

See Also
---------
opt_helper.OptHelper for easy setup and control over optimization procedures.

* User interfaces
    * optimization_factory()
    * OptimizationManager
* optimiziation Classes
    * NewtonRaphson
    * SteepestDescent
        * overlap
        * barzilai_borwein
    * RestricedStepRFO
    * ParitionedRFO
* Linesearch
    * ThreePointEnergy
* Abstract Classes
    * OptimizationInterface
    * OptimizationAlgorithm
    * RFO
    * QuasiNewtonOptimization

The optimization classes above may be created using the factory pattern through optimization_factory(). If
linesearching or more advanced management of the optimization process is desired an OptimizationManager
should be created.
(More features are coming to the OptimizationManager)

"""

import logging
from math import fabs, sqrt
from abc import ABC, abstractmethod
from typing import Union

import numpy as np

from . import optparams as op
from . import convcheck
from .addIntcos import linear_bend_check
from .displace import displace_molsys
from .exceptions import AlgError, OptError
from .linearAlgebra import asymm_mat_eig, symm_mat_eig
from .misc import is_dq_symmetric
from .printTools import print_array_string, print_mat_string
from . import log_name

logger = logging.getLogger(f"{log_name}{__name__}")


class OptimizationInterface(ABC):
    """Declares that ALL OptKing optimization methods/algorithms will have
    a self.take_step() method. All methods must be able to determine what the next step
    to take should be given a history. See take_step() docstring for details."""

    def __init__(self, molsys, history, params):
        """set history and molsys. Create a default params object if required.
        Individual params will be set as instance attributes by the child classes as needed

        Parameters
        ----------
        molsys: molsys.Molsys
        history: history.History
        params: op.OptParams"""

        self.molsys = molsys
        self.history = history

        if not params:
            params = op.OptParams({})
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

    All child classes implement a step() method using the forces and Hessian to compute
    a step direction and possibly a step_length. trust_radius_on = False allows a child class
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

    def expected_energy(self, step, grad, hess):
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

        self.history.append(self.molsys.geom, energy, fq, self.molsys.gradient_to_cartesians(-1 * fq))

        if self.backstep_needed():
            dq = self.backstep()
        else:
            dq = self.step(fq, H)

        if self.trust_radius_on:
            self.apply_intrafrag_step_scaling(dq)

        self.molsys.interfrag_dq_discontinuity_correction(dq)
        achieved_dq, return_str = displace_molsys(self.molsys,
                                                  dq,
                                                  fq,
                                                  ensure_convergence=self.params.ensure_bt_convergence,
                                                  return_str=return_str)
        dq_norm, unit_dq, projected_fq, projected_hess = self.step_metrics(achieved_dq, fq, H)
        delta_energy = self.expected_energy(dq_norm, projected_fq, projected_hess)
        logger.debug("\tProjected energy change: %10.10lf\n" % delta_energy)
        self.update_history(delta_energy, achieved_dq, unit_dq, projected_fq, projected_hess)

        dq_norm = np.linalg.norm(achieved_dq)
        logger.info("\tNorm of achieved step-size %15.10f" % dq_norm)

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

    def backstep(self):
        """takes a partial step backwards. fq and H should correspond to the previous not current point

        Notes
        -----
        Take partial backward step.  Update current step in history.
        Divide the last step size by 1/2 and displace from old geometry.
        History contains:
            consecutiveBacksteps : increase by 1
        Step contains:
            forces, geom, E, followedUnitVector, oneDgradient, oneDhessian, Dq, and projectedDE
        update:
            Dq - cut in half
            projectedDE - recompute
        leave remaining

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

    def converged(self, dq, fq, step_number, str_mode=None):
        energies = [step.E for step in self.history.steps]
        conv_info = {'step_type': 'standard',
                     'energies': energies,
                     'dq': dq,
                     'fq': fq,
                     'iternum': step_number}
        converged = convcheck.conv_check(conv_info, self.params.__dict__, str_mode=str_mode)
        if str_mode:
            return converged
        logger.info("\tConvergence check returned %s" % converged)

        return converged

    def backstep_needed(self):
        """Simple logic for whether a backstep is advisable (or too many have been taken).

        Returns
        -------
        bool : True if a backstep should be taken
        """

        previous_is_decent = self.assess_previous_step()

        if previous_is_decent:
            self.history.consecutive_backsteps = 0
            return False

        if len(self.history.steps) < 5:
            # ignore early issues.
            logger.info("\tNear start of optimization, so ignoring bad step.\n")
            return False

        if self.history.consecutive_backsteps < self.params.consecutive_backsteps_allowed:
            self.history.consecutive_backsteps += 1
            logger.info("\tThis is consecutive backstep %d.\n", self.history.consecutive_backsteps)
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

        if self.params.opt_type == "MIN" and not self.params.linesearch:
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
        """ Increase trust radius by factor of 3 """
        maximum = self.params.intrafrag_trust_max
        if self.params.intrafrag_trust != maximum:
            new_val = self.params.intrafrag_trust * 3
            new_val = maximum if new_val > self.params.intrafrag_trust_max else new_val
            logger.info("\tEnergy ratio indicates good step: Trust radius increased to %6.3e.\n", new_val)
            self.params.intrafrag_trust = new_val

    def decrease_trust_radius(self):
        """Scale trust radius by 0.25 """
        minimum = self.params.intrafrag_trust_min
        if self.params.intrafrag_trust != minimum:
            new_val = self.params.intrafrag_trust / 4
            new_val = minimum if new_val < minimum else new_val
            logger.warning("\tEnergy ratio indicates iffy step: Trust radius decreased to %6.3e.\n", new_val)
            self.params.intrafrag_trust = new_val

    def update_history(self, delta_e, achieved_dq, unit_dq, projected_f, projected_hess):
        """Basic history update method. This should be expanded here and in child classes in
        future"""

        self.history.append_record(delta_e, achieved_dq, unit_dq, projected_f, projected_hess)

        linear_list = linear_bend_check(self.molsys, achieved_dq)
        if linear_list:
            raise AlgError("New linear angles", newLinearBends=linear_list)

    # def converged(self, step_number, dq, fq):
    #     energies = (self.history.steps[-1].E, self.history.steps[-2].E)  # grab last two energies
    #     converged = convcheck.conv_check(step_number, self.molsys, dq, energies)
    #     logger.info("\tConvergence check returned %s" % converged)
    #     return converged


class QuasiNewtonOptimization(OptimizationAlgorithm, ABC):
    def requires(self):
        return "energy", "gradient", "hessian"

    def expected_energy(self, step, grad, hess):
        """Quadratic energy model"""
        return step * grad + 0.5 * step * step * hess

    def take_step(self, fq=None, H=None, energy=None, return_str=False, **kwargs):
        if len(H) == 0:
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

    def step_size_scalar(self, fq):
        """Perform two point calculation of steepest descent step_size"""

        methods = {"OVERLAP": self._force_overlap, "BARZILAI_BORWEIN": self._barzilai_borwein}

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

    def expected_energy(self, step, grad, hess):
        """Quadratic energy model"""
        return step * grad + 0.5 * step * step * hess


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

    def expected_energy(self, step, grad, hess):
        """RFO model - 2x2 Pade Approximation"""
        return (step * grad + 0.5 * step * step * hess) / (1 + step * step)

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
        i_norm = np.where(np.abs(eigenvectors[:, -1]) > 1.0e-10, eigenvectors[:, -1], 1)  # last element or 1
        i_norm = i_norm.reshape(-1, 1)
        tmp = eigenvectors / i_norm
        max_values = (np.amax(np.abs(tmp), axis=1)).reshape(-1, 1)
        eigenvectors = np.where(max_values < self.params.rfo_normalization_max, tmp, eigenvectors)
        return eigenvectors


class RestrictedStepRFO(RFO):
    """Rational Function approximation (or 2x2 Pade approximation) for step.
    Uses Scaling Parameter to adjust step size analagous to the NR hessian shift parameter."""

    def __init__(self, molsys, history, params):
        super().__init__(molsys, history, params)
        # self.rsrfo_alpha_max = params.rsrfo_alpha_max
        # self.accept_symmetry_breaking = params.accept_symmetry_breaking

    def step(self, fq, H, *args, **kawrgs):
        """The step is an eigenvector of the gradient augmented Hessian. Looks for the
        lowest eigenvalue / eigenvector that is normalizable. If rfo_follow_root the previous root/step
        is considered first"""

        logger.debug("\tTaking RFO optimization step.")

        # Build the original, unscaled RFO matrix.
        RFOmat = RFO.build_rfo_matrix(0, len(H), fq, H)  # use entire hessian for RFO matrix
        logger.debug("\tOriginal, unscaled RFO matrix:\n\n" + print_mat_string(RFOmat))

        if self.params.simple_step_scaling:
            e_vectors, e_values = self._intermediate_normalize(RFOmat)
            rfo_root = self._select_rfo_root(
                self.history.steps[-2].followedUnitVector, e_vectors, e_values, alpha_iter=0
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
        max_rfo_iter = 25  # max. # of iterations to try to converge RS-RFO
        Hevals, Hevects = symm_mat_eig(H)  # Need for computing alpha at end of loop
        follow_root = self.params.rfo_follow_root

        last_evect = np.zeros(len(fq))
        if self.params.rfo_follow_root and len(self.history.steps) > 1:
            last_evect[:] = self.history.steps[-2].followedUnitVector  # RFO vector from previous geometry step
        rfo_step_report = ""

        while not converged and alpha_iter < max_rfo_iter:
            alpha_iter += 1

            # If we exhaust iterations without convergence, then bail on the
            #  restricted-step algorithm.  Set alpha=1 and apply crude scaling instead.
            if alpha_iter == max_rfo_iter:
                logger.warning("\tFailed to converge alpha. Doing simple step-scaling instead.")
                alpha = 1.0

            SRFOevals, SRFOevects = self._scale_and_normalize(RFOmat, alpha)

            # Determine best (lowest eigenvalue), acceptable root and take as step
            rfo_root = self._select_rfo_root(last_evect, SRFOevects, SRFOevals, alpha_iter)
            dq = SRFOevects[rfo_root][:-1]  # omit last column
            # last_evect = dq / np.linalg.norm(dq)

            dqtdq = np.dot(dq, dq)
            # If alpha explodes, give up on iterative scheme
            if fabs(alpha) > self.params.rsrfo_alpha_max:
                converged = False
                alpha_iter = max_rfo_iter - 1
            elif sqrt(dqtdq) < (self.params.intrafrag_trust + 1e-5):
                converged = True

            alpha, print_out = self._update_alpha(alpha, dqtdq, alpha_iter, dq, fq, Hevects, Hevals)
            rfo_step_report += print_out

        # end alpha RS-RFO iterations
        logger.debug(rfo_step_report)
        self.params.rfo_follow_root = follow_root
        return converged, dq

    def _update_alpha(self, alpha, dqtdq, alpha_iter, dq, fq, Hevects, Hevals):

        rfo_step_report = ""

        if alpha_iter == 0 and not self.params.simple_step_scaling:
            logger.debug("\tDetermining step-restricting scale parameter for RS-RFO.")

        if alpha_iter == 0:
            rfo_step_report += (
                "\n\n\t Iter      |step|        alpha        rfo_root"
                + "\n\t------------------------------------------------"
                + "\n\t%5d%12.5lf%14.5lf%12d\n" % (alpha_iter + 1, sqrt(dqtdq), alpha, self.rfo_root + 1)
            )

        elif alpha_iter > 0 and not op.Params.simple_step_scaling:
            rfo_step_report += "\t%5d%12.5lf%14.5lf%12d\n" % (alpha_iter + 1, sqrt(dqtdq), alpha, self.rfo_root + 1,)

        Lambda = -1 * fq @ dq

        # Calculate derivative of step size wrt alpha.
        tval = np.einsum("ij, j -> i", Hevects, fq) ** 2 / (Hevals - Lambda * alpha) ** 3
        tval = np.sum(tval)
        analyticDerivative = 2 * Lambda / (1 + alpha * dqtdq) * tval

        if self.print_lvl >= 2:
            logger.debug("\tLambda calculated by (dq^t).(-f) = %15.10lf\n" % Lambda)
            rfo_step_report += "\t  Analytic derivative d(norm)/d(alpha) = %15.10lf\n" % analyticDerivative

        # Calculate new scaling alpha value.
        # Equation 20, Besalu and Bofill, Theor. Chem. Acc., 1998, 100:265-274
        alpha += 2 * (self.params.intrafrag_trust * sqrt(dqtdq) - dqtdq) / analyticDerivative

        return alpha, rfo_step_report

    def _scale_and_normalize(self, RFOmat, alpha=1):
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
        SRFOmat[-1, :-1] = RFOmat[-1, :-1] / alpha**0.5
        SRFOmat[:-1, -1] = RFOmat[:-1, -1] / alpha**0.5
        SRFOevals, SRFOevects = symm_mat_eig(SRFOmat)

        SRFOevects = self._intermediate_normalize(SRFOevects)

        # transform step back.
        scale_mat = np.diag(np.repeat(1 / alpha ** 0.5, dim1))
        scale_mat[-1, -1] = 1
        # SRFOevects = np.einsum("ij, jk -> ik", scale_mat, SRFOevects)
        SRFOevects = np.transpose(scale_mat @ np.transpose(SRFOevects))

        if self.print_lvl >= 4:
            logger.debug("\tScaled RFO matrix.\n\n" + print_mat_string(SRFOmat))
            logger.debug("\tEigenvectors of scaled RFO matrix.\n\n" + print_mat_string(SRFOevects))
            logger.debug("\tEigenvalues of scaled RFO matrix.\n\n\t" + print_array_string(SRFOevals))
            logger.debug(
                "\tFirst eigenvector (unnormalized) of scaled RFO matrix.\n\n\t" + print_array_string(SRFOevects[0])
            )
            logger.debug("\tAll intermediate normalized eigenvectors (rows).\n\n" + print_mat_string(SRFOevects))

        return SRFOevals, SRFOevects

    def _select_rfo_root(self, last_evect, SRFOevects, SRFOevals, alpha_iter=0):
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

                    acceptable = self._check_rfo_eigenvector(SRFOevects[i], i)
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

    def _check_rfo_eigenvector(self, vector, index):
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
        symmetric = True if not self.params.accept_symmetry_breaking else is_dq_symmetric(self.molsys, vector[:-1])

        if not symmetric:
            return reject_root("it breaks the molecular point group")

        if vector[-1] < 1e-10:
            return reject_root("Normalization gives large value")

        if np.amax(np.abs(vector)) > self.params.rfo_normalization_max:
            return reject_root("Normalization gives large value")

        return True


class PartitionedRFO(RFO):
    """Partitions the gradient augmented Hessian into eigenvectors to maximize along (direction of the TS)
    and minimize along all other directions. Rational Function or (2x2 Pade approximation)"""

    def step(self, fq, H, *args, **kwargs):

        hdim = len(fq)  # size of Hessian

        # Diagonalize H (technically only have to semi-diagonalize)
        h_eig_values, h_eig_vectors = symm_mat_eig(H)
        hess_diag = np.diag(h_eig_values)

        if self.print_lvl > 2:
            logger.info("\tEigenvalues of Hessian\n\n\t%s", print_array_string(h_eig_values))
            logger.info("\tEigenvectors of Hessian (rows)\n%s", print_mat_string(h_eig_vectors))
            logger.debug("\tFor P-RFO, assuming rfo_root=1, maximizing along lowest eigenvalue of Hessian.")
            logger.debug("\tLarger values of rfo_root are not yet supported.")

        rfo_root = 0
        # self._select_rfo_root() TODO

        # number of degrees along which to maximize; assume 1 for now
        mu = 1
        fq_prime = np.dot(h_eig_vectors, fq)  # gradient transformation
        logger.info("\tInternal forces in au, in Hevect basis:\n\n\t" + print_array_string(fq_prime))

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

        logger.info("\tRFO step in Hessian Eigenvector Basis\n\n\t" + print_array_string(prfo_evect))
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
        raise NotImplementedError("Partitioned RFO only follows the lowest eigenvalue / vector currently")

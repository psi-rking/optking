import logging
from math import acos, sqrt, tan

import copy
import numpy as np

from . import IRCdata, convcheck
from .displace import displace_molsys
from .exceptions import AlgError
from .linearAlgebra import symm_mat_eig, symm_mat_inv, symm_mat_root, lowest_eigenvector_symm_mat
from .printTools import print_array_string, print_mat_string
from .stepAlgorithms import OptimizationInterface
from . import log_name
from . import op

logger = logging.getLogger(f"{log_name}{__name__}")


class IntrinsicReactionCoordinate(OptimizationInterface):
    def __init__(self, molsys, history, params):
        super().__init__(molsys, history, params)
        self.orig_molsys = copy.deepcopy(molsys)

        self.params = params
        # grab irc specific information
        # self.irc_direction = params.irc_direction
        # self.irc_step_size = params.irc_step_size
        # self.irc_points = params.irc_points
        self.irc_step_number = 0
        self.sub_step_number = -1
        self.total_steps_taken = 0
        self.irc_history = IRCdata.IRCHistory()
        self.irc_history.set_atom_symbols(self.molsys.atom_symbols)
        self.irc_history.set_step_size_and_direction(self.params.irc_step_size, self.params.irc_direction)

    def to_dict(self):

        return {
            "irc_step_number": self.irc_step_number,
            "sub_step_number": self.sub_step_number,
            "total_steps_taken": self.total_steps_taken,
            "irc_history": self.irc_history.to_dict(),
        }

    @classmethod
    def from_dict(cls, d, molsys, history, params):

        irc = cls(molsys, history, params)
        irc.irc_step_number = d["irc_step_number"]
        irc.sub_step_number = d["sub_step_number"]
        irc.total_steps_taken = d["total_steps_taken"]
        irc.irc_history = IRCdata.IRCHistory.from_dict(d.get("irc_history"))
        return irc

    def requires(self):

        return "energy", "gradient", "hessian"

    def take_step(self, fq=None, H=None, energy=None, return_str=False, **kwargs):

        if self.sub_step_number == -1:
            if self.irc_step_number == 0:
                logger.info("\tBeginning IRC from the transition state.")
                logger.info("\tStepping along lowest Hessian eigenvector.")
                logger.debug(print_mat_string(H, title="Transformed Hessian in internals."))

                # Add the transition state as the first IRC point
                q_0 = self.molsys.q_array()
                x_0 = self.molsys.geom
                f_x = self.molsys.gradient_to_cartesians(fq)

                self.irc_history.add_irc_point(0, q_0, x_0, fq, f_x, energy)
                self.irc_step_number += 1

                # Lowest eigenvector of mass-weighted Hessian.
                G = self.molsys.Gmat(massWeight=True)
                G_root = symm_mat_root(G, threshold=self.params.linear_algebra_tol)
                H_q_m = G_root @ H @ G_root
                logger.info(print_mat_string(H_q_m, title="Mass Weighted Hessian in Internals"))
                vM = lowest_eigenvector_symm_mat(H_q_m)
                logger.info(print_array_string(vM, title="Lowest eigenvector of Mass Weighted, Internal Hessian"))

                # Un mass-weight vector.
                G_root_inv = symm_mat_inv(
                                        G_root,
                                        redundant=True,
                                        small_val_limit=self.params.linear_algebra_tol
                                    )
                v = np.dot(G_root_inv, vM)

                if self.params.irc_direction == "BACKWARD":
                    v *= -1
            else:
                logger.info("\tBeginning search for next IRC point.\n")
                logger.info("\tStepping along gradient.\n")
                v = self.irc_history.f_q()
                self.irc_step_number += 1

            dq, return_str = self.compute_pivot_and_guess_points(v, fq, return_str=True)
        else:

            self.history.append(self.molsys.geom, energy, fq, self.molsys.gradient_to_cartesians(-1 * fq))
            dq = self.dq_irc(fq, H)
            dq, dx, return_str = displace_molsys(
                self.molsys,
                dq,
                fq,
                **self.params.__dict__,
                return_str=True,
                ensure_convergence=True
            )
            logger.info("IRC Constrained step calculation finished.")

            # Complete history entry of step.
            # Compute gradient and hessian in step direction
            dq_norm, dq_unit, grad, hess = self.step_metrics(dq, fq, H)
            DE = irc_de_projected(dq_norm, grad, hess)
            self.history.append_record(DE, dq, dq_unit, grad, hess)

        self.sub_step_number += 1
        self.total_steps_taken += 1

        if return_str:
            return dq, return_str
        return dq

    def converged(self, dq, fq, step_number, str_mode="", **kwargs):
        # This method no longer clears out the history after converging. This reproduces old optking
        # behavior. It is also found that clearing the history negatively impacts hessian updating.

        energies = [step.E for step in self.history.steps]
        fq_new = self.irc_history._project_forces(fq, self.molsys, self.params.linear_algebra_tol)

        if self.sub_step_number < 2 and self.irc_step_number == 1:
            logger.debug("Too few steps. continue optimization")
            return False

        # On second irc_point. Have not begun sub optimization
        if self.irc_history.test_for_irc_minimum(fq, energies[-1], self.params.irc_convergence):
            logger.info("A minimum has been reached on the IRC.  Stopping here.\n")
            return True

            if self.irc_history.test_for_irc_minimum(fq_new, energies[-1]):
                logger.info("A minimum has been reached on the IRC.  Stopping here.\n")
                return True

            if self.params.irc_mode.upper() == "CONFIRM":
                if self.irc_history.test_for_dissociation(self.molsys, self.orig_molsys):
                    logger.info("A new fragment has been detected on the along the reaction path.\n"
                                "IRC is running in 'confirm' mode. Stopping here.\n")
                    return True

        # Need to communicate that we want to print an IRC report
        # Need not total_steps_taken but the irc_step_number and sub_step_number
        conv_data = {
            "step_type": "irc",
            "iternum": self.irc_step_number,
            "sub_step_num": self.sub_step_number,
            "energies": energies,
            "dq": dq,
            "fq": fq_new,
        }

        substep_convergence = convcheck.conv_check(conv_data, self.params, self.requires(), str_mode=str_mode)
        if not str_mode:
            logger.info("\tConvergence check returned %s for constrained optimization." % substep_convergence)

        if substep_convergence is True:
            self.add_converged_point(fq, self.history.steps[-1].E)
            self.sub_step_number = -1
            logger.info("\tStarting search for next IRC point.")
            logger.info("\tClearing old constrained optimization history.")

            if self.irc_step_number >= self.params.irc_points:
                logger.info(f"\tThe requested {self.params.irc_points} IRC points have been obtained.")
                return True

        if str_mode:
            return substep_convergence
        return False  # return True means we're finished

    def compute_pivot_and_guess_points(self, v, fq, return_str=False):
        """Takes a half step along v to the 'pivot point', then
        an additional half step as first guess in constrained opt.

        Parameters
        ----------
        v : ndarray
            initial vector to step along. Hessian eigenvector for first step. Gradient at subsequent steps

        """

        # Compute and save pivot point
        G = self.molsys.Gmat(massWeight=True)
        N = step_n_factor(G, v)
        dq_pivot = 0.5 * N * self.params.irc_step_size * np.dot(G, v)
        logger.debug("\n Dq to Pivot Point:" + print_array_string(dq_pivot))

        # x_pivot = o_molsys.geom # starting geom but becomes pivot point on next line
        # displace(o_molsys.intcos, x_pivot, dq_pivot, ensure_convergence=True)

        # revisit
        dq1, dx1, return_str1 = displace_molsys(
            self.molsys,
            dq_pivot,
            fq,
            **self.params.__dict__,
            ensure_convergence=True,
            return_str=True
        )
        x_pivot = self.molsys.geom
        q_pivot = self.molsys.q_array()
        self.irc_history.add_pivot_point(q_pivot, x_pivot)

        # Step again to get initial guess for next step.  Leave geometry in o_molsys.
        logger.info("Computing Dq to First Guess Point")
        logger.debug(print_array_string(dq_pivot))
        x_guess = x_pivot.copy()
        # displace(o_molsys.intcos, x_guess, dq_pivot, ensure_convergence=True)
        dq2, dx2, return_str2 = displace_molsys(
            self.molsys,
            dq_pivot,
            fq,
            ensure_convergence=True,
            return_str=return_str,
            print_lvl=self.params.print_lvl
        )
        # self.molsys.geom = x_guess
        if return_str:
            return dq1 + dq2, return_str1 + return_str2
        return dq1 + dq2

    def dq_irc(self, f_q, H_q):
        """Before dq_irc is called, the geometry must be updated to the guess point
        Returns Dq from qk+1 to gprime.
        """

        logger.debug("Starting IRC constrained optimization\n")
        threshold = self.params.linear_algebra_tol  # shortcut

        G_prime = self.molsys.Gmat(massWeight=True)
        logger.debug("Mass-weighted Gmatrix at hypersphere point: \n" + print_mat_string(G_prime))
        G_prime_root = symm_mat_root(G_prime, threshold=threshold)
        G_prime_inv = symm_mat_inv(G_prime, redundant=True, small_val_limit=threshold)
        G_prime_root_inv = symm_mat_root(G_prime_inv, threshold=threshold)

        logger.debug("G prime root matrix: \n" + print_mat_string(G_prime_root))

        deltaQM = 0
        g_M = np.dot(G_prime_root, -f_q)
        logger.debug("g_M: \n" + print_array_string(g_M))

        H_M = np.dot(np.dot(G_prime_root, H_q), G_prime_root.T)
        logger.debug("H_M: \n" + print_mat_string(H_M))

        # Compute p_prime, difference from pivot point
        orig_geom = self.molsys.geom
        self.molsys.geom = self.irc_history.x_pivot()
        q_pivot = self.molsys.q_array()
        self.molsys.geom = orig_geom
        p_prime = self.molsys.q_array() - q_pivot

        # p_prime = intcosMisc.q_values(o_molsys.intcos, o_molsys.geom) -  \
        #          intcosMisc.q_values(o_molsys.intcos, IRCdata.history.x_pivot())
        p_M = np.dot(G_prime_root_inv, p_prime)
        logger.debug("p_M: \n" + print_array_string(p_M))

        HMEigValues, HMEigVects = symm_mat_eig(H_M)
        logger.debug("HMEigValues: \n" + print_array_string(HMEigValues))
        logger.debug("HMEigVects: \n" + print_mat_string(HMEigVects))

        # Variables for solving lagrangian function
        lb_lagrangian = -100
        up_lagrangian = 100
        lb_lambda = 0
        if HMEigValues[0] < 0:  # Make lower than the lowest eval
            Lambda = 1.1 * HMEigValues[0]
        else:
            Lambda = 0.9 * HMEigValues[0]
        up_lambda = Lambda

        # Solve Eqn. 26 in Gonzalez & Schlegel (1990) for lambda.
        # Sum_j { [(b_j p_bar_j - g_bar_j)/(b_j - lambda)]^2} - (s/2)^2 = 0.
        # For each j (dimension of H_M):
        #  b is an eigenvalues of H_M
        #  p_bar is projection p_M onto an eigenvector of H_M
        #  g_bar is projection g_M onto an eigenvector of H_M

        lagrangian = self.calc_lagrangian(Lambda, HMEigValues, HMEigVects, g_M, p_M)
        prev_lagrangian = lagrangian

        logger.debug("Starting coarse-grain multiplier search.")
        logger.debug("lambda        Lagrangian value:\n")
        # print("lambda        Lagrangian value:")

        lagIter = 0
        while lagIter < 1000:
            lagrangian = self.calc_lagrangian(Lambda, HMEigValues, HMEigVects, g_M, p_M)
            logger.debug("%15.10e  %8.3e" % (Lambda, lagrangian))
            # print("%15.10e  %8.3e" % (Lambda, lagrangian) )

            if lagrangian < 0 and abs(lagrangian) < abs(lb_lagrangian):
                lb_lagrangian = lagrangian
                lb_lambda = Lambda
            elif lagrangian > 0 and abs(lagrangian) < abs(up_lagrangian):
                up_lagrangian = lagrangian
                up_lambda = Lambda

            if prev_lagrangian * lagrangian < 0:
                break
            prev_lagrangian = lagrangian

            lagIter += 1
            Lambda -= 0.001

        logger.debug("Coarse graining results:")
        logger.debug("Lambda between %10.5f and %10.5f" % (lb_lambda, up_lambda))
        logger.debug("for Lagrangian %10.5f to  %10.5f" % (lb_lagrangian, up_lagrangian))

        # Calulate lambda using Householder method
        # prev_lambda = -999
        prev_lambda = Lambda
        lagIter = 0
        Lambda = (lb_lambda + up_lambda) / 2  # start in middle of coarse range

        logger.debug("lambda        Lagrangian:")
        while abs(Lambda - prev_lambda) > 10**-15:
            prev_lagrangian = lagrangian
            L_derivs = self.calc_lagrangian_derivs(Lambda, HMEigValues, HMEigVects, g_M, p_M)
            lagrangian = L_derivs[0]
            logger.debug("%15.5e%15.5e" % (Lambda, lagrangian))

            h_f = -1 * L_derivs[0] / L_derivs[1]

            # Keep track of lowest and highest results thus far
            if lagrangian < 0 and (abs(lagrangian) < abs(lb_lagrangian)):
                lb_lagrangian = lagrangian
                lb_lambda = Lambda
            elif lagrangian > 0 and abs(lagrangian) < abs(up_lagrangian):
                up_lagrangian = lagrangian
                up_lambda = Lambda

            # Bisect if lagrangian has changed signs.
            if lagrangian * prev_lagrangian < 0:
                current_lambda = Lambda
                Lambda = (prev_lambda + Lambda) / 2
                prev_lambda = current_lambda
            else:
                prev_lambda = Lambda
                Lambda += (
                    h_f
                    * (24 * L_derivs[1] + 24 * L_derivs[2] * h_f + 4 * L_derivs[3] * h_f**2)
                    / (
                        24 * L_derivs[1]
                        + 36 * h_f * L_derivs[2]
                        + 6 * L_derivs[2] ** 2 * h_f**2 / L_derivs[1]
                        + 8 * L_derivs[3] * h_f**2
                        + L_derivs[4] * h_f**3
                    )
                )

            lagIter += 1
            if lagIter > 50:
                prev_lambda = Lambda
                Lambda = (lb_lambda + up_lambda) / 2  # Try a bisection after 50 attempts

            if lagIter > 200:
                err_msg = "Could not converge Lagrangian multiplier for constrained rxnpath search."
                logger.warning(err_msg)
                raise AlgError(err_msg)

        logger.info("Lambda converged at %15.5e" % Lambda)

        # Find dq_M from Eqn. 24 in Gonzalez & Schlegel (1990).
        # dq_M = (H_M - lambda I)^(-1) [lambda * p_M - g_M]
        LambdaI = np.identity(self.molsys.num_intcos)
        LambdaI = np.multiply(Lambda, LambdaI)
        deltaQM = symm_mat_inv(np.subtract(H_M, LambdaI), redundant=True, small_val_limit=threshold)
        deltaQM = np.dot(deltaQM, np.subtract(np.multiply(Lambda, p_M), g_M))
        logger.debug("dq_M to next geometry\n" + print_array_string(deltaQM))

        # Find dq = G^(1/2) dq_M and do displacements.
        dq = np.dot(G_prime_root, deltaQM)
        logger.info("dq to next geometry\n" + print_array_string(dq))

        return dq

        # TODO write geometry for multiple fragments
        # displace(o_molsys.intcos, o_molsys._fragments[0].geom, dq)

    def calc_line_dist_step(self):
        """mass-weighted distance from previous rxnpath point to new one"""
        G = self.molsys.Gmat(massWeight=True)
        G_root = symm_mat_root(G)
        G_inv = symm_mat_inv(G_root, redundant=True,small_val_limit=self.params.linear_algebra_tol)
        G_root_inv = symm_mat_root(G_inv)

        rxn_Dq = np.subtract(self.molsys.q_array(), self.irc_history.q())
        # mass weight (not done in old C++ code)
        rxn_Dq_M = np.dot(G_root_inv, rxn_Dq)
        return np.linalg.norm(rxn_Dq_M)

    def calc_arc_dist_step(self):
        """Let q0 be last rxnpath point and q1 be new rxnpath point. q* is the pivot
        point (1/2)s from each of these. Returns the length of circular arc connecting
        q0 and q1, whose center is equidistant from q0 and q1, and for which line segments
        from q* to q0 and from q* to q1 are perpendicular to segments from the center
        to q0 and q1."""
        qp = self.irc_history.q_pivot(-1)  # pivot point is stored in previous step
        q0 = self.irc_history.q(-1)
        q1 = self.molsys.q_array()

        p = np.subtract(q1, qp)  # Dq from pivot point to latest rxnpath pt.
        line = np.subtract(q1, q0)  # Dq from rxnpath pt. to rxnpath pt.

        # mass-weight
        p[:] = np.multiply(1.0 / np.linalg.norm(p), p)
        line[:] = np.multiply(1.0 / np.linalg.norm(line), line)

        alpha = acos(np.dot(p, line))
        arcDistStep = self.irc_history.step_size * alpha / tan(alpha)
        return arcDistStep

    def add_converged_point(self, fq, energy):
        q_irc_point = self.molsys.q_array()
        cart_forces = self.molsys.gradient_to_cartesians(fq)
        lineDistStep = self.calc_line_dist_step()
        arcDistStep = self.calc_arc_dist_step()

        self.irc_history.add_irc_point(
            self.irc_step_number,
            q_irc_point,
            self.molsys.geom,
            fq,
            cart_forces,
            energy,
            lineDistStep,
            arcDistStep,
        )
        self.irc_history.progress_report()

    def calc_lagrangian(self, Lambda, HMEigValues, HMEigVects, g_M, p_M):
        """Calculates and returns value of Lagrangian function given multiplier Lambda."""
        lagrangian = 0
        for i in range(len(HMEigValues)):
            numerator = HMEigValues[i] * np.dot(HMEigVects[i], p_M) - np.dot(HMEigVects[i], g_M)
            denom = HMEigValues[i] - Lambda
            lagrangian += (numerator / denom) ** 2

        lagrangian -= (0.5 * self.params.irc_step_size) ** 2
        return lagrangian

    def calc_lagrangian_derivs(self, Lambda, HMEigValues, HMEigVects, g_M, p_M):
        """Calculates and returns value of derivative of Lagrangian function given multiplier Lambda."""
        deriv = np.array([0.0, 0.0, 0.0, 0.0, 0.0], float)
        for i in range(len(HMEigValues)):
            numerator = HMEigValues[i] * np.dot(HMEigVects[i], p_M) - np.dot(HMEigVects[i], g_M)
            D = HMEigValues[i] - Lambda
            deriv[0] += (numerator / D) ** 2
            deriv[1] += 2 * (numerator / D) ** 2 / (D)
            deriv[2] += 6 * (numerator / D) ** 2 / (D * D)
            deriv[3] += 24 * (numerator / D) ** 2 / (D * D * D)
            deriv[4] += 120 * (numerator / D) ** 2 / (D * D * D * D)

        deriv[0] -= (0.5 * self.params.irc_step_size) ** 2

        return deriv


def step_n_factor(G, g):
    """Computes distance scaling factor for mass-weighted internals."""
    return 1.0 / sqrt(np.dot(g.T, np.dot(G, g)))


def irc_de_projected(step_size, grad, hess):
    """Compute anticipated energy change along one dimension"""
    return step_size * grad + 0.5 * step_size * step_size * hess

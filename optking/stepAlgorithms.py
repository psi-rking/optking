# Functions for step algorithms: Newton-Raphson, Rational Function Optimization,
# Steepest Descent.
# from .OptParams import Params # this will not cause changes in trust to persist
import logging
from math import fabs, sqrt

import numpy as np

from . import optimize
from . import optparams as op
from . import v3d
from .addIntcos import linear_bend_check
from .displace import displace_molsys
from .exceptions import AlgError, OptError
from .history import oHistory
from .linearAlgebra import abs_max, asymm_mat_eig, norm, symm_mat_eig, symm_mat_inv
from .misc import is_dq_symmetric
from .printTools import print_array_string, print_mat_string


def take_step(o_molsys, E, q_forces, H, stepType=None, computer=None, hist=None, params=None):
    """  This method computes the step, calls displaces the geometry and updates history with
     the results.

    Parameters
    ----------
    o_molsys : molsys.Molsys
        optking's molecular system
    E : double
        energy [aO]
    q_forces : ndarray
        forces in internal coordinates [aO]
    H : ndarray
        hessian in internal coordinates
    stepType : string, optional
        defaults to stepType in options
    computer : computeWrapper, optional
    hist : history.History object

    Returns
    -------
    np.ndarray
        dispalcement in internals

    Notes
    -----
    step_grad and step_hess are the gradient and hessian in the direction of the step.

    """
    if hist is None:
        hist = oHistory
    if params is None:
        params = op.Params
    logger = logging.getLogger(__name__)

    if len(H) == 0 or len(q_forces) == 0:
        logger.warning("Missing Hessian or Forces. Step is 0")
        return np.zeros(0)

    if not stepType:
        stepType = params.step_type

    if stepType == "NR":
        delta_E_projected, dq, unit_step, step_grad, step_hess = dq_nr(q_forces, H)
    elif stepType == "RFO":
        delta_E_projected, dq, unit_step, step_grad, step_hess = dq_rfo(o_molsys, q_forces, H)
    elif stepType == "SD":
        delta_E_projected, dq, unit_step, step_grad, step_hess = dq_sd(q_forces)
    elif stepType == "BACKSTEP":
        return dq_backstep(o_molsys)  # Do an early quit Back step takes care of history and displacing
    elif stepType == "P_RFO":
        delta_E_projected, dq, unit_step, step_grad, step_hess = dq_p_rfo(q_forces, H)
    elif stepType == "LINESEARCH":
        # achieved_dq already back
        delta_E_projected, achieved_dq, unit_step, step_grad, step_hess = dq_linesearch(
            o_molsys, E, q_forces, H, computer
        )
    else:
        raise OptError("Dq: step type not yet implemented")

    if stepType != "LINESEARCH":
        # linesearch performs multiple displacements in order to calculate energies
        o_molsys.interfrag_dq_discontinuity_correction(dq)
        achieved_dq = displace_molsys(o_molsys, dq, q_forces)

    dq_norm = np.linalg.norm(achieved_dq)
    logger.info("\tNorm of achieved step-size %15.10f" % dq_norm)

    hist.append_record(delta_E_projected, achieved_dq, unit_step, step_grad, step_hess)

    linearList = linear_bend_check(o_molsys, achieved_dq)
    if linearList:
        raise AlgError("New linear angles", newLinearBends=linearList)

    # Before quitting, make sure step is reasonable.  It should only be
    # screwball if we are using the "First Guess" after the back-transformation failed.
    dq_norm = np.linalg.norm(achieved_dq[0 : o_molsys.num_intrafrag_intcos])
    if dq_norm > 5 * params.intrafrag_trust:
        raise AlgError("opt.py: Step is far too large.")

    return achieved_dq


# TODO this method was described as crude do we need to revisit?
def apply_intrafrag_step_scaling(dq):
    """ Apply maximum step limit by scaling."""
    logger = logging.getLogger(__name__)
    trust = op.Params.intrafrag_trust
    if sqrt(np.dot(dq, dq)) > trust:
        scale = trust / sqrt(np.dot(dq, dq))
        logger.info("\tStep length exceeds trust radius of %10.5f." % trust)
        logger.info("\tScaling displacements by %10.5f" % scale)
        dq *= scale
    return


def de_projected(model, step, grad, hess):
    """ Compute anticpated energy change along one dimension """
    if model == "NR":
        return step * grad + 0.5 * step * step * hess
    elif model == "RFO":
        return (step * grad + 0.5 * step * step * hess) / (1 + step * step)
    else:
        raise OptError("de_projected does not recognize model.")


def dq_nr(fq, H):
    """ Takes a step according to Newton Raphson algorithm

    Parameters
    ----------
    o_molsys : molsys.Molsys
        optking molecular system
    E : double
        energy
    fq : ndarray
        forces in internal coordiantes
    H : ndarray
        hessian in internal coordinates

    Notes
    -----
    Presently, the attempted dq is stored in history not the
    actual dq from the backtransformation

    """

    logger = logging.getLogger(__name__)
    logger.info("\tTaking NR optimization step.")

    # Hinv fq = dq
    Hinv = symm_mat_inv(H, redundant=True)
    dq = np.dot(Hinv, fq)

    # applies maximum internal coordinate change
    apply_intrafrag_step_scaling(dq)

    # get norm |q| and unit vector in the step direction
    nr_dqnorm = sqrt(np.dot(dq, dq))
    nr_u = dq / nr_dqnorm
    logger.info("\tNorm of target step-size %15.10lf" % nr_dqnorm)

    # get gradient and hessian in step direction
    nr_g = -1 * np.dot(fq, nr_u)  # gradient, not force
    nr_h = np.dot(nr_u, np.dot(H, nr_u))

    if op.Params.print_lvl > 1:
        logger.info("\tNR target step|: %15.10f" % nr_dqnorm)
        logger.info("\tNR_gradient: %15.10f" % nr_g)
        logger.info("\tNR_hessian: %15.10f" % nr_h)
    DEprojected = de_projected("NR", nr_dqnorm, nr_g, nr_h)
    logger.info("\tProjected energy change by quadratic approximation: %10.10lf\n" % DEprojected)

    return DEprojected, dq, nr_u, nr_g, nr_h


# Take Rational Function Optimization step
def dq_rfo(oMolsys, fq, H):
    """ Takes a step using Rational Function Optimization

    Parameters
    ----------
    oMolsys : molsys.Molsys
        optking molecular system
    E : double
        energy
    fq : ndarray
        forces in internal coordinates
    H : ndarray
        hessian in internal coordinates

    """

    logger = logging.getLogger(__name__)
    logger.info("\tTaking RFO optimization step.")
    dim = len(fq)
    dq = np.zeros(dim)  # To be determined and returned.
    trust = op.Params.intrafrag_trust  # maximum step size
    max_projected_rfo_iter = 25  # max. # of iterations to try to converge RS-RFO
    rfo_follow_root = op.Params.rfo_follow_root  # whether to follow root
    rfo_root = op.Params.rfo_root  # if following, which root to follow

    # Determine the eigenvectors/eigenvalues of H.
    Hevals, Hevects = symm_mat_eig(H)

    # Build the original, unscaled RFO matrix.
    RFOmat = np.zeros((dim + 1, dim + 1))
    for i in range(dim):
        for j in range(dim):
            RFOmat[i, j] = H[i, j]
        RFOmat[i, dim] = RFOmat[dim, i] = -fq[i]

    if op.Params.print_lvl >= 4:
        logger.debug("\tOriginal, unscaled RFO matrix:\n\n" + print_mat_string(RFOmat))

    # symm_rfo_step = False NOT USED
    SRFOmat = np.zeros((dim + 1, dim + 1))  # For scaled RFO matrix.
    converged = False
    # dqtdq = 10  # square of norm of step NOT USED
    alpha = 1.0  # scaling factor for RS-RFO, scaling matrix is sI

    last_iter_evect = np.zeros(dim)
    if rfo_follow_root and len(oHistory.steps) > 1:
        last_iter_evect[:] = oHistory.steps[-2].followedUnitVector  # RFO vector from previous geometry step

    # Iterative sequence to find alpha
    alphaIter = -1
    while not converged and alphaIter < max_projected_rfo_iter:
        alphaIter += 1

        # If we exhaust iterations without convergence, then bail on the
        #  restricted-step algorithm.  Set alpha=1 and apply crude scaling instead.
        if alphaIter == max_projected_rfo_iter:
            logger.warning("\tFailed to converge alpha. Doing simple step-scaling instead.")
            alpha = 1.0
        elif op.Params.simple_step_scaling:
            # Simple_step_scaling is on, not an iterative method.
            # Proceed through loop with alpha == 1, and then continue
            alphaIter = max_projected_rfo_iter

        # Scale the RFO matrix.
        for i in range(dim + 1):
            for j in range(dim):
                SRFOmat[j, i] = RFOmat[j, i] / alpha
            SRFOmat[dim, i] = RFOmat[dim, i]

        if op.Params.print_lvl >= 4:
            logger.debug("\tScaled RFO matrix.\n\n" + print_mat_string(SRFOmat))

        # Find the eigenvectors and eigenvalues of RFO matrix.
        SRFOevals, SRFOevects = asymm_mat_eig(SRFOmat)

        if op.Params.print_lvl >= 4:
            logger.debug("\tEigenvectors of scaled RFO matrix.\n\n" + print_mat_string(SRFOevects))

        if op.Params.print_lvl >= 4:
            logger.debug("\tEigenvalues of scaled RFO matrix.\n\n\t" + print_array_string(SRFOevals))
            logger.debug(
                "\tFirst eigenvector (unnormalized) of scaled RFO matrix.\n\n\t" + print_array_string(SRFOevects[0])
            )

        # Do intermediate normalization.  RFO paper says to scale eigenvector
        # to make the last element equal to 1. Bogus evect leads can be avoided
        # using root following.
        for i in range(dim + 1):
            # How big is dividing going to make the largest element?
            # Same check occurs below for acceptability.
            if fabs(SRFOevects[i][dim]) > 1.0e-10:
                tval = abs_max(SRFOevects[i] / SRFOevects[i][dim])
                if tval < op.Params.rfo_normalization_max:
                    for j in range(dim + 1):
                        SRFOevects[i, j] /= SRFOevects[i, dim]

        if op.Params.print_lvl >= 4:
            logger.debug("\tAll scaled RFO eigenvectors (rows).\n\n" + print_mat_string(SRFOevects))

        # Use input rfo_root
        # If root-following is turned off, then take the eigenvector with the
        # rfo_root'th lowest eigvenvalue. If its the first iteration, then do the same.
        # In subsequent steps, overlaps will be checked.
        if not rfo_follow_root or len(oHistory.steps) < 2:

            # Determine root only once at beginning ?
            if alphaIter == 0:
                logger.debug("\tChecking RFO solution %d." % (rfo_root + 1))

                for i in range(rfo_root, dim + 1):
                    # Check symmetry of root.
                    dq[:] = SRFOevects[i, 0:dim]
                    if not op.Params.accept_symmetry_breaking:
                        symm_rfo_step = is_dq_symmetric(oMolsys, dq)

                        if not symm_rfo_step:  # Root is assymmetric so reject it.
                            logger.warning(
                                "\tRejecting RFO root %d because it breaks \
                                           the molecular point group."
                                % (rfo_root + 1)
                            )
                            continue

                    # Check normalizability of root.
                    if fabs(SRFOevects[i][dim]) < 1.0e-10:  # don't even try to divide
                        logger.warning(
                            "\tRejecting RFO root %d because normalization \
                                       gives large value."
                            % (rfo_root + 1)
                        )
                        continue
                    tval = abs_max(SRFOevects[i] / SRFOevects[i][dim])
                    if tval > op.Params.rfo_normalization_max:  # matching test in code above
                        logger.warning(
                            "\tRejecting RFO root %d because normalization \
                                       gives large value."
                            % (rfo_root + 1)
                        )
                        continue
                    rfo_root = i  # This root is acceptable.
                    break
                else:
                    rfo_root = op.Params.rfo_root
                    # no good one found, use the default

                # Save initial root. 'Follow' during the RS-RFO iterations.
                rfo_follow_root = True

        else:  # Do root following.
            # Find maximum overlap. Dot only within H block.
            dots = np.array([v3d.dot(SRFOevects[i], last_iter_evect, dim) for i in range(dim)], float,)
            bestfit = np.argmax(dots)
            if bestfit != rfo_root:
                logger.info("\tRoot-following has changed rfo_root value to %d." % (bestfit + 1))
                rfo_root = bestfit

        if alphaIter == 0:
            logger.info("\tUsing RFO solution %d." % (rfo_root + 1))
        last_iter_evect[:] = SRFOevects[rfo_root][0:dim]  # omit last column on right

        # Print only the lowest eigenvalues/eigenvectors
        if op.Params.print_lvl >= 2:
            logger.info("\trfo_root is %d" % (rfo_root + 1))
            for i in range(dim + 1):
                if SRFOevals[i] < -1e-6 or i < rfo_root:
                    eigen_val_vec = "\n\tScaled RFO eigenvalue %d:\n\t%15.10lf (or 2*%-15.10lf)\n" % (
                        i + 1,
                        SRFOevals[i],
                        SRFOevals[i] / 2,
                    )
                    eigen_val_vec += "\n\teigenvector:\n\t"
                    eigen_val_vec += print_array_string(SRFOevects[i])
                    logger.info(eigen_val_vec)
        dq[:] = SRFOevects[rfo_root][0:dim]  # omit last column

        # Project out redundancies in steps.
        # Added this projection in 2014; but doesn't seem to help, as f,H are already projected.
        # project_dq(dq);
        # zero steps for frozen coordinates?

        dqtdq = np.dot(dq, dq)
        # If alpha explodes, give up on iterative scheme
        if fabs(alpha) > op.Params.rsrfo_alpha_max:
            converged = False
            alphaIter = max_projected_rfo_iter - 1
        elif sqrt(dqtdq) < (trust + 1e-5):
            converged = True

        if alphaIter == 0 and not op.Params.simple_step_scaling:
            logger.info("\tDetermining step-restricting scale parameter for RS-RFO.")

        if alphaIter == 0:
            logger.info("\n\tMaximum step size allowed %10.5lf" % trust)
            rfo_step_report = (
                "\n\n\t Iter      |step|        alpha        rfo_root"
                + "\n\t------------------------------------------------"
                + "\n\t %5d%12.5lf%14.5lf%12d\n" % (alphaIter + 1, sqrt(dqtdq), alpha, rfo_root + 1)
            )
            logger.info(rfo_step_report)

        elif alphaIter > 0 and not op.Params.simple_step_scaling:
            rfo_step_report = "\t%5d%12.5lf%14.5lf%12d\n" % (alphaIter + 1, sqrt(dqtdq), alpha, rfo_root + 1,)
            logger.info(rfo_step_report)

        # Find the analytical derivative, d(norm step squared) / d(alpha)
        # rfo_step_report += ("\t------------------------------------------------\n")

        Lambda = -1 * v3d.dot(fq, dq, dim)
        if op.Params.print_lvl >= 2:
            disp_forces = "\tDisplacement and Forces\n\n"
            disp_forces += "\tDq:" + print_array_string(dq, dim)
            disp_forces += "\tFq:" + print_array_string(fq, dim)
            logger.info(disp_forces)
            logger.info("\tLambda calculated by (dq^t).(-f) = %15.10lf\n" % Lambda)

        # Calculate derivative of step size wrt alpha.
        tval = 0
        for i in range(dim):
            tval += (pow(v3d.dot(Hevects[i], fq, dim), 2)) / (pow((Hevals[i] - Lambda * alpha), 3))

        analyticDerivative = 2 * Lambda / (1 + alpha * dqtdq) * tval
        if op.Params.print_lvl >= 2:
            rfo_step_report = "\t  Analytic derivative d(norm)/d(alpha) = %15.10lf\n" % analyticDerivative
            # + "\n\t------------------------------------------------\n")
            logger.info(rfo_step_report)

        # Calculate new scaling alpha value.
        # Equation 20, Besalu and Bofill, Theor. Chem. Acc., 1998, 100:265-274
        alpha += 2 * (trust * sqrt(dqtdq) - dqtdq) / analyticDerivative

    # end alpha RS-RFO iterations

    # TODO remove if this is indeed old
    # Crude/old way to limit step size if RS-RFO iterations
    if not converged or op.Params.simple_step_scaling:
        apply_intrafrag_step_scaling(dq)

    if op.Params.print_lvl >= 3:
        logger.debug("\tFinal scaled step dq:\n\n\t" + print_array_string(dq))

    # Get norm |dq|, unit vector, gradient and hessian in step direction
    # TODO double check Hevects[i] here instead of H ? as for NR
    rfo_dqnorm = sqrt(np.dot(dq, dq))
    logger.info("\tNorm of target step-size: %15.10f\n" % rfo_dqnorm)
    rfo_u = dq / rfo_dqnorm
    rfo_g = -1 * np.dot(fq, rfo_u)
    rfo_h = np.dot(rfo_u, np.dot(H, rfo_u))
    DEprojected = de_projected("RFO", rfo_dqnorm, rfo_g, rfo_h)
    if op.Params.print_lvl > 1:
        logger.info("\tRFO target step = %15.10f" % rfo_dqnorm)
        logger.info("\tRFO gradient = %15.10f" % rfo_g)
        logger.info("\tRFO hessian = %15.10f" % rfo_h)
    logger.info("\tProjected energy change by RFO approximation %15.5f\n" % DEprojected)

    return DEprojected, dq, rfo_u, rfo_g, rfo_h


def dq_p_rfo(fq, H):
    logger = logging.getLogger(__name__)
    hdim = len(fq)  # size of Hessian
    trust = op.Params.intrafrag_trust  # maximum step size
    # rfo_follow_root = op.Params.rfo_follow_root  # whether to follow root
    # rfo follow root is not currently implemented
    print_lvl = op.Params.print_lvl

    if print_lvl > 2:
        logger.info("\tHessian matrix\n" + print_mat_string(H))

    # Diagonalize H (technically only have to semi-diagonalize)
    h_eig_values, h_eig_vectors = symm_mat_eig(H)

    if print_lvl > 2:
        logger.info("\tEigenvalues of Hessian\n\n\t" + print_array_string(h_eig_values))
        logger.info("\tEigenvectors of Hessian (rows)\n" + print_mat_string(h_eig_vectors))

    # Construct diagonalized Hessian with evals on diagonal

    hess_diag = np.diag(h_eig_values)

    if print_lvl > 2:
        logger.info("\tH diagonal\n" + print_mat_string(hess_diag))

    logger.debug("\tFor P-RFO, assuming rfo_root=1, maximizing along lowest eigenvalue of Hessian.")
    logger.debug("\tLarger values of rfo_root are not yet supported.")

    rfo_root = 0
    """  TODO: use rfo_root to decide which eigenvectors are moved into the max/mu space.
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

    # number of degrees along which to maximize; assume 1 for now
    mu = 1

    logger.info("\tInternal forces in au:\n\n\t" + print_array_string(fq))

    fqTransformed = np.dot(h_eig_vectors, fq)  # gradient transformation
    logger.info("\tInternal forces in au, in Hevect basis:\n\n\t" + print_array_string(fqTransformed))
    # Build RFO max
    # Lowest eigenvalue of hessian augmented with corresponding gradient components

    maximize_rfo = np.zeros((mu + 1, mu + 1))
    maximize_rfo[:mu, :mu] = hess_diag[:mu, :mu]
    maximize_rfo[:mu, -1] = maximize_rfo[-1, :mu] = -fqTransformed[:mu]

    if print_lvl > 2:
        logger.info("\tRFO max\n" + print_mat_string(maximize_rfo))

    # Build RFO min
    # All remaining hessian eigenvalues augmented with gradient

    minimize_rfo = np.zeros((hdim - mu + 1, hdim - mu + 1))
    minimize_rfo[: hdim - mu, : hdim - mu] = hess_diag[mu:, mu:]
    minimize_rfo[: hdim - mu, -1] = minimize_rfo[-1, : hdim - mu] = -fqTransformed[mu:hdim]

    if print_lvl > 2:
        logger.info("\tRFO min\n" + print_mat_string(minimize_rfo))

    RFOMaxEValues, RFOMaxEVectors = symm_mat_eig(maximize_rfo)
    RFOMinEValues, RFOMinEVectors = symm_mat_eig(minimize_rfo)

    logger.info("\tRFO min eigenvalues:\n\n\t" + print_array_string(RFOMinEValues))
    logger.info("\tRFO max eigenvalues:\n\n\t" + print_array_string(RFOMaxEValues))

    if print_lvl > 2:
        logger.info("\tRFO min eigenvectors (rows) before normalization:\n" + print_mat_string(RFOMinEVectors))
        logger.info("\tRFO max eigenvectors (rows) before normalization:\n" + print_mat_string(RFOMaxEVectors))

    # Normalize max and min eigenvectors
    for i in range(mu + 1):
        if abs(RFOMaxEVectors[i, mu]) > 1.0e-10:
            tval = abs(abs_max(RFOMaxEVectors[i, 0:mu]) / RFOMaxEVectors[i, mu])
            if fabs(tval) < op.Params.rfo_normalization_max:
                RFOMaxEVectors[i] /= RFOMaxEVectors[i, mu]
    if print_lvl > 2:
        logger.info("\tRFO max eigenvectors (rows):\n" + print_mat_string(RFOMaxEVectors))

    for i in range(hdim - mu + 1):
        if abs(RFOMinEVectors[i][hdim - mu]) > 1.0e-10:
            tval = abs(abs_max(RFOMinEVectors[i, 0 : hdim - mu]) / RFOMinEVectors[i, hdim - mu])
            if fabs(tval) < op.Params.rfo_normalization_max:
                RFOMinEVectors[i] /= RFOMinEVectors[i, hdim - mu]
    if print_lvl > 2:
        logger.info("\tRFO min eigenvectors (rows):\n" + print_mat_string(RFOMinEVectors))

    VectorP = RFOMaxEVectors[mu, 0:mu]
    VectorN = RFOMinEVectors[rfo_root, 0 : hdim - mu]
    logger.debug("\tVector P\n\n\t" + print_array_string(VectorP))
    logger.debug("\tVector N\n\n\t" + print_array_string(VectorN))

    # Combines the eignvectors from RFO max and min
    prfoe_vector = np.zeros(hdim)
    prfoe_vector[0 : len(VectorP)] = VectorP
    prfoe_vector[len(VectorP) :] = VectorN

    prfo_step = np.dot(h_eig_vectors.transpose(), prfoe_vector)

    if print_lvl > 1:
        logger.info("\tRFO step in Hessian Eigenvector Basis\n\n\t" + print_array_string(prfoe_vector))
        logger.info("\tRFO step in original Basis\n\n\t" + print_array_string(prfo_step))

    dq = prfo_step

    # if not converged or op.Params.simple_step_scaling:
    apply_intrafrag_step_scaling(dq)

    # Get norm |dq|, unit vector, gradient and hessian in step direction
    # TODO double check Hevects[i] here instead of H ? as for NR
    rfo_dqnorm = sqrt(np.dot(dq, dq))
    logger.info("\tNorm of target step-size %15.10f" % rfo_dqnorm)
    rfo_u = dq / rfo_dqnorm
    rfo_g = -1 * np.dot(fq, rfo_u)
    rfo_h = np.dot(rfo_u, np.dot(H, rfo_u))
    DEprojected = de_projected("RFO", rfo_dqnorm, rfo_g, rfo_h)
    if op.Params.print_lvl > 1:
        logger.info("\t|RFO target step|  : %15.10f" % rfo_dqnorm)
        logger.info("\tRFO gradient       : %15.10f" % rfo_g)
        logger.info("\tRFO hessian        : %15.10f" % rfo_h)
    logger.info("\tProjected Delta(E) : %15.10f" % DEprojected)

    return DEprojected, dq, rfo_u, rfo_g, rfo_h


def dq_sd(fq):
    """ Take a step using steepest descent method

    Parameters
    ----------
    fq : ndarray
        forces in internal coordinates
    """

    logger = logging.getLogger(__name__)
    logger.info("\tTaking SD optimization step.")
    dim = len(fq)
    sd_h = op.Params.sd_hessian  # default value

    if len(oHistory.steps) > 1:
        previous_forces = oHistory.steps[-2].forces
        previous_dq = oHistory.steps[-2].Dq

        # Compute overlap of previous forces with current forces.
        previous_forces_u = previous_forces / np.linalg.norm(previous_forces)
        forces_u = fq / np.linalg.norm(fq)
        overlap = np.dot(previous_forces_u, forces_u)
        logger.debug("\tOverlap of current forces with previous forces %8.4lf" % overlap)
        previous_dq_norm = np.linalg.norm(previous_dq)

        if overlap > 0.50:
            # Magnitude of current force
            fq_norm = np.linalg.norm(fq)
            # Magnitude of previous force in step direction
            previous_forces_norm = v3d.dot(previous_forces, fq, dim) / fq_norm
            sd_h = (previous_forces_norm - fq_norm) / previous_dq_norm

    logger.info("\tEstimate of Hessian along step: %10.5e" % sd_h)
    dq = fq / sd_h

    apply_intrafrag_step_scaling(dq)

    sd_dqnorm = np.linalg.norm(dq)
    logger.info("\tNorm of target step-size %10.5f" % sd_dqnorm)

    # unit vector in step direction
    sd_u = dq / np.linalg.norm(dq)
    sd_g = -1.0 * sd_dqnorm

    DEprojected = de_projected("NR", sd_dqnorm, sd_g, sd_h)
    logger.info("\tProjected energy change by quadratic approximation: %20.5lf" % DEprojected)

    return DEprojected, dq, sd_u, sd_g, sd_h


def dq_backstep(o_molsys):
    """ takes a partial step backwards

    Notes
    -----
    Take partial backward step.  Update current step in history.
    Divide the last step size by 1/2 and displace from old geometry.
    HISTORY contains:
        consecutiveBacksteps : increase by 1
    HISTORY.STEP contains:
    No change to these:
        forces, geom, E, followedUnitVector, oneDgradient, oneDhessian
    Update these:
        Dq - cut in half
        projectedDE - recompute

    """

    logger = logging.getLogger(__name__)
    logger.warning("\tRe-doing last optimization step - smaller this time.\n")

    # Calling function shouldn't let this happen; this is a check for developer
    if len(oHistory.steps) < 2:
        raise OptError("Backstep called, but no history is available.")

    # Erase last, partial step data for current step.
    del oHistory.steps[-1]

    # Get data from previous step.
    fq = oHistory.steps[-1].forces
    dq = oHistory.steps[-1].Dq
    oneDgradient = oHistory.steps[-1].oneDgradient
    oneDhessian = oHistory.steps[-1].oneDhessian
    # Copy old geometry so displace doesn't change history
    geom = oHistory.steps[-1].geom.copy()

    # Compute new Dq and energy step projection.
    dq /= 2
    dqNorm = np.linalg.norm(dq)
    logger.info("\tNorm of target step-size %10.5f" % dqNorm)

    # Compute new Delta(E) projection.
    if op.Params.step_type == "RFO":
        DEprojected = de_projected("RFO", dqNorm, oneDgradient, oneDhessian)
    else:
        DEprojected = de_projected("NR", dqNorm, oneDgradient, oneDhessian)
    logger.info("\tProjected energy change : %20.5lf" % DEprojected)

    o_molsys.geom = geom  # uses setter; writes into all fragments
    dq_achieved = displace_molsys(o_molsys, dq, fq)

    dqNormActual = np.linalg.norm(dq_achieved)
    logger.info("\tNorm of achieved step-size %15.10f" % dqNormActual)

    oHistory.steps[-1].projectedDE = DEprojected
    oHistory.steps[-1].Dq[:] = dq_achieved

    return dq_achieved


def dq_linesearch(o_molsys, E, fq, H, computer):
    """ performs linesearch in direction of gradient

    Parameters
    ----------
    o_molsys : object
        optking molecular system
    E : double
        energy
    fq : ndarray
        forces in internal coordinates
    H : ndarray
        hessian in internal coordinates
    computer : computeWrapper
    """

    logger = logging.getLogger(__name__)
    s = op.Params.linesearch_step

    if len(oHistory.steps) > 1:
        s = norm(oHistory.steps[-2].Dq) / 2
        logger.info("\tModifying linesearch s to %10.6f" % s)

    logger.info("\n\tTaking LINESEARCH optimization step.")
    fq_unit = fq / sqrt(np.dot(fq, fq))
    logger.info("\tUnit vector in gradient direction.\n\n\t" + print_array_string(fq_unit) + "\n")
    Ea = E
    geomA = o_molsys.geom  # get copy of original geometry
    Eb = Ec = 0
    bounded = False
    ls_iter = 0
    stepScale = 2

    # Iterate until we find 3 points bounding minimum.
    while ls_iter < 10 and not bounded:
        ls_iter += 1

        if Eb == 0:
            logger.debug("\n\tStepping along forces distance %10.5f" % s)
            dq = s * fq_unit
            dq_achieved = displace_molsys(o_molsys, dq, fq)
            xyz = o_molsys.geom
            logger.debug("\tComputing energy at this point now.")
            Eb = computer.compute(xyz, driver="energy", return_full=False)

            o_molsys.geom = geomA  # reset geometry to point A

        if Ec == 0:
            logger.debug("\n\tStepping along forces distance %10.5f" % (stepScale * s))
            dq = (stepScale * s) * fq_unit
            dq_achieved = displace_molsys(o_molsys, dq, fq)
            xyz = o_molsys.geom
            logger.debug("\tComputing energy at this point now.")
            Ec = computer.compute(xyz, driver="energy", return_full=False)
            o_molsys.geom = geomA  # reset geometry to point A

        logger.info("\n\tCurrent linesearch bounds.\n")
        logger.info("\t s=%7.5f, Ea=%17.12f" % (0, Ea))
        logger.info("\t s=%7.5f, Eb=%17.12f" % (s, Eb))
        logger.info("\t s=%7.5f, Ec=%17.12f\n" % (stepScale * s, Ec))

        if Eb < Ea and Eb < Ec:
            # second point is lowest do projection
            logger.debug("\tMiddle point is lowest energy. Good. Projecting minimum.")
            Sa = 0.0
            Sb = s
            Sc = stepScale * s

            A = np.zeros((2, 2))
            A[0, 0] = Sc * Sc - Sb * Sb
            A[0, 1] = Sc - Sb
            A[1, 0] = Sb * Sb - Sa * Sa
            A[1, 1] = Sb - Sa
            B = np.zeros(2)
            B[0] = Ec - Eb
            B[1] = Eb - Ea
            x = np.linalg.solve(A, B)
            Xmin = -x[1] / (2 * x[0])

            logger.debug("\tParabolic fit ax^2 + bx + c along gradient.")
            logger.debug("\t *a = %15.10f" % x[0])
            logger.debug("\t *b = %15.10f" % x[1])
            logger.debug("\t *c = %15.10f" % Ea)
            Emin_projected = x[0] * Xmin * Xmin + x[1] * Xmin + Ea
            dq = Xmin * fq_unit
            logger.info("\tProjected step size to minimum is %12.6f" % Xmin)
            dq_achieved = displace_molsys(o_molsys, dq, fq)
            xyz = o_molsys.geom
            logger.debug("\tComputing energy at projected point.")
            Emin = computer.compute(xyz, driver="energy", return_full=False)
            logger.info("\tProjected energy along line: %15.10f" % Emin_projected)
            logger.info("\t   Actual energy along line: %15.10f" % Emin)

            bounded = True

        elif Ec < Eb and Ec < Ea:
            # unbounded.  increase step size
            logger.debug("\tSearching with larger step beyond 3rd point.")
            s *= stepScale
            Eb = Ec
            Ec = 0

        else:
            logger.debug("\tSearching with smaller step between first 2 points.")
            s *= 0.5
            Ec = Eb
            Eb = 0

    # get norm |q| and unit vector in the step direction
    ls_dqnorm = np.linalg.norm(dq_achieved)
    ls_u = dq_achieved / ls_dqnorm

    # get gradient and hessian in step direction
    ls_g = -1 * np.dot(fq, ls_u)  # should be unchanged
    ls_h = np.dot(ls_u, np.dot(H, ls_u))

    if op.Params.print_lvl > 1:
        logger.info("\n\\t|target step|: %15.10f" % ls_dqnorm)
        logger.info("\tLS_gradient     : %15.10f" % ls_g)
        logger.info("\tLS_hessian      : %15.10f" % ls_h)

    DEprojected = de_projected("NR", ls_dqnorm, ls_g, ls_h)
    logger.debug("\tProjected quadratic energy change using full Hessian: %15.10f\n" % DEprojected)

    oHistory.nuclear_repulsion_energy = computer.trajectory[-1]["properties"]["nuclear_repulsion_energy"]

    return DEprojected, dq_achieved, ls_u, ls_g, ls_h

    # Scale fq into aJ for printing
    # fq_aJ = o_molsys.q_show_forces(fq)
    # displace_molsys(o_molsys, dq, fq_aJ)

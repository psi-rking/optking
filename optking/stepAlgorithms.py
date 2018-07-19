# Functions for step algorithms: Newton-Raphson, Rational Function Optimization,
# Steepest Descent.
import numpy as np
from math import sqrt, fabs
# from .optParams import Params # this will not cause changes in trust to persist
import logging

import v3d
import optExceptions
import optparams as op
from history import oHistory
from displace import displace
from intcosMisc import qShowForces
from addIntcos import linearBendCheck
from misc import isDqSymmetric
from printTools import printArrayString, printMatString
from linearAlgebra import absMax, symmMatEig, asymmMatEig, symmMatInv, norm


# This function and its components:
# 1. Computes Dq, the step in internal coordinates.
# 2. Calls displace and attempts to take the step.
# 3. Updates history with results.
def Dq(oMolsys, E, qForces, H, stepType=None, energy_function=None, o_json=None):
    if len(H) == 0 or len(qForces) == 0:
        return np.zeros((0), float)

    if not stepType:
        stepType = op.Params.step_type

    if stepType == 'NR':
        return Dq_NR(oMolsys, E, qForces, H)
    elif stepType == 'RFO':
        return Dq_RFO(oMolsys, E, qForces, H)
    elif stepType == 'SD':
        return Dq_SD(oMolsys, E, qForces)
    elif stepType == 'BACKSTEP':
        return Dq_BACKSTEP(oMolsys)
    elif stepType == 'P_RFO':
        return Dq_P_RFO(oMolsys, E, qForces, H)
    elif stepType == 'LINESEARCH':
        return Dq_LINESEARCH(oMolsys, E, qForces, H, energy_function, o_json)
    else:
        raise optExceptions.OptFail('Dq: step type not yet implemented')


# Apply crude maximum step limit by scaling.
def applyIntrafragStepScaling(dq):
    logger = logging.getLogger(__name__)
    trust = op.Params.intrafrag_trust
    if sqrt(np.dot(dq, dq)) > trust:
        scale = trust / sqrt(np.dot(dq, dq))
        logger.info("\tStep length exceeds trust radius of %10.5f." % trust)
        logger.info("\tScaling displacements by %10.5f" % scale)
        dq *= scale
    return


# Compute energy change along one dimension
def DE_projected(model, step, grad, hess):
    if model == 'NR':
        return (step * grad + 0.5 * step * step * hess)
    elif model == 'RFO':
        return (step * grad + 0.5 * step * step * hess) / (1 + step * step)
    else:
        raise optExceptions.OptFail("DE_projected does not recognize model.")


# geometry and E are just for passing
# at present we are not storing the ACTUAL dq but the attempted
def Dq_NR(oMolsys, E, fq, H):
    logger = logging.getLogger(__name__)
    logger.info("\tTaking NR optimization step.")

    # Hinv fq = dq
    Hinv = symmMatInv(H, redundant=True)
    dq = np.dot(Hinv, fq)

    # applies maximum internal coordinate change
    applyIntrafragStepScaling(dq)

    # get norm |q| and unit vector in the step direction
    nr_dqnorm = sqrt(np.dot(dq, dq))
    nr_u = dq.copy() / nr_dqnorm
    logger.info("\tNorm of target step-size %15.10lf" % nr_dqnorm)

    # get gradient and hessian in step direction
    nr_g = -1 * np.dot(fq, nr_u)  # gradient, not force
    nr_h = np.dot(nr_u, np.dot(H, nr_u))

    if op.Params.print_lvl > 1:
        logger.info('\tNR target step|: %15.10f' % nr_dqnorm)
        logger.info('\tNR_gradient: %15.10f' % nr_g)
        logger.info('\tNR_hessian: %15.10f' % nr_h)
    DEprojected = DE_projected('NR', nr_dqnorm, nr_g, nr_h)
    logger.info("\tProjected energy change by quadratic approximation: %10.10lf\n"
                % DEprojected)

    # Scale fq into aJ for printing
    fq_aJ = qShowForces(oMolsys.intcos, fq)
    displace(oMolsys._fragments[0].intcos, oMolsys._fragments[0].geom, dq, fq_aJ)
    dq_actual = sqrt(np.dot(dq, dq))
    logger.info("\tNorm of achieved step-size %15.10f" % dq_actual)

    # Symmetrize the geometry for next step
    # symmetrize_geom()

    # save values in step data
    oHistory.appendRecord(DEprojected, dq, nr_u, nr_g, nr_h)

    # Can check full geometry, but returned indices will correspond then to that.
    linearList = linearBendCheck(oMolsys.intcos, oMolsys.geom, dq)
    if linearList:
        raise optExceptions.AlgFail("New linear angles", newLinearBends=linearList)

    return dq


# Take Rational Function Optimization step
def Dq_RFO(oMolsys, E, fq, H):
    logger = logging.getLogger(__name__)
    logger.info("\tTaking RFO optimization step.")
    dim = len(fq)
    dq = np.zeros((dim), float)  # To be determined and returned.
    trust = op.Params.intrafrag_trust  # maximum step size
    max_projected_rfo_iter = 25  # max. # of iterations to try to converge RS-RFO
    rfo_follow_root = op.Params.rfo_follow_root  # whether to follow root
    rfo_root = op.Params.rfo_root  # if following, which root to follow

    # Determine the eigenvectors/eigenvalues of H.
    Hevals, Hevects = symmMatEig(H)

    # Build the original, unscaled RFO matrix.
    RFOmat = np.zeros((dim + 1, dim + 1), float)
    for i in range(dim):
        for j in range(dim):
            RFOmat[i, j] = H[i, j]
        RFOmat[i, dim] = RFOmat[dim, i] = -fq[i]

    if op.Params.print_lvl >= 4:
        logger.debug("\tOriginal, unscaled RFO matrix:\n\n" +
                     printMatString(RFOmat))

    symm_rfo_step = False
    SRFOmat = np.zeros((dim + 1, dim + 1), float)  # For scaled RFO matrix.
    converged = False
    dqtdq = 10  # square of norm of step
    alpha = 1.0  # scaling factor for RS-RFO, scaling matrix is sI

    last_iter_evect = np.zeros((dim), float)
    if rfo_follow_root and len(oHistory.steps) > 1:
        last_iter_evect[:] = oHistory.steps[
            -2].followedUnitVector  # RFO vector from previous geometry step

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
            logger.debug("\tScaled RFO matrix.\n\n" +
                         printMatString(SRFOmat))

        # Find the eigenvectors and eigenvalues of RFO matrix.
        SRFOevals, SRFOevects = asymmMatEig(SRFOmat)

        if op.Params.print_lvl >= 4:
            logger.debug("\tEigenvectors of scaled RFO matrix.\n\n" +
                         printMatString(SRFOevects))

        if op.Params.print_lvl >= 2:
            logger.debug("\tEigenvalues of scaled RFO matrix.\n\n\t" +
                         printArrayString(SRFOevals))
            logger.debug("\tFirst eigenvector (unnormalized) of scaled RFO matrix.\n\n\t" +
                         printArrayString(SRFOevects[0]))

        # Do intermediate normalization.  RFO paper says to scale eigenvector
        # to make the last element equal to 1. Bogus evect leads can be avoided
        # using root following.
        for i in range(dim + 1):
            # How big is dividing going to make the largest element?
            # Same check occurs below for acceptability.
            if fabs(SRFOevects[i][dim]) > 1.0e-10:
                tval = absMax(SRFOevects[i] / SRFOevects[i][dim])
                if tval < op.Params.rfo_normalization_max:
                    for j in range(dim + 1):
                        SRFOevects[i, j] /= SRFOevects[i, dim]

        if op.Params.print_lvl >= 4:
            logger.debug("\tAll scaled RFO eigenvectors (rows).\n\n" +
                         printMatString(SRFOevects))

        # Use input rfo_root
        # If root-following is turned off, then take the eigenvector with the
        # rfo_root'th lowest eigvenvalue. If its the first iteration, then do the same.
        # In subsequent steps, overlaps will be checked.
        if not rfo_follow_root or len(oHistory.steps) < 2:

            # Determine root only once at beginning ?
            if alphaIter == 0:
                logger.info("\tChecking RFO solution %d." % (rfo_root + 1))

                for i in range(rfo_root, dim + 1):
                    # Check symmetry of root.
                    dq[:] = SRFOevects[i, 0:dim]
                    if not op.Params.accept_symmetry_breaking:
                        symm_rfo_step = isDqSymmetric(oMolsys.intcos, oMolsys.geom, dq)

                        if not symm_rfo_step:  # Root is assymmetric so reject it.
                            logger.warning("\tRejecting RFO root %d because it breaks \
                                           the molecular point group." % (rfo_root+1))
                            continue

                    # Check normalizability of root.
                    if fabs(SRFOevects[i][dim]) < 1.0e-10:  # don't even try to divide
                        logger.warning("\tRejecting RFO root %d because normalization \
                                       gives large value." % (rfo_root + 1))
                        continue
                    tval = absMax(SRFOevects[i] / SRFOevects[i][dim])
                    if tval > op.Params.rfo_normalization_max:  # matching test in code above
                        logger.warning("\tRejecting RFO root %d because normalization \
                                       gives large value." % (rfo_root + 1))
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
            dots = np.array(
                [v3d.dot(SRFOevects[i], last_iter_evect, dim) for i in range(dim)], float)
            bestfit = np.argmax(dots)
            if bestfit != rfo_root:
                logger.info("\tRoot-following has changed rfo_root value to %d."
                            % (bestfit + 1))
                rfo_root = bestfit

        if alphaIter == 0:
            logger.info("\tUsing RFO solution %d." % (rfo_root + 1))
        last_iter_evect[:] = SRFOevects[rfo_root][0:dim]  # omit last column on right

        # Print only the lowest eigenvalues/eigenvectors
        if op.Params.print_lvl >= 2:
            logger.info("\trfo_root is %d" % (rfo_root + 1))
            for i in range(dim + 1):
                if SRFOevals[i] < -1e-6 or i < rfo_root:
                    eigen_val_vec = ("\n\tScaled RFO eigenvalue %d:\n\t%15.10lf (or 2*%-15.10lf)\n"
                                     % (i + 1, SRFOevals[i], SRFOevals[i] / 2))
                    eigen_val_vec += ("\n\teigenvector:\n\t")
                    eigen_val_vec += printArrayString(SRFOevects[i])
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
            rfo_step_report = ("\n\n\t Iter      |step|        alpha        rfo_root"
                               + "\n\t------------------------------------------------"
                               + "\n\t %5d%12.5lf%14.5lf%12d\n"
                               % (alphaIter + 1, sqrt(dqtdq), alpha, rfo_root + 1))

        elif alphaIter > 0 and not op.Params.simple_step_scaling:
            rfo_step_report = ("\t%5d%12.5lf%14.5lf%12d\n"
                               % (alphaIter + 1, sqrt(dqtdq), alpha, rfo_root + 1))

        # Find the analytical derivative, d(norm step squared) / d(alpha)
        rfo_step_report += ("\t------------------------------------------------\n")
        logger.info(rfo_step_report)
        Lambda = -1 * v3d.dot(fq, dq, dim)
        if op.Params.print_lvl >= 2:
            disp_forces = ("\tDisplacement and Forces\n\n")
            disp_forces += ("\tDq:" + printArrayString(dq, dim))
            disp_forces += ("\tFq:" + printArrayString(fq, dim))
            logger.info(disp_forces)
            logger.info("\tLambda calculated by (dq^t).(-f) = %15.10lf\n" % Lambda)

        # Calculate derivative of step size wrt alpha.
        # Equation 20, Besalu and Bofill, Theor. Chem. Acc., 1999, 100:265-274
        tval = 0
        for i in range(dim):
            tval += (pow(v3d.dot(Hevects[i], fq, dim), 2)) / (pow(
                (Hevals[i] - Lambda * alpha), 3))

        analyticDerivative = 2 * Lambda / (1 + alpha * dqtdq) * tval
        if op.Params.print_lvl >= 2:
            rfo_step_report += ("\t  Analytic derivative d(norm)/d(alpha) = %15.10lf\n"
                                % analyticDerivative
                                + "\n\t------------------------------------------------\n")

        logger.info(rfo_step_report)
        # Calculate new scaling alpha value.
        # Equation 20, Besalu and Bofill, Theor. Chem. Acc., 1999, 100:265-274
        alpha += 2 * (trust * sqrt(dqtdq) - dqtdq) / analyticDerivative

    # end alpha RS-RFO iterations

    # Crude/old way to limit step size if RS-RFO iterations
    if not converged or op.Params.simple_step_scaling:
        applyIntrafragStepScaling(dq)

    if op.Params.print_lvl >= 3:
        logger.debug("\tFinal scaled step dq:\n\n\t" + printArrayString(dq))

    # Get norm |dq|, unit vector, gradient and hessian in step direction
    # TODO double check Hevects[i] here instead of H ? as for NR
    rfo_dqnorm = sqrt(np.dot(dq, dq))
    logger.info("\tNorm of target step-size \n\n\t %15.10f\n" % rfo_dqnorm)
    rfo_u = dq.copy() / rfo_dqnorm
    rfo_g = -1 * np.dot(fq, rfo_u)
    rfo_h = np.dot(rfo_u, np.dot(H, rfo_u))
    DEprojected = DE_projected('RFO', rfo_dqnorm, rfo_g, rfo_h)
    if op.Params.print_lvl > 1:
        logger.info('\tRFO target step = %15.10f\n' % rfo_dqnorm)
        logger.info('\tRFO gradient = %15.10f\n' % rfo_g)
        logger.info('\tRFO hessian = %15.10f\n' % rfo_h)
    logger.info("\tProjected energy change by RFO approximation %15.5f\n" % DEprojected)

    # Scale fq into aJ for printing
    fq_aJ = qShowForces(oMolsys.intcos, fq)

    # this won't work for multiple fragments yet until dq and fq get cut up.
    for F in oMolsys._fragments:
        displace(F.intcos, F.geom, dq, fq_aJ)
    # For now, saving RFO unit vector and using it in projection to match C++ code,
    # could use actual Dq instead.
    dqnorm_actual = sqrt(np.dot(dq, dq))
    logger.info("\tNorm of achieved step-size \n\n\t %15.10f\n" % dqnorm_actual)

    # To test step sizes
    # x_before = original geometry
    # x_after = new geometry
    # masses
    # change = 0.0;
    # for i in range(Natom):
    #   for xyz in range(3):
    #     change += (x_before[3*i+xyz] - x_after[3*i+xyz]) * (x_before[3*i+xyz] - x_after[3*i+xyz])
    #             * masses[i]
    # change = sqrt(change);
    # printxopt("Step-size in mass-weighted cartesian coordinates [bohr (amu)^1/2] : %20.10lf\n"
    #    % change)

    # printxopt("\tSymmetrizing new geometry\n")
    # geom = symmetrizeXYZ(geom)
    oHistory.appendRecord(DEprojected, dq, rfo_u, rfo_g, rfo_h)
    linearList = linearBendCheck(oMolsys.intcos, oMolsys.geom, dq)
    if linearList:
        raise optExceptions.AlgFail("New linear angles", newLinearBends=linearList)

    # Before quitting, make sure step is reasonable.  It should only be
    # screwball if we are using the "First Guess" after the back-transformation failed.
    if sqrt(np.dot(dq, dq)) > 10 * trust:
        raise optExceptions.AlgFail("opt.py: Step is far too large.")

    return dq


def Dq_P_RFO(oMolsys, E, fq, H):
    logger = logging.getLogger(__name__)
    Hdim = len(fq)  # size of Hessian
    trust = op.Params.intrafrag_trust  # maximum step size
    # rfo_follow_root = op.Params.rfo_follow_root  # whether to follow root
    # rfo follow root is not currently implemented
    print_lvl = op.Params.print_lvl

    if print_lvl > 2:
        logger.info("\tHessian matrix\n" + printMatString(H))

    # Diagonalize H (technically only have to semi-diagonalize)
    hEigValues, hEigVectors = symmMatEig(H)

    if print_lvl > 2:
        logger.info("\tEigenvalues of Hessian\n\n\t" + printArrayString(hEigValues) + "\n")
        logger.info("\tEigenvectors of Hessian (rows)\n" + printMatString(hEigVectors))

    # Construct diagonalized Hessian with evals on diagonal
    HDiag = np.zeros((Hdim, Hdim), float)
    for i in range(Hdim):
        HDiag[i, i] = hEigValues[i]

    if print_lvl > 2:
        logger.info("\tH diagonal\n" + printMatString(HDiag))

    logger.debug(
        "\tFor P-RFO, assuming rfo_root=1, maximizing along lowest eigenvalue of Hessian.")
    logger.debug("\tLarger values of rfo_root are not yet supported.")

    rfo_root = 0
    """  TODO: use rfo_root to decide which eigenvectors are moved into the max/mu space.
    if not rfo_follow_root or len(oHistory.steps) < 2:
        rfo_root = op.Params.rfo_root
        printxopt("\tMaximizing along %d lowest eigenvalue of Hessian.\n" % (rfo_root+1) )
    else:
        last_iter_evect = history[-1].Dq
        dots = np.array([v3d.dot(hEigVectors[i],last_iter_evect,Hdim) for i in range(Hdim)], float)
        rfo_root = np.argmax(dots)
        printxopt("\tOverlaps with previous step checked for root-following.\n")
        printxopt("\tMaximizing along %d lowest eigenvalue of Hessian.\n" % (rfo_root+1) )
    """

    # number of degrees along which to maximize; assume 1 for now
    mu = 1

    logger.info("\tInternal forces in au:\n\n\t" + printArrayString(fq) + "\n")

    fqTransformed = np.dot(hEigVectors, fq)  # gradient transformation
    logger.info("\tInternal forces in au, in Hevect basis:\n\n\t"
                + printArrayString(fqTransformed) + "\n")
    # Build RFO max
    maximizeRFO = np.zeros((mu + 1, mu + 1), float)
    for i in range(mu):
        maximizeRFO[i, i] = hEigValues[i]
        maximizeRFO[i, -1] = -fqTransformed[i]
        maximizeRFO[-1, i] = -fqTransformed[i]
    if print_lvl > 2:
        logger.info("\tRFO max\n" + printMatString(maximizeRFO))

    # Build RFO min
    minimizeRFO = np.zeros((Hdim - mu + 1, Hdim - mu + 1), float)
    for i in range(0, Hdim - mu):
        minimizeRFO[i, i] = HDiag[i + mu, i + mu]
        minimizeRFO[i, -1] = -fqTransformed[i + mu]
        minimizeRFO[-1, i] = -fqTransformed[i + mu]
    if print_lvl > 2:
        logger.info("\tRFO min\n" + printMatString(minimizeRFO))

    RFOMaxEValues, RFOMaxEVectors = symmMatEig(maximizeRFO)
    RFOMinEValues, RFOMinEVectors = symmMatEig(minimizeRFO)

    logger.info("\tRFO min eigenvalues:\n\n\t" + printArrayString(RFOMinEValues))
    logger.info("\tRFO max eigenvalues:\n\n\t" + printArrayString(RFOMaxEValues) + "\n")

    if print_lvl > 2:
        logger.info("\tRFO min eigenvectors (rows) before normalization:\n"
                    + printMatString(RFOMinEVectors))
        logger.info("\tRFO max eigenvectors (rows) before normalization:\n"
                    + printMatString(RFOMaxEVectors))

    # Normalize max and min eigenvectors
    for i in range(mu + 1):
        if abs(RFOMaxEVectors[i, mu]) > 1.0e-10:
            tval = abs(absMax(RFOMaxEVectors[i, 0:mu]) / RFOMaxEVectors[i, mu])
            if fabs(tval) < op.Params.rfo_normalization_max:
                RFOMaxEVectors[i] /= RFOMaxEVectors[i, mu]
    if print_lvl > 2:
        logger.info("\tRFO max eigenvectors (rows):\n" + printMatString(RFOMaxEVectors))

    for i in range(Hdim - mu + 1):
        if abs(RFOMinEVectors[i][Hdim - mu]) > 1.0e-10:
            tval = abs(
                absMax(RFOMinEVectors[i, 0:Hdim - mu]) / RFOMinEVectors[i, Hdim - mu])
            if fabs(tval) < op.Params.rfo_normalization_max:
                RFOMinEVectors[i] /= RFOMinEVectors[i, Hdim - mu]
    if print_lvl > 2:
        logger.info("\tRFO min eigenvectors (rows):\n" + printMatString(RFOMinEVectors))

    VectorP = RFOMaxEVectors[mu, 0:mu]
    VectorN = RFOMinEVectors[rfo_root, 0:Hdim - mu]
    logger.debug("\tVector P\n\n\t" + str(VectorP) + '\n')
    logger.debug("\tVector N\n\n\t" + str(VectorN) + '\n')

    # Combines the eignvectors from RFO max and min
    PRFOEVector = np.zeros(Hdim, float)
    PRFOEVector[0:len(VectorP)] = VectorP
    PRFOEVector[len(VectorP):] = VectorN

    PRFOStep = np.dot(hEigVectors.transpose(), PRFOEVector)

    if print_lvl > 1:
        logger.info("\tRFO step in Hessian Eigenvector Basis\n\n\t"
                    + printArrayString(PRFOEVector) + "\n")
        logger.info("\tRFO step in original Basis\n\n\t"
                    + printArrayString(PRFOStep) + "\n")

    dq = PRFOStep

    # if not converged or op.Params.simple_step_scaling:
    applyIntrafragStepScaling(dq)

    # Get norm |dq|, unit vector, gradient and hessian in step direction
    # TODO double check Hevects[i] here instead of H ? as for NR
    rfo_dqnorm = sqrt(np.dot(dq, dq))
    logger.info("\tNorm of target step-size %15.10f" % rfo_dqnorm)
    rfo_u = dq.copy() / rfo_dqnorm
    rfo_g = -1 * np.dot(fq, rfo_u)
    rfo_h = np.dot(rfo_u, np.dot(H, rfo_u))
    DEprojected = DE_projected('RFO', rfo_dqnorm, rfo_g, rfo_h)
    if op.Params.print_lvl > 1:
        logger.info('\t|RFO target step|  : %15.10f' % rfo_dqnorm)
        logger.info('\tRFO gradient       : %15.10f' % rfo_g)
        logger.info('\tRFO hessian        : %15.10f' % rfo_h)
    logger.info("\tProjected Delta(E) : %15.10f\n" % DEprojected)

    # Scale fq into aJ for printing
    fq_aJ = qShowForces(oMolsys.intcos, fq)

    # this won't work for multiple fragments yet until dq and fq get cut up.
    for F in oMolsys._fragments:
        displace(F.intcos, F.geom, dq, fq_aJ)

    # For now, saving RFO unit vector and using it in projection to match C++ code,
    # could use actual Dq instead.
    dqnorm_actual = sqrt(np.dot(dq, dq))
    logger.info("\tNorm of achieved step-size %15.10f" % dqnorm_actual)

    oHistory.appendRecord(DEprojected, dq, rfo_u, rfo_g, rfo_h)

    linearList = linearBendCheck(oMolsys.intcos, oMolsys.geom, dq)
    if linearList:
        raise optExceptions.AlgFail("New linear angles", newLinearBends=linearList)

    # Before quitting, make sure step is reasonable.  It should only be
    # screwball if we are using the "First Guess" after the back-transformation failed.
    if sqrt(np.dot(dq, dq)) > 10 * trust:
        raise optExceptions.AlgFail("opt.py: Step is far too large.")

    return dq


# Take Steepest Descent step
def Dq_SD(oMolsys, E, fq):
    logger = logging.getLogger(__name__)
    logger.info("\tTaking SD optimization step.")
    dim = len(fq)
    sd_h = op.Params.sd_hessian  # default value

    if len(oHistory.steps) > 1:
        previous_forces = oHistory.steps[-2].forces
        previous_dq = oHistory.steps[-2].Dq

        # Compute overlap of previous forces with current forces.
        previous_forces_u = previous_forces.copy() / np.linalg.norm(previous_forces)
        forces_u = fq.copy() / np.linalg.norm(fq)
        overlap = np.dot(previous_forces_u, forces_u)
        logger.info("\tOverlap of current forces with previous forces %8.4lf" % overlap)
        previous_dq_norm = np.linalg.norm(previous_dq)

        if overlap > 0.50:
            # Magnitude of current force
            fq_norm = np.linalg.norm(fq)
            # Magnitude of previous force in step direction
            previous_forces_norm = v3d.dot(previous_forces, fq, dim) / fq_norm
            sd_h = (previous_forces_norm - fq_norm) / previous_dq_norm

    logger.info("\tEstimate of Hessian along step: %10.5e\n" % sd_h)
    dq = fq / sd_h

    applyIntrafragStepScaling(dq)

    sd_dqnorm = np.linalg.norm(dq)
    logger.info("\tNorm of target step-size %10.5f\n" % sd_dqnorm)

    # unit vector in step direction
    sd_u = dq.copy() / np.linalg.norm(dq)
    sd_g = -1.0 * sd_dqnorm

    DEprojected = DE_projected('NR', sd_dqnorm, sd_g, sd_h)
    logger.info(
        "\tProjected energy change by quadratic approximation: %20.5lf\n" % DEprojected)

    fq_aJ = qShowForces(oMolsys.intcos, fq)  # for printing
    displace(oMolsys._fragments[0].intcos, oMolsys._fragments[0].geom, dq, fq_aJ)
    logger.debug("\tSuccessfully displaced. Back to step algs")
    dqnorm_actual = np.linalg.norm(dq)
    logger.info("\tNorm of achieved step-size %15.10f\n" % dqnorm_actual)

    # Symmetrize the geometry for next step
    # symmetrize_geom()

    oHistory.appendRecord(DEprojected, dq, sd_u, sd_g, sd_h)

    linearList = linearBendCheck(oMolsys.intcos, oMolsys.geom, dq)
    if linearList:
        raise optExceptions.AlgFail("New linear angles", newLinearBends=linearList)

    return dq


# Take partial backward step.  Update current step in history.
# Divide the last step size by 1/2 and displace from old geometry.
# HISTORY contains:
#   consecutiveBacksteps : increase by 1
#   HISTORY.STEP contains:
#     No change to these:
#       forces, geom, E, followedUnitVector, oneDgradient, oneDhessian
#     Update these:
#       Dq - cut in half
#       projectedDE - recompute


def Dq_BACKSTEP(oMolsys):
    logger = logging.getLogger(__name__)
    logger.warning("\tRe-doing last optimization step - smaller this time.\n")

    # Calling function shouldn't let this happen; this is a check for developer
    if len(oHistory.steps) < 2:
        raise optExceptions.OptFail("Backstep called, but no history is available.")

    # Erase last, partial step data for current step.
    del oHistory.steps[-1]

    # Get data from previous step.
    fq = oHistory.steps[-1].forces
    dq = oHistory.steps[-1].Dq
    oneDgradient = oHistory.steps[-1].oneDgradient
    oneDhessian = oHistory.steps[-1].oneDhessian
    # Copy old geometry so displace doesn't change history
    geom = oHistory.steps[-1].geom.copy()

    # printxopt('test geom old from history\n')
    # printMat(oMolsys.geom)

    # Compute new Dq and energy step projection.
    dq /= 2
    dqNorm = np.linalg.norm(dq)
    logger.info("\tNorm of target step-size %10.5f\n" % dqNorm)

    # Compute new Delta(E) projection.
    if op.Params.step_type == 'RFO':
        DEprojected = DE_projected('RFO', dqNorm, oneDgradient, oneDhessian)
    else:
        DEprojected = DE_projected('NR', dqNorm, oneDgradient, oneDhessian)
    logger.info("\tProjected energy change : %20.5lf\n" % DEprojected)

    fq_aJ = qShowForces(oMolsys.intcos, fq)  # for printing
    # Displace from previous geometry
    displace(oMolsys._fragments[0].intcos, geom, dq, fq_aJ)
    oMolsys.geom = geom  # uses setter; writes into all fragments

    dqNormActual = np.linalg.norm(dq)
    logger.info("\tNorm of achieved step-size %15.10f\n" % dqNormActual)
    # Symmetrize the geometry for next step
    # symmetrize_geom()

    # Update the history entries which changed.
    oHistory.steps[-1].projectedDE = DEprojected
    oHistory.steps[-1].Dq[:] = dq

    linearList = linearBendCheck(oMolsys.intcos, oMolsys.geom, dq)
    if linearList:
        raise optExceptions.AlgFail("New linear angles", newLinearBends=linearList)

    return dq


# Take Rational Function Optimization step
def Dq_LINESEARCH(oMolsys, E, fq, H, energy_function, o_json):
    logger = logging.getLogger(__name__)
    s = op.Params.linesearch_step

    if len(oHistory.steps) > 1:
        s = norm(oHistory.steps[-2].Dq) / 2
        logger.info("\tModifying linesearch s to %10.6f\n" % s)

    logger.info("\n\tTaking LINESEARCH optimization step.\n")
    fq_unit = fq / sqrt(np.dot(fq, fq))
    logger.info("\tUnit vector in gradient direction.\n\n\t"
                + printArrayString(fq_unit) + "\n")
    Ea = E
    geomA = oMolsys.geom  # get copy of original geometry
    Eb = Ec = 0
    bounded = False
    ls_iter = 0
    stepScale = 2

    # Iterate until we find 3 points bounding minimum.
    while ls_iter < 10 and not bounded:
        ls_iter += 1

        if Eb == 0:
            logger.info("\n\tStepping along forces distance %10.5f\n" % s)
            dq = s * fq_unit
            fq_aJ = qShowForces(oMolsys.intcos, fq)
            displace(oMolsys._fragments[0].intcos, oMolsys._fragments[0].geom, dq, fq_aJ)
            xyz = oMolsys.geom
            logger.debug("\tComputing energy at this point now.\n")
            Eb, nuc = energy_function(xyz, o_json)
            oMolsys.geom = geomA  # reset geometry to point A

        if Ec == 0:
            logger.info("\n\tStepping along forces distance %10.5f\n" % (stepScale * s))
            dq = (stepScale * s) * fq_unit
            fq_aJ = qShowForces(oMolsys.intcos, fq)
            displace(oMolsys._fragments[0].intcos, oMolsys._fragments[0].geom, dq, fq_aJ)
            xyz = oMolsys.geom
            logger.debug("\tComputing energy at this point now.\n")
            Ec, nuc = energy_function(xyz, o_json)
            oMolsys.geom = geomA  # reset geometry to point A

        logger.info("\n\tCurrent linesearch bounds.\n")
        logger.info("\t s=%7.5f, Ea=%17.12f\n" % (0, Ea))
        logger.info("\t s=%7.5f, Eb=%17.12f\n" % (s, Eb))
        logger.info("\t s=%7.5f, Ec=%17.12f\n" % (stepScale * s, Ec))

        if Eb < Ea and Eb < Ec:
            # second point is lowest do projection
            logger.info("\tMiddle point is lowest energy. Good. Projecting minimum.\n")
            Sa = 0.0
            Sb = s
            Sc = stepScale * s

            A = np.zeros((2, 2), float)
            A[0, 0] = Sc * Sc - Sb * Sb
            A[0, 1] = Sc - Sb
            A[1, 0] = Sb * Sb - Sa * Sa
            A[1, 1] = Sb - Sa
            B = np.zeros((2), float)
            B[0] = Ec - Eb
            B[1] = Eb - Ea
            x = np.linalg.solve(A, B)
            Xmin = -x[1] / (2 * x[0])

            logger.debug("\tParabolic fit ax^2 + bx + c along gradient.\n")
            logger.debug("\t *a = %15.10f\n" % x[0])
            logger.debug("\t *b = %15.10f\n" % x[1])
            logger.debug("\t *c = %15.10f\n" % Ea)
            Emin_projected = x[0] * Xmin * Xmin + x[1] * Xmin + Ea
            dq = Xmin * fq_unit
            logger.info("\tProjected step size to minimum is %12.6f\n" % Xmin)
            displace(oMolsys._fragments[0].intcos, oMolsys._fragments[0].geom, dq, fq_aJ)
            xyz = oMolsys.geom
            logger.debug("\tComputing energy at projected point.\n")
            Emin, nuc = energy_function(xyz, o_json)
            logger.info("\tProjected energy along line: %15.10f\n" % Emin_projected)
            logger.info("\t   Actual energy along line: %15.10f\n" % Emin)

            bounded = True

        elif Ec < Eb and Ec < Ea:
            # unbounded.  increase step size
            logger.debug("\tSearching with larger step beyond 3rd point.\n")
            s *= stepScale
            Eb = Ec
            Ec = 0

        else:
            logger.debug("\tSearching with smaller step between first 2 points.\n")
            s *= 0.5
            Ec = Eb
            Eb = 0

    # get norm |q| and unit vector in the step direction
    ls_dqnorm = sqrt(np.dot(dq, dq))
    ls_u = dq.copy() / ls_dqnorm

    # get gradient and hessian in step direction
    ls_g = -1 * np.dot(fq, ls_u)  # should be unchanged
    ls_h = np.dot(ls_u, np.dot(H, ls_u))

    if op.Params.print_lvl > 1:
        logger.info('\n\\t|target step|: %15.10f\n' % ls_dqnorm)
        logger.info('\tLS_gradient     : %15.10f\n' % ls_g)
        logger.info('\tLS_hessian      : %15.10f\n' % ls_h)

    DEprojected = DE_projected('NR', ls_dqnorm, ls_g, ls_h)
    logger.info(
        "\tProjected quadratic energy change using full Hessian: %15.10f\n" % DEprojected)

    # Scale fq into aJ for printing
    # fq_aJ = qShowForces(oMolsys.intcos, fq)
    # displace(oMolsys._fragments[0].intcos, oMolsys._fragments[0].geom, dq, fq_aJ)

    dq_actual = sqrt(np.dot(dq, dq))
    logger.info("\tNorm of achieved step-size %15.10f\n" % dq_actual)

    # Symmetrize the geometry for next step
    # symmetrize_geom()

    # save values in step data
    oHistory.appendRecord(DEprojected, dq, ls_u, ls_g, ls_h)

    # Can check full geometry, but returned indices will correspond then to that.
    linearList = linearBendCheck(oMolsys.intcos, oMolsys.geom, dq)
    if linearList:
        raise optExceptions.AlgFail("New linear angles", newLinearBends=linearList)

    oHistory.nuclear_repulsion_energy = nuc

    return dq

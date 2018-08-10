from math import sqrt, fabs
import numpy as np
import logging

import displace
import intcosMisc
from linearAlgebra import symmMatEig, symmMatInv, symmMatRoot
from printTools import printArrayString, printMatString


def takeHessianHalfStep(oMolsys, Hq, B, fq, s, direction='forward'):
    """ Takes a 'half step' from starting geometry along the gradient,
    then takes an additional 'half step' as a guess
    Returns: dq (change in internal coordinates (vector))
    """
    logger = logging.getLogger(__name__)
    IRC_starting = (
        "==================================================================================\n")
    IRC_starting += ("Taking Hessian IRC HalfStep and Guess Step\n")
    # Calculate G, G^-1/2 and G-1/2
    Gm = intcosMisc.Gmat(oMolsys.intcos, oMolsys.geom, oMolsys.masses)
    GmInv = symmMatInv(Gm)
    GmRoot = symmMatRoot(Gm)
    GmRootInv = symmMatInv(GmRoot)

    # PrintStartingMatrices
    logger.debug("B matrix\n" + printMatString(B))
    logger.debug("Mass Weighted G Root Matrix\n" + printMatString(GmRoot))
    logger.debug("Hesian in Internals\n" + printMatString(Hq))
    # Hq = intcosMisc.convertHessianToInternals(H, oMolsys.intcos, oMolsys.geom)
    HEigVals, HEigVects = symmMatEig(Hq)
    # get gradient from Psi4 in cartesian geom
    # gx = fgradient(geom)
    # gq = np.dot(GmInv, np.dot(B, gx))

    # initial internal coordinates from .intcosMisc
    qZero = intcosMisc.qValues(oMolsys.intcos, oMolsys.geom)

    # symmMatEig returns the Eigen Vectors as rows in order of increasing eigen values
    # first step from TS will be along the smallest eigenvector
    gk = np.zeros(len(HEigVects[0]), float)

    for col in range(len(HEigVects[0])):
        gk[col] = HEigVects[0, col]

    if direction == 'backward':
        for i in range(len(gk)):
            gk[i] = -1 * gk[i]

    # To solve N = (gk^t * G * gk)
    N = np.dot(gk.T, np.dot(Gm, gk))
    N = 1 / sqrt(N)

    # g*k+1 = qk - 1/2 * s * (N*G*gk)
    # N*G*gk
    dqPivot = np.dot(N, np.dot(Gm, gk))

    # applying weight 1/2 s to product (N*G*gk)
    # then applies vector addition q -  1/2 * s* (N*G*gk)
    dqPivot = np.dot(0.5, np.dot(s, dqPivot))
    qPivot = np.add(qZero, dqPivot)
    # To-Do add cartesian coordinate for pivot point
    # qPrime = np.dot (N, np.dot(G, gQ))
    dqPrime = np.dot(2, dqPivot)
    qPrime = np.add(dqPrime, qZero)
    displaceIRCStep(oMolsys, dqPrime, Hq, fq)

    logger.info("next geometry\n" + printArrayString(qPrime))
    logger.info("Dq to Pivot Point\n" + printArrayString(dqPivot))
    logger.info("Dq to Guess Point\n" + printArrayString(dqPrime) +
                "\n================================================================="
                + "==================\n")

    return dqPivot, qPivot, qPrime


def takeGradientHalfStep(oMolsys, H, B, s, gX):
    """
    Takes a half step from starting geometry along the gradient, then takes an additional
    half step as a guess
    Returns dq: displacement in internals as a numpy array
    """
    # Calculate G, G^-1/2 and G-1/2
    Gm = intcosMisc.Gmat(oMolsys.intcos, oMolsys.geom, oMolsys.masses)
    GmInv = symmMatInv(Gm)
    GmRoot = symmMatRoot(Gm)
    GmRootInv = symmMatInv(GmRoot)

    qZero = intcosMisc.qValues(oMolsys.intcos, oMolsys.geom)

    # convert gradient to Internals
    gQ = np.dot(GmInv, np.dot(B, gX))

    # To solve N = (gk^t * G * gk)
    N = symmMatRoot(np.dot(gQ.T, np.dot(Gm, gQ)), True)

    # g*k+1 = qk - 1/2 * s * (N*G*gk)
    # N*G*gk
    qPivot = np.dot(N, np.dot(G, gQ))

    # applying weight 1/2 s to product (N*G*gk)
    # then applies vector addition q -  1/2 * s* (N*G*gk)
    for i in range(len(gQ)):
        qPivot[i] = 0.5 * s * qPivot[i]
        qPivot = np.subtract(qZero, qPivot)

    # displaceIRCStep(oMolsys.intcos, oMolsys.geom, np.subtract(qPivot, qZero), H, g)

    qPrime = np.dot(N, np.dot(G, gQ))
    for i in range(len(gQ)):
        qPrime[i] = 0.5 * s * qPrime[i]
        qPrime[i] = qPivot[i] - qPrime[i]

    dq = np.subtract(qPrime, qZero)
    dq = sqrt(np.dot(dq, dq))

    return qPivot, qPrime, Dq


def Dq(oMolsys, g, E, Hq, B, s, qPrime, dqPrime):
    """ Before Dq_IRC is called, the goemetry must be updated to the guess point
    Returns Dq from qk+1 to gprime.
    """
    logger = logging.getLogger(__name__)
    IRC_search_start = (
        "==============================================================================="
        + "=======================\n")
    IRC_search_start += ("Starting IRC constrained optimization\n")
    logger.info(IRC_search_start)

    GPrime = intcosMisc.Gmat(oMolsys.intcos, oMolsys.geom, oMolsys.masses)
    GPrimeInv = symmMatInv(GPrime)
    GPrimeRoot = symmMatRoot(GPrime)
    GPrimeRootInv = symmMatRoot(GPrime, True)
    # Hq = intcosMisc.convertHessianToInternals(Hq, oMolsys.intcos, oMolsys.geom)
    # Hq = intcosMisc.convertHessianToInternals(H, oMolsys.intcos, oMolsys.geom)
    # vectors nessecary to solve for Lambda, naming is taken from Gonzalez and Schlegel
    deltaQM = 0
    pPrime = dqPrime
    # print_opt ("G prime root matrix")
    # printMat (GPrimeRoot)
    # print_opt ("gradient")
    # printArray (g)
    # print_opt ("Hessian in Internals")
    # printMat (Hq)

    u = np.identity(oMolsys.Natom * 3)
    logger.info("Cartesian Gradient\n" + printArrayString(g))
    g = np.dot(GPrimeInv, np.dot(B, np.dot(u, g)))
    logger.info("Internal Gradient\n" + printArrayString(g))
    gM = np.dot(GPrimeRoot, g)
    # print ("gM")
    # print (gM)
    HM = np.dot(GPrimeRoot, np.dot(Hq, GPrimeRoot))
    pM = np.dot(GPrimeRootInv, pPrime)
    HMEigValues, HMEigVects = symmMatEig(HM)
    # Variables for solving lagrangian function
    lower_b_lagrangian = -100
    upper_b_lagrangian = 100
    lower_b_lambda = 0.5 * HMEigValues[0]
    upper_b_lambda = 0.5 * HMEigValues[0]
    Lambda = 0.5 * HMEigValues[0]
    prev_lambda = Lambda
    prev_lagrangian = 1
    lagrangian = 1

    # Solves F(L) = Sum[(bj*pj - gj)/(bj-L)]^2 - 1/2s = 0
    # coarse search
    lagIter = 0
    while (prev_lagrangian * lagrangian > 0) and lagIter < 1000:
        prev_lagrangian = lagrangian
        Lambda -= 1
        lagrangian = calc_lagrangian(Lambda, HMEigValues, HMEigVects, gM, pM, s)
        if lagrangian < 0 and fabs(lagrangian) < fabs(lower_b_lagrangian):
            lower_b_lagrangian = lagrangian
            lower_b_lambda = Lambda

        if lagrangian > 0 and fabs(lagrangian) < fabs(upper_b_lagrangian):
            upper_b_lagrangian = lagrangian
            upper_b_lambda = Lambda
        lagIter += 1
        Lambda -= 1

    # fine search
    # calulates next lambda using Householder method

    d_lagrangian = np.array([2, 6, 24, 120])
    # array of lagrangian derivatives to solve Householder method with weights to solve derivative
    lagIter = 0

    while lagrangian - prev_lagrangian > 10**-16:
        prev_lagrangian = lagrangian
        for i in range(4):
            d_lagrangian[i] *= calc_lagrangian(Lambda, HMEigValues, HMEigVects, gM, pM, s)

        h_f = -lagrangian / d_lagrangian[1]

        if lagrangian < 0 and (fabs(lagrangian) < fabs(lower_b_lagrangian)):
            lower_b_lagrangian = lagrangian
            lower_b_lambda = Lambda
        elif lagrangian > 0 and fabs(lagrangian) < fabs(upper_b_lagrangian):
            upper_b_lagrangian = lagrangian
            upper_b_lambda = Lambda

        elif lagrangian * prev_lagrangian < 0:
            lagrangian = (lagrangian + prev_lagrangian) / 2
            # Lagrangian found
        else:
            prev_lambda = Lambda
            Lambda += h_f * (24 * d_lagrangian[0] * 24 * d_lagrangian[1] * 8 * h_f +
                             4 * d_lagrangian[2] * 8 * h_f**2) / (
                                 24 * d_lagrangian[0] + 36 * h_f * d_lagrangian[1] + 6 *
                                 (d_lagrangian[1]**2 / d_lagrangian[0]) * 8 * h_f**2 +
                                 8 * d_lagrangian[2] * h_f**2 + d_lagrangian[3] * h_f**3)
        lagIter += 1

        if lagIter > 50:
            prev_lambda = Lambda
            Lambda = (lower_b_lambda + upper_b_lambda) / 2  # check this later. these may not
            # be the correct variables

        if lagIter > 200:
            print("Exception should have been thrown")
            # needs to throw failure to converge exception

    # constructing Lambda * intcosxintcosI
    LambdaI = np.zeros((len(g), len(g)), float)

    for i in range(len(g)):
        LambdaI[i][i] = 1 * Lambda

    deltaQM = symmMatInv(-np.subtract(HM, LambdaI))
    deltaQM = np.dot(deltaQM, np.subtract(gM, np.multiply(Lambda, pM)))
    logger.info("initial geometry\n" + printArrayString(qPrime))
    dq = np.dot(GPrimeRoot, deltaQM)
    logger.info("dq to next geometry\n" + printArrayString(dq))
    displaceIRCStep(oMolsys, dq, Hq, np.dot(-1, g))
    qNew = np.add(qPrime, dq)
    logger.info("New internal coordinates\n" + printArrayString(qNew))

    IRC_search_ending = (
            "Constrained optimization finished\n"
            + "==============================================================="
            + "=======================================\n")

    # save values in step data
    # History.appendRecord(DEprojected, dq, ircU, ircG, ircH)
    logger.info(IRC_search_ending)
    return dq


# calculates Lagrangian function of Lambda
# returns value of lagrangian function
def calc_lagrangian(Lambda, HMEigValues, HMEigVects, gM, pM, s):
    lagrangian = 0
    for i in range(len(HMEigValues)):
        numerator = (HMEigValues[i] * np.dot(pM.T, HMEigVects[i])) - (np.dot(
            gM.T, HMEigVects[i]))
        denominator = HMEigValues[i] - Lambda
        lagrangian += numerator / denominator
    lagrangian *= lagrangian
    lagrangian -= (0.5 * s)**2

    return lagrangian


# displaces an atom with the dq from the IRC data
# returns void
def displaceIRCStep(oMolsys, dq, H, fq):
    logger = logging.getLogger(__name__)
    # get norm |q| and unit vector in the step direction
    ircDqNorm = sqrt(np.dot(dq, dq))
    ircU = dq / ircDqNorm
    logger.info("\tNorm of target step-size %15.10f\n" % ircDqNorm)

    # get gradient and hessian in step direction
    ircG = np.dot(-1, np.dot(fq, ircU))  # gradient, not force
    ircH = np.dot(ircU, np.dot(H, ircU))

    # if op.Params.print_lvl > 1:
    # print_opt('\t|IRC target step|: %15.10f\n' % ircDqNorm)
    # print_opt('\tIRC gradient     : %15.10f\n' % ircG)
    # print_opt('\tIRC hessian      : %15.10f\n' % ircH)
    # DEprojected = stepAlgorithms.DE_projected('IRC', ircDqNorm, ircG, ircH)
    # print_opt("\tProjected energy change by quadratic approximation: %20.10lf\n" % DEprojected)

    # Scale fq into aJ for printing
    fq_aJ = intcosMisc.qShowForces(oMolsys.intcos, fq)
    # print ("------------------------------")
    # print (oMolsys._fragments[0].intcos)
    # print (oMolsys._fragments[0].geom)
    # print (dq)
    # print (fq_aJ)
    # print ("--------------------------------")
    displace(oMolsys._fragments[0].intcos, oMolsys._fragments[0].geom, dq, fq_aJ)

    dq_actual = sqrt(np.dot(dq, dq))
    logger.info("\tNorm of achieved step-size %15.10f\n" % dq_actual)
    # Symmetrize the geometry for next step

from math import sqrt, fabs
import numpy as np
import logging

from . import optparams as op
from . import intcosMisc
from . import stepAlgorithms
from .displace import displace
from .history import oHistory
from .linearAlgebra import symmMatEig, symmMatInv, symmMatRoot
from .printTools import printArrayString, printMatString


def take_half_step(oMolsys, HqM, fq, s, gX, initial=False, direction='forward'):
    """ Takes a 'half step' from starting geometry along the gradient,
    then takes an additional 'half step' as a guess
    
    Parameters
    ----------
    oMolsys : class
    Hq : ndarray
        Mass Weighted Hessian in Internal Coordinates
    B : ndarray
        B matrix
    fq : ndarray
        forces in internal coordinates
    s : float

    Returns
    -------

    qGuess


    Returns: qPivot
    """
    import scipy
    #FOR TEST PURPOSES
    direction = 'backward'
    logger = logging.getLogger(__name__)
    IRC_starting = (
        "==================================================================================\n")
    IRC_starting += ("Taking Hessian IRC HalfStep and Guess Step\n")
    # Calculate G, G^-1/2
    B = intcosMisc.Bmat(oMolsys.intcos, oMolsys.geom)
    Bm = intcosMisc.inv_mass_weighted(B, oMolsys.intcos, oMolsys.masses)
    Gm = intcosMisc.Gmat_B(Bm, oMolsys.intcos)
    GmInv = symmMatInv(Gm)
    GmRoot = scipy.linalg.sqrtm(Gm)
    GmRootInv = symmMatInv(GmRoot)

    # PrintStartingMatrices
    #if op.Params.print_lvl >= 4:
        # logger.debug("B matrix\n" + printMatString(B))

    #TODO this should be the mass weighted hessian
    H_eig_vals, H_eig_vects = symmMatEig(HqM) 

    # initial internal coordinates from .intcosMisc
    qZero = intcosMisc.qValues(oMolsys.intcos, oMolsys.geom)

    # symmMatEig returns the Eigen Vectors as rows in order of increasing eigenvalues
    # first step from TS will be along the smallest eigenvector
    #gk = np.zeros(len(HEigVects[0]), float)


    if initial:
        gk = np.copy(H_eig_vects[0])
        logger.debug("Smallest EigenVector of Hessian" + printArrayString(gk))

    #for col in range(len(HEigVects[0])):
    #    gk[col] = HEigVects[0, col]

        if direction == 'backward': #This should be called iff initial=True
            gk = np.multiply(-1, gk)
    else:
        gk = intcosMisc.qForces(oMolsys.intcos, oMolsys.geom, gX, B)

    # To solve N = (gk^t * G * gk)
    N = 1 / sqrt(np.dot(gk.T, np.dot(Gm, gk)))

    # g*k+1 = qk - 1/2 * s * (N*G*gk)
    # N*G*gk
    normalized_vector = np.dot(N, np.dot(Gm, gk))

    # applying weight 1/2 s to product (N*G*gk)
    # then applies vector addition q -  1/2 * s* (N*G*gk)
    dqPivot = np.multiply(0.5, np.multiply(s, normalized_vector))
    dqGuess = np.multiply(2,dqPivot)
    qPivot = np.subtract(qZero, dqPivot)
    qGuess = np.subtract(qPivot, dqGuess)
    # TODO add cartesian coordinate for pivot point

    logger.info("Dq to Pivot Point\n" + printArrayString(dqPivot))
    logger.info("Dq to Guess Point\n" + printArrayString(dqGuess) +
                "\n================================================================="
                + "==================\n")
    
    displaceIRCStep(oMolsys, dqGuess, HqM, fq, ensure_convergence=True)

    return [dqPivot, qPivot, dqGuess, qGuess]


"""

def takeGradientHalfStep(oMolsys, H, B, s, gX):
    Takes a half step from starting geometry along the gradient, then takes an additional
    half step as a guess
    Returns dq: displacement in internals as a numpy array

    # Calculate G, G^-1/2 and G-1/2
    G = intcosMisc.Gmat(oMolsys.intcos, oMolsys.geom, oMolsys.masses)
    GInv = symmMatInv(G)
    GRoot = symmMatRoot(G)
    GRootInv = symmMatInv(GRoot)

    qZero = intcosMisc.qValues(oMolsys.intcos, oMolsys.geom)

    # convert gradient to Internals
    gQ = np.dot(GInv, np.dot(B, gX))

    # To solve N = (gk^t * G * gk)^-1/2
    N = scipy.linalg.sqrtm(symmMatInv(np.dot(gQ.T, np.dot(G, gQ)), True))
    
    # g*k+1 = qk - 1/2 * s * (N*G*gk)
    # N*G*gk
    # q*k+1 = qk -  1/2 * s* (N*G*gk)
    qPivot = np.subtract(qZero, (np.multiply(0.5, np.dot(np.dot(N, G), gQ))))
    # applying weight 1/2 s to product (N*G*gk)
    # then applies vector addition q -  1/2 * s* (N*G*gk)
        
    dqPivot = np.subtract(qPivot, qZero)
    
    displaceIRCStep(oMolsys.intcos, oMolsys.geom,dqPivot, H, g, ensure_convergence=True)

    qPrime = np.dot(N, np.dot(G, gQ)) #duplication of above why are we looping through this?
    for i in range(len(gQ)):
        qPrime[i] = 0.5 * s * qPrime[i]
        qPrime[i] = qPivot[i] - qPrime[i]

    dq = np.subtract(qPrime, qZero)
    dq = sqrt(np.dot(dq, dq))

    return dqPivot, qPrime, dq

"""

def Dq(oMolsys, g, E, H_q, B, s, dqGuess):
    """ Before Dq_IRC is called, the goemetry must be updated to the guess point
    Returns Dq from qk+1 to gprime.
    """
    import scipy
    logger = logging.getLogger(__name__)
    logger.debug("Starting IRC constrained optimization\n")

    B_m = intcosMisc.inv_mass_weighted(B, oMolsys.intcos, oMolsys.masses)
    G_prime = intcosMisc.Gmat_B(B_m, oMolsys.intcos)
    logger.debug("Mass weighted Gmatrix at hypersphere point: " + printMatString(G_prime))
    G_prime_inv = symmMatInv(G_prime)
    G_prime_root = scipy.linalg.sqrtm(G_prime)
    G_prime_root_inv = scipy.linalg.sqrtm(G_prime_inv)

    # Hq = intcosMisc.convertHessianToInternals(Hq, oMolsys.intcos, oMolsys.geom)
    # Hq = intcosMisc.convertHessianToInternals(H, oMolsys.intcos, oMolsys.geom)
    # vectors nessecary to solve for Lambda, naming is taken from Gonzalez and Schlegel
    deltaQM = 0
    p_prime = dqGuess
    logger.debug("G prime root matrix: " + printMatString(G_prime_root))
    # print_opt ("gradient")
    # printArray (g)
    logger.debug("Hessian in Internals: " + printMatString(H_q))

    g_q = intcosMisc.qForces(oMolsys.intcos, oMolsys.geom, g, B)
    g_m = np.dot(G_prime_root, g_q) 
    logger.info("Internal Gradient\n" + printArrayString(g_q))
    logger.debug("g_m: " + printArrayString(g_m))

    H_m = intcosMisc.mass_weight_hessian_internals(H_q, B, oMolsys.intcos, oMolsys.masses)
    pM = np.dot(G_prime_root_inv, p_prime)
    logger.debug("H_m: " + printMatString(H_m))
    logger.debug("pM: " + printArrayString(pM))
    HMEigValues, HMEigVects = symmMatEig(H_m)
    # Variables for solving lagrangian function
    lower_b_lagrangian = -100
    upper_b_lagrangian = 100
    lower_b_lambda = 0.5 * HMEigValues[0]
    upper_b_lambda = 0.5 * HMEigValues[0]
    Lambda = lower_b_lambda
    prev_lambda = Lambda
    prev_lagrangian = 1
    lagrangian = 1

    # Solves F(L) = Sum[(bj*pj - gj)/(bj-L)]^2 - 1/2s = 0
    # coarse search
    lagIter = 0
    while (prev_lagrangian * lagrangian > 0) and lagIter < 1000:
        prev_lagrangian = lagrangian
        Lambda -= 1
        lagrangian = calc_lagrangian(Lambda, HMEigValues, HMEigVects, g_m, pM, s)
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
            d_lagrangian[i] *= calc_lagrangian(Lambda, HMEigValues, HMEigVects, g_m, pM, s)

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
            # TODO needs to throw failure to converge exception

    # constructing Lambda * intcosxintcosI
    LambdaI = np.identity(len(oMolsys.intcos))

    deltaQM = symmMatInv(-np.subtract(H_m, LambdaI))
    deltaQM = np.dot(deltaQM, np.subtract(g_m, np.multiply(Lambda, pM)))
    #logger.info("initial geometry\n" + printArrayString(qPrime))
    dq = np.dot(G_prime_root, deltaQM)
    logger.info("dq to next geometry\n" + printArrayString(dq))
    displaceIRCStep(oMolsys, dq, H_q, np.multiply(-1, g_q))
    q_new = np.add(dqGuess, dq)
    logger.info("New internal coordinates\n" + printArrayString(q_new))

    # save values in step data
    logger.info("IRC Constrained optimization finished\n")
    return dq


# calculates Lagrangian function of Lambda
# returns value of lagrangian function
def calc_lagrangian(Lambda, HMEigValues, HMEigVects, gM, pM, s):
    lagrangian = 0
    numerator = HMEigValues
    for i in range(len(HMEigValues)):
        numerator = np.subtract(np.multiply(HMEigValues[i], np.dot(pM.T, HMEigVects[i])), np.dot(
            gM.T, HMEigVects[i]))
        denominator = HMEigValues[i] - Lambda
        lagrangian += (numerator / denominator) **2
    lagrangian -= (0.5 * s)**2

    return lagrangian


# displaces an atom with the dq from the IRC data
# returns void
def displaceIRCStep(oMolsys, dq, H, fq, ensure_convergence=False):
    logger = logging.getLogger(__name__)
    logger.info("Displacing IRC Step")
    # get norm |q| and unit vector in the step direction
    ircDqNorm = sqrt(np.dot(dq, dq))
    ircU = np.divide(dq, ircDqNorm)
    logger.info("\tNorm of target step-size %15.10f\n" % ircDqNorm)

    # get gradient and hessian in step direction
    ircG = np.dot(-1, np.dot(fq, ircU))  # gradient, not force
    ircH = np.dot(ircU, np.dot(H, ircU))

    # if op.Params.print_lvl > 1:
    logger.debug('\n\t|IRC target step|: %15.10f\n' % ircDqNorm)
    logger.debug('\n\tIRC gradient     : %15.10f\n' % ircG)
    logger.debug('\n\tIRC hessian      : %15.10f\n' % ircH)
    DEprojected = stepAlgorithms.DE_projected('RFO', ircDqNorm, ircG, ircH)
    logger.debug("Projected energy change by quadratic approximation: %20.10lf\n" % DEprojected)

    # Scale fq into aJ for printing
    fq_aJ = intcosMisc.qShowForces(oMolsys.intcos, fq)
    # print ("------------------------------")
    # print (oMolsys._fragments[0].intcos)
    # print (oMolsys._fragments[0].geom)
    # print (dq)
    # print (fq_aJ)
    # print ("--------------------------------")
    displace(oMolsys._fragments[0].intcos, oMolsys._fragments[0].geom, dq, fq_aJ, ensure_convergence)

    oHistory.appendRecord(DEprojected, dq, ircU, ircG, ircH)

    dq_actual = sqrt(np.dot(dq, dq))
    logger.info("\tNorm of achieved step-size %15.10f\n" % dq_actual)

    oHistory.appendRecord(DEprojected, dq, ircU, ircG, ircH)

    # Symmetrize the geometry for next step

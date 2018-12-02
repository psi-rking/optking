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
    #FOR TEST PURPOSES
    direction = 'backward'
    logger = logging.getLogger(__name__)
    logger.info("Taking Hessian IRC HalfStep and Guess Step\n")
    # Calculate G, G^-1/2
    B = intcosMisc.Bmat(oMolsys.intcos, oMolsys.geom)
    Bm = intcosMisc.inv_mass_weighted(B, oMolsys.intcos, oMolsys.masses)
    Gm = intcosMisc.Gmat_B(Bm, oMolsys.intcos)
    GmInv = symmMatInv(Gm)
    GmRoot = symmMatRoot(Gm)
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
    qPivot = np.subtract(qZero, dqPivot)
    qGuess = np.subtract(qPivot, 2 * dqPivot)
    # TODO add cartesian coordinate for pivot point

    logger.info("Dq to Pivot Point\n" + printArrayString(dqPivot))
    logger.info("Dq to Guess Point\n" + printArrayString(2 * dqPivot))
    
    displaceIRCStep(oMolsys, 2 * dqPivot, HqM, fq, ensure_convergence=True)

    return [dqPivot, qPivot, qGuess]


# def takeGradientHalfStep(oMolsys, H, B, s, gX):
#     Takes a half step from starting geometry along the gradient, then takes an additional
#     half step as a guess
#     Returns dq: displacement in internals as a numpy array
# 
#     # Calculate G, G^-1/2 and G-1/2
#     G = intcosMisc.Gmat(oMolsys.intcos, oMolsys.geom, oMolsys.masses)
#     GInv = symmMatInv(G)
#     GRoot = symmMatRoot(G)
#     GRootInv = symmMatInv(GRoot)
# 
#     qZero = intcosMisc.qValues(oMolsys.intcos, oMolsys.geom)
# 
#     # convert gradient to Internals
#     gQ = np.dot(GInv, np.dot(B, gX))
# 
#     # To solve N = (gk^t * G * gk)^-1/2
#     N = symmMatRoot(symmMatInv(np.dot(gQ.T, np.dot(G, gQ)), True))
#     
#     # g*k+1 = qk - 1/2 * s * (N*G*gk)
#     # N*G*gk
#     # q*k+1 = qk -  1/2 * s* (N*G*gk)
#     qPivot = np.subtract(qZero, (np.multiply(0.5, np.dot(np.dot(N, G), gQ))))
#     # applying weight 1/2 s to product (N*G*gk)
#     # then applies vector addition q -  1/2 * s* (N*G*gk)
#         
#     dqPivot = np.subtract(qPivot, qZero)
#     
#     displaceIRCStep(oMolsys.intcos, oMolsys.geom,dqPivot, H, g, ensure_convergence=True)
# 
#     qPrime = np.dot(N, np.dot(G, gQ)) #duplication of above why are we looping through this?
#     for i in range(len(gQ)):
#         qPrime[i] = 0.5 * s * qPrime[i]
#         qPrime[i] = qPivot[i] - qPrime[i]
# 
#     dq = np.subtract(qPrime, qZero)
#     dq = sqrt(np.dot(dq, dq))
# 
#     return dqPivot, qPrime, dq


def Dq(oMolsys, g, E, H_q, B, s, dqGuess):
    """ Before Dq_IRC is called, the goemetry must be updated to the guess point
    Returns Dq from qk+1 to gprime.
    """
    
    logger = logging.getLogger(__name__)
    logger.debug("Starting IRC constrained optimization\n")

    B_m = intcosMisc.inv_mass_weighted(B, oMolsys.intcos, oMolsys.masses)
    G_prime = intcosMisc.Gmat_B(B_m, oMolsys.intcos)
    logger.debug("Mass weighted Gmatrix at hypersphere point: " + printMatString(G_prime))
    G_prime_inv = symmMatInv(G_prime)
    G_prime_root = symmMatRoot(G_prime)
    G_prime_root_inv = symmMatRoot(G_prime_inv)

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
    lower_b_lambda = 0
    upper_b_lambda = 0
    Lambda = 0.5 * HMEigValues[0]
    prev_lambda = -999

    # Solves F(L) = Sum[(bj*pj - gj)/(bj-L)]^2 - 1/2s = 0
    # coarse search

    lagrangian = calc_lagrangian(Lambda, HMEigValues, HMEigVects, g_m, pM, s, 0)
    prev_lagrangian = lagrangian

    lagIter = 0
    while (prev_lagrangian * lagrangian > 0) and lagIter < 1000:
        prev_lagrangian = lagrangian
        lagrangian = calc_lagrangian(Lambda, HMEigValues, HMEigVects, g_m, pM, s, 0)
        logger.debug("lagrangian: " + str(lagrangian))
        if lagrangian < 0 and abs(lagrangian) < abs(lower_b_lagrangian):
            lower_b_lagrangian = lagrangian
            lower_b_lambda = Lambda

        if lagrangian > 0 and abs(lagrangian) < abs(upper_b_lagrangian):
            upper_b_lagrangian = lagrangian
            upper_b_lambda = Lambda
        lagIter += 1
        prev_lambda = Lambda
        Lambda -= 1

    # fine search
    # calulates next lambda using Householder method

    d_lagrangian = np.array([1.0, 2.0, 6.0, 24.0, 120.0])
    # will contain an array of lagrangian derivatives to solve Householder method
    # starts with weights to solve derivative
    
    lagIter = 0

    while abs(Lambda - prev_lambda) > 10**-16:
        for lag_order in range(5):
            d_lagrangian[lag_order] *= calc_lagrangian(Lambda, HMEigValues, HMEigVects, g_m, pM, 
                                                       s, lag_order)
        prev_lagrangian = lagrangian
        lagrangian = d_lagrangian[0]
        
        h_f = -1 * d_lagrangian[0] / d_lagrangian[1]

        if lagrangian < 0 and (abs(lagrangian) < abs(lower_b_lagrangian)):
            lower_b_lagrangian = lagrangian
            lower_b_lambda = Lambda
        elif lagrangian > 0 and abs(lagrangian) < abs(upper_b_lagrangian):
            upper_b_lagrangian = lagrangian
            upper_b_lambda = Lambda
        elif lagrangian * prev_lagrangian < 0:
            current_lambda = Lambda
            Lambda = (prev_lambda + Lambda ) / 2
            prev_lambda = current_lambda
        else:
            prev_lambda = Lambda
            Lambda += (h_f * (24 * d_lagrangian[1] * 24 * d_lagrangian[2] * h_f +
                              4 * d_lagrangian[3] * h_f**2)) / (
                              24 * d_lagrangian[1] + 36 * h_f * d_lagrangian[2] +
                              6 * (d_lagrangian[2]**2 * h_f**2 / d_lagrangian[1]) +
                              8 * d_lagrangian[3] * h_f**2 + d_lagrangian[4] * h_f**3)

        lagIter += 1
        if lagIter > 50:
            prev_lambda = Lambda
            Lambda = (lower_b_lambda + upper_b_lambda) / 2 #Try a bisection after 50 attempts

        if lagIter > 200:
            logger.warning("Exception should have been thrown")
            # TODO needs to throw failure to converge exception

    # constructing Lambda * intcosxintcos
    LambdaI = np.identity(len(oMolsys.intcos))
    LambdaI = np.multiply(Lambda, LambdaI)
    deltaQM = np.multiply(-1, symmMatInv(np.subtract(H_m, LambdaI)))
    deltaQM = np.dot(deltaQM, np.subtract(np.multiply(Lambda, pM), g_m))
    #logger.info("initial geometry\n" + printArrayString(qPrime))
    dq = np.dot(G_prime_root, deltaQM)
    logger.info("dq to next geometry\n" + printArrayString(dq))
    displaceIRCStep(oMolsys, dq, H_q, np.multiply(-1, g_q))

    # save values in step data
    logger.info("IRC Constrained optimization finished\n")
    return dq


# calculates Lagrangian function of Lambda
# returns value of lagrangian function
def calc_lagrangian(Lambda, HMEigValues, HMEigVects, gM, pM, s, order):
    lagrangian = 0
    numerator = HMEigValues
    for i in range(len(HMEigValues)):
        numerator = np.subtract(np.multiply(HMEigValues[i], np.dot(pM.T, HMEigVects[i])), np.dot(
                    gM.T, HMEigVects[i]))
        denominator = HMEigValues[i] - Lambda
        lagrangian += (numerator / denominator) ** 2 / (denominator**order) 
    lagrangian -= (0.5 * s)**2
    logger = logging.getLogger(__name__)
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
    irc_displace_info = ('\n\t|IRC target step|: %15.10f\n'
                         '\tIRC gradient     : %15.10f\n' 
                         '\tIRC hessian      : %15.10f\n' % ( ircDqNorm, ircG, ircH))
    logger.debug(irc_displace_info)
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
    logger.info("Norm of achieved step-size %15.10f\n" % dq_actual)

    oHistory.appendRecord(DEprojected, dq, ircU, ircG, ircH)

    # Symmetrize the geometry for next step

from math import sqrt, fabs, acos, tan
import numpy as np
import logging

from . import optparams as op
from . import intcosMisc
from . import stepAlgorithms
from . import IRCdata
from .displace import displaceMolsys
from .history import oHistory
from .linearAlgebra import symmMatEig, symmMatInv, symmMatRoot
from .printTools import printArrayString, printMatString
from .exceptions import AlgError

def step_N_factor(G, g):
    """ Computes distance scaling factor for mass-weighted internals. """
    return 1.0 / sqrt(np.dot(g.T, np.dot(G, g)))

def IRC_DE_projected(step_size, grad, hess):
    """ Compute anticipated energy change along one dimension """
    return (step_size * grad + 0.5 * step_size * step_size * hess)

def computePivotAndGuessPoints(oMolsys, v, IRCstepSize):
    """ Takes a half step along v to the 'pivot point', then 
    an additional half step as first guess in constrained opt.
    
    Parameters
    ----------
    oMolsys : class
    v : ndarray
        vector to step along
    step_size : float
        step size

    Returns
    -------

    """
    logger = logging.getLogger(__name__)

    # Compute and save pivot point
    G = oMolsys.Gmat(oMolsys.masses)
    N = step_N_factor(G, v)
    dq_pivot = 0.5 * N * IRCstepSize * np.dot(G, v)
    logger.debug("\n Dq to Pivot Point:" + printArrayString(dq_pivot))

    #x_pivot = oMolsys.geom # starting geom but becomes pivot point on next line
    #displace(oMolsys.intcos, x_pivot, dq_pivot, ensure_convergence=True)

    displaceMolsys(oMolsys, dq_pivot, ensure_convergence=True)
    x_pivot = oMolsys.geom
    q_pivot = oMolsys.qArray()
    IRCdata.history.add_pivot_point(q_pivot, x_pivot)

    # Step again to get initial guess for next step.  Leave geometry in oMolsys.
    logger.info("Computing Dq to First Guess Point")
    logger.debug(printArrayString(dq_pivot))
    x_guess = x_pivot.copy()
    #displace(oMolsys.intcos, x_guess, dq_pivot, ensure_convergence=True)
    displaceMolsys(oMolsys, dq_pivot, ensure_convergence=True)
    oMolsys.geom = x_guess


def Dq_IRC(oMolsys, E, f_q, H_q, s, dqGuess):
    """ Before Dq_IRC is called, the geometry must be updated to the guess point
    Returns Dq from qk+1 to gprime.
    TODO: What is dqGuess for?  Remove it?
    """
    
    logger = logging.getLogger(__name__)
    logger.debug("Starting IRC constrained optimization\n")

    G_prime = oMolsys.Gmat(oMolsys.masses)
    logger.debug("Mass-weighted Gmatrix at hypersphere point: \n" + printMatString(G_prime))
    G_prime_root = symmMatRoot(G_prime)
    G_prime_inv = symmMatInv(G_prime)
    G_prime_root_inv = symmMatRoot(G_prime_inv)

    logger.debug("G prime root matrix: \n" + printMatString(G_prime_root))

    deltaQM = 0
    g_M = np.dot(G_prime_root, - f_q) 
    logger.debug("g_M: \n" + printArrayString(g_M))

    H_M = np.dot(np.dot(G_prime_root, H_q), G_prime_root.T)
    logger.debug("H_M: \n" + printMatString(H_M))

    # Compute p_prime, difference from pivot point
    orig_geom = oMolsys.geom
    oMolsys.geom = IRCdata.history.x_pivot()
    q_pivot = oMolsys.qArray()
    oMolsys.geom = orig_geom
    p_prime = oMolsys.qArray() - q_pivot

    #p_prime = intcosMisc.qValues(oMolsys.intcos, oMolsys.geom) -  \
    #          intcosMisc.qValues(oMolsys.intcos, IRCdata.history.x_pivot())
    p_M = np.dot(G_prime_root_inv, p_prime)
    logger.debug("p_M: \n" + printArrayString(p_M))

    HMEigValues, HMEigVects = symmMatEig(H_M)
    logger.debug("HMEigValues: \n" + printArrayString(HMEigValues))
    logger.debug("HMEigVects: \n" + printMatString(HMEigVects))

    # Variables for solving lagrangian function
    lb_lagrangian = -100
    up_lagrangian = 100
    lb_lambda = 0
    if HMEigValues[0] < 0: # Make lower than the lowest eval
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

    lagrangian = calcLagrangian(Lambda, HMEigValues, HMEigVects, g_M, p_M, s)
    prev_lagrangian = lagrangian

    logger.debug("Starting coarse-grain multiplier search.")
    logger.debug("lambda        Lagrangian value:\n")
    #print("lambda        Lagrangian value:")

    lagIter = 0
    while lagIter < 1000:
        lagrangian = calcLagrangian(Lambda, HMEigValues, HMEigVects, g_M, p_M, s)
        logger.debug("%15.10e  %8.3e" % (Lambda, lagrangian) )
        #print("%15.10e  %8.3e" % (Lambda, lagrangian) )

        if lagrangian < 0 and abs(lagrangian) < abs(lb_lagrangian):
            lb_lagrangian = lagrangian
            lb_lambda = Lambda
        elif lagrangian > 0 and abs(lagrangian) < abs(up_lagrangian):
            up_lagrangian = lagrangian
            up_lambda = Lambda

        if prev_lagrangian * lagrangian < 0:
            break;
        prev_lagrangian = lagrangian

        lagIter += 1
        Lambda -= 0.001

    logger.debug("Coarse graining results:")
    logger.debug("Lambda between %10.5f and %10.5f" % (lb_lambda,up_lambda))
    logger.debug("for Lagrangian %10.5f to  %10.5f" % (lb_lagrangian,up_lagrangian))

    # Calulate lambda using Householder method
    prev_lambda = -999
    lagIter = 0
    Lambda = (lb_lambda + up_lambda)/2 # start in middle of coarse range

    logger.debug("lambda        Lagrangian:")
    while abs(Lambda - prev_lambda) > 10**-15:
        prev_lagrangian = lagrangian
        L_derivs = calcLagrangianDerivs(Lambda, HMEigValues, HMEigVects, g_M, p_M, s)
        lagrangian = L_derivs[0]
        logger.debug("%15.5e%15.5e" % (Lambda, lagrangian) )
        
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
            Lambda = (prev_lambda + Lambda ) / 2
            prev_lambda = current_lambda
        else:
            prev_lambda = Lambda
            Lambda += h_f * (
            24 * L_derivs[1] + 24 * L_derivs[2] * h_f + 4 * L_derivs[3] * h_f**2) / (
            24 * L_derivs[1] + 36 * h_f * L_derivs[2] +
             6 * L_derivs[2]**2 * h_f**2 / L_derivs[1] +
             8 * L_derivs[3] * h_f**2 + L_derivs[4] * h_f**3)

        lagIter += 1
        if lagIter > 50:
            prev_lambda = Lambda
            Lambda = (lb_lambda + up_lambda) / 2 #Try a bisection after 50 attempts

        if lagIter > 200:
            err_msg = "Could not converge Lagrangian multiplier for constrained rxnpath search."
            logger.warning(err_msg)
            raise AlgError(err_msg)

    logger.info("Lambda converged at %15.5e" % Lambda)

    # Find dq_M from Eqn. 24 in Gonzalez & Schlegel (1990).
    # dq_M = (H_M - lambda I)^(-1) [lambda * p_M - g_M]
    LambdaI = np.identity(oMolsys.Nintcos)
    LambdaI = np.multiply(Lambda, LambdaI)
    deltaQM = symmMatInv(np.subtract(H_M, LambdaI))
    deltaQM = np.dot(deltaQM, np.subtract(np.multiply(Lambda, p_M), g_M))
    logger.debug("dq_M to next geometry\n" + printArrayString(deltaQM))

    # Find dq = G^(1/2) dq_M and do displacements.
    dq = np.dot(G_prime_root, deltaQM)
    logger.info("dq to next geometry\n" + printArrayString(dq))
    # TODO write geometry for multiple fragments
    #displace(oMolsys.intcos, oMolsys._fragments[0].geom, dq)
    displaceMolsys(oMolsys, dq)

    # Complete history entry of step.
    # Compute gradient and hessian in step direction
    dq_norm = np.linalg.norm(dq)
    dq_unit = dq / dq_norm
    dq_grad = -1 * f_q.dot(dq_unit)
    dq_hess = dq_unit.dot( H_q.dot(dq_unit) )
    DE      = IRC_DE_projected(dq_norm, dq_grad, dq_hess)

    if op.Params.print_lvl > 1:
        logger.info('\tQuadratic |target step|             : %15.10f' % dq_norm)
        logger.info('\tQuadratic gradient in step direction: %15.10f' % dq_grad)
        logger.info('\tQuadratic hessian in step direction : %15.10f' % dq_hess)
    logger.info("\tQuadratic Projected Delta(E)        : %15.10f" % DE)

    oHistory.appendRecord(DE, dq, dq_unit, dq_grad, dq_hess)

    logger.info("IRC Constrained step calculation finished.")
    return dq


# Calculates and returns value of Lagrangian function given multiplier Lambda.
def calcLagrangian(Lambda, HMEigValues, HMEigVects, g_M, p_M, s):
    lagrangian = 0
    for i in range(len(HMEigValues)):
        numerator = HMEigValues[i] * np.dot(HMEigVects[i], p_M) - np.dot(HMEigVects[i], g_M)
        denom = HMEigValues[i] - Lambda
        lagrangian += (numerator / denom) ** 2

    lagrangian -= (0.5 * s)**2
    return lagrangian

# Calculates and returns value of derivative of Lagrangian function given multiplier Lambda.
def calcLagrangianDerivs(Lambda, HMEigValues, HMEigVects, g_M, p_M, s):
    deriv = np.array( [0.0, 0.0, 0.0, 0.0, 0.0], float)
    for i in range(len(HMEigValues)):
        numerator = HMEigValues[i] * np.dot(HMEigVects[i], p_M) - np.dot(HMEigVects[i], g_M)
        D = HMEigValues[i] - Lambda
        deriv[0] +=     (numerator / D) ** 2
        deriv[1] += 2  *(numerator / D) ** 2 / (D) 
        deriv[2] += 6  *(numerator / D) ** 2 / (D*D)
        deriv[3] += 24 *(numerator / D) ** 2 / (D*D*D)
        deriv[4] += 120*(numerator / D) ** 2 / (D*D*D*D)

    deriv[0] -= (0.5 * s)**2

    return deriv

# mass-weighted distance from previous rxnpath point to new one
def calcLineDistStep(oMolsys):
    G      = oMolsys.Gmat(oMolsys.masses)
    G_root = symmMatRoot(G)
    G_inv  = symmMatInv(G_root)
    G_root_inv  = symmMatRoot(G_inv)

    rxn_Dq  = np.subtract(oMolsys.qArray(), IRCdata.history.q())
    # mass weight (not done in old C++ code)
    rxn_Dq_M = np.dot(G_root_inv, rxn_Dq)
    return np.linalg.norm ( rxn_Dq_M )

# Let q0 be last rxnpath point and q1 be new rxnpath point.  q* is the pivot
# point (1/2)s from each of these.  Returns the length of circular arc connecting
# q0 and q1, whose center is equidistant from q0 and q1, and for which line segments
# from q* to q0 and from q* to q1 are perpendicular to segments from the center
# to q0 and q1.
def calcArcDistStep(oMolsys):
    qp = IRCdata.history.q_pivot(-1) # pivot point is stored in previous step
    q0 = IRCdata.history.q(-1)
    q1 = oMolsys.qArray()

    p    = np.subtract(q1, qp)  # Dq from pivot point to latest rxnpath pt.
    line = np.subtract(q1, q0)  # Dq from rxnpath pt. to rxnpath pt.

    # mass-weight
    G      = oMolsys.Gmat(oMolsys.masses)
    G_root = symmMatRoot(G)
    G_inv  = symmMatInv(G_root)
    G_root_inv  = symmMatRoot(G_inv)
    p[:]    = np.multiply( 1.0/np.linalg.norm(p),    p )
    line[:] = np.multiply( 1.0/np.linalg.norm(line), line )

    alpha = acos( np.dot(p, line) )
    arcDistStep = IRCdata.history.step_size * alpha / tan(alpha)
    return arcDistStep


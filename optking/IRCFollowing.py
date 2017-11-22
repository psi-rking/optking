from math import sqrt, fabs

import numpy as np

from . import displace
from . import intcosMisc
from .linearAlgebra import symmMatEig, symmMatInv, symmMatRoot
from .printTools import printArray, printMat, print_opt


#Takes a half step from starting geometry along the gradient, then takes an additional half step as a guess
#returns dq
def takeHessianHalfStep(Molsys, Hq, B, fq, s, direction='forward'):
    print_opt(
        "==================================================================================\n"
    )
    print_opt("Taking Hessian IRC HalfStep and Guess Step\n")
    #Calculate G, G^-1/2 and G-1/2
    Gm = intcosMisc.Gmat(Molsys.intcos, Molsys.geom, Molsys.masses)
    GmInv = symmMatInv(Gm)
    GmRoot = symmMatRoot(Gm)
    GmRootInv = symmMatInv(GmRoot)

    #PrintStartingMatrices
    print_opt("B matrix\n")
    printMat(B)
    print_opt("Mass Weighted G Root Matrix\n")
    printMat(GmRoot)
    print_opt("Hesian in Internals\n")
    printMat(Hq)
    #Hq = intcosMisc.convertHessianToInternals(H, Molsys.intcos, Molsys.geom)
    HEigVals, HEigVects = symmMatEig(Hq)
    #get gradient from Psi4 in cartesian geom
    #gx = fgradient(geom)
    #gq = np.dot(GmInv, np.dot(B, gx))

    #initial internal coordinates from .intcosMisc
    qZero = intcosMisc.qValues(Molsys.intcos, Molsys.geom)

    #symmMatEig returns the Eigen Vectors as rows in order of increasing eigen values
    #first step from TS will be along the smallest eigenvector
    gk = np.zeros(len(HEigVects[0]), float)

    for col in range(len(HEigVects[0])):
        gk[col] = HEigVects[0, col]

    if direction == 'backward':
        for i in range(len(gk)):
            gk[i] = -1 * gk[i]

    #To solve N = (gk^t * G * gk)
    N = np.dot(gk.T, np.dot(Gm, gk))
    N = 1 / sqrt(N)

    #g*k+1 = qk - 1/2 * s * (N*G*gk)
    #N*G*gk
    dqPivot = np.dot(N, np.dot(Gm, gk))

    #applying weight 1/2 s to product (N*G*gk)
    #then applies vector addition q -  1/2 * s* (N*G*gk)
    dqPivot = np.dot(0.5, np.dot(s, dqPivot))
    qPivot = np.add(qZero, dqPivot)
    #To-Do add cartesian coordinate for pivot point
    #qPrime = np.dot (N, np.dot(G, gQ))
    dqPrime = np.dot(2, dqPivot)
    qPrime = np.add(dqPrime, qZero)
    displaceIRCStep(Molsys, dqPrime, Hq, fq)

    print_opt("next geometry\n ")
    print(qPrime)
    print_opt("Dq to Pivot Point\n")
    printArray(dqPivot)
    print_opt("Dq to Guess Point\n")
    print(dqPrime)
    print_opt(
        "===================================================================================\n"
    )

    return dqPivot, qPivot, qPrime


#Takes a half step from starting geometry along the gradient, then takes an additional half step as a guess
#returns dq
def takeGradientHalfStep(Molsys, H, B, s, gX):
    #Calculate G, G^-1/2 and G-1/2
    Gm = intcosMisc.Gmat(Molsys.intcos, Molsys.geom, Molsys.masses)
    GmInv = symmMatInt(Gm)
    GmRoot = symmMatRoot(Gm)
    GmRootInv = symmMatInv(GmRoot)

    qZero = intcosMisc.qValues(Molsys.intcos, Molsys.geom)

    #convert gradient to Internals
    gQ = np.dot(GmInv, np.dot(B, gX))

    #To solve N = (gk^t * G * gk)
    N = symmMatRoot(np.dot(gQ.T, np.dot(Gm, gQ)), True)

    #g*k+1 = qk - 1/2 * s * (N*G*gk)
    #N*G*gk
    qPivot = np.dot(N, np.dot(G, gQ))

    #applying weight 1/2 s to product (N*G*gk)
    #then applies vector addition q -  1/2 * s* (N*G*gk)
    for i in range(len(gQ)):
        qPivot[i] = 0.5 * s * qPivot[i]
        qPivot = np.subtract(qZero, qPivot)

    #displaceIRCStep(Molsys.intcos, Molsys.geom, np.subtract(qPivot, qZero), H, g)

    qPrime = np.dot(N, np.dot(G, gQ))
    for i in range(len(gQ)):
        qPrime[i] = 0.5 * s * qPrime[i]
        qPrime[i] = qPivot[i] - qPrime[i]

    dq = np.subtract(qPrime, qZero)
    dq = sqrt(np.dot(dq, dq))

    return qPivot, qPrime, Dq


#Before Dq_IRC is called, the goemetry must be updated to the guess point
#Returns Dq from qk+1 to gprime.
def Dq(Molsys, g, E, Hq, B, s, qPrime, dqPrime):
    print_opt(
        "======================================================================================================\n"
    )
    print_opt("Starting constrained optimization\n")
    GPrime = intcosMisc.Gmat(Molsys.intcos, Molsys.geom, Molsys.masses)
    GPrimeInv = symmMatInv(GPrime)
    GPrimeRoot = symmMatRoot(GPrime)
    GPrimeRootInv = symmMatRoot(GPrime, True)
    #Hq = intcosMisc.convertHessianToInternals(Hq, Molsys.intcos, Molsys.geom)
    #Hq = intcosMisc.convertHessianToInternals(H, Molsys.intcos, Molsys.geom)
    #vectors nessecary to solve for Lambda, naming is taken from Gonzalez and Schlegel
    deltaQM = 0
    pPrime = dqPrime
    #print_opt ("G prime root matrix")
    #printMat (GPrimeRoot)
    #print_opt ("gradient")
    #printArray (g)
    #print_opt ("Hessian in Internals")
    #printMat (Hq)

    u = np.identity(Molsys.Natom * 3)
    print_opt("Cartesian Gradient\n")
    printArray(g)
    g = np.dot(GPrimeInv, np.dot(B, np.dot(u, g)))
    print_opt("Internal Gradient\n")
    printArray(g)
    gM = np.dot(GPrimeRoot, g)
    #print ("gM")
    #print (gM)
    HM = np.dot(GPrimeRoot, np.dot(Hq, GPrimeRoot))
    pM = np.dot(GPrimeRootInv, pPrime)
    HMEigValues, HMEigVects = symmMatEig(HM)
    #Variables for solving lagrangian function
    lowerBLagrangian = -100
    upperBLagraan = 100
    lowerBLambda = 0.5 * HMEigValues[0]
    upperBLambda = 0.5 * HMEigValues[0]
    Lambda = 0.5 * HMEigValues[0]
    prevLambda = Lambda
    prevLagrangian = 1
    lagrangian = 1

    #Solves F(L) = Sum[(bj*pj - gj)/(bj-L)]^2 - 1/2s = 0
    #coarse search
    lagIter = 0
    while (prevLagrangian * lagrangian > 0) and lagIter < 1000:
        prevLagrangian = lagrangian
        Lambda -= 1
        lagrangian = calcLagrangian(Lambda, HMEigValues, HMEigVects, gM, pM, s)
        if lagrangian < 0 and fabs(lagrangian) < fabs(lowerBLagrangian):
            lowerBLagrangian = lagrangian
            lowerBLambda = Lambda

        if lagrangian > 0 and fabs(lagrangian) < fabs(upperBLagrangian):
            upperBLagrangian = lagrangian
            upperBLambda = Lambda
        lagIter += 1
        Lambda -= 1

    #fine search
    #calulates next lambda using Householder method

    dLagrangian = np.array(
        [2, 6, 24, 120]
    )  #array of lagrangian derivatives to solve Householder method with weights to solve derivative
    lagIter = 0

    while lagrangian - prevLagrangian > 10**-16:
        prevLagrangian = lagrangian
        for i in range(4):
            dLagrangian[i] *= calcLagrangian(Lambda, HMEigValues, HMEigVects, gM, pM, s)

        h_f = -lagrangian / dLagrangian[1]

        if lagrangian < 0 and (fabs(lagrangian) < fabs(lowerBLagrangian)):
            lowerBLagrangian = lagrangian
            lowerBLambda = Lambda

        elif lagrangian > 0 and fabs(lagrangian) < fabs(upperBLagrangian):
            upperBLagrangian = lagrangian
            upperBLambda = Lambda

        elif Lagrangian * prevLagrangian < 0:
            Lagrangian = (Lagrangian + prevLagrangian) / 2
            #Lagrangian found
        else:
            prevLambda = Lambda
            Lambda += h_f * (24 * dLagrangian[0] * 24 * dLagrangian[1] * 8 * h_f +
                             4 * dLagrangian[2] * 8 * h_f**2) / (
                                 24 * dLagrangian[0] + 36 * h_f * dLagrangian[1] + 6 *
                                 (dlagrangian[1]**2 / dLagrangian[0]) * 8 * h_f**2 +
                                 8 * dLagrangian[2] * h_f**2 + dLagrangian[3] * h_f**3)
        lagIter += 1

        if lagIter > 50:
            prevLambda = Lambda
            Lambda = (lb_Lambda + ub_Lambda) / 2

        if lagIter > 200:
            print("Exception should have been thrown")
            #needs to throw failure to converge exception

    #constructing Lambda * intcosxintcosI
    LambdaI = np.zeros((len(g), len(g)), float)

    for i in range(len(g)):
        LambdaI[i][i] = 1 * Lambda

    deltaQM = symmMatInv(-np.subtract(HM, LambdaI))
    deltaQM = np.dot(deltaQM, np.subtract(gM, np.multiply(Lambda, pM)))
    print_opt("initial geometry\n")
    printArray(qPrime)
    dq = np.dot(GPrimeRoot, deltaQM)
    print_opt("dq to next geometry\n")
    printArray(dq)
    displaceIRCStep(Molsys, dq, Hq, np.dot(-1, g))
    print_opt("New internal coordinates\n")
    qNew = np.add(qPrime, dq)
    printArray(qNew)

    print_opt("Constrained optimization finished\n")
    print_opt(
        "======================================================================================================\n"
    )
    # save values in step data
    #History.appendRecord(DEprojected, dq, ircU, ircG, ircH)

    return dq


# calculates Lagrangian function of Lambda
# returns value of lagrangian function
def calcLagrangian(Lambda, HMEigValues, HMEigVects, gM, pM, s):
    lagrangian = 0
    for i in range(len(HMEigValues)):
        numerator = (HMEigValues[i] * np.dot(pM.T, HMEigVects[i])) - (np.dot(
            gM.T, HMEigVects[i]))
        denominator = HMEigValues[i] - Lambda
        lagrangian += numerator / denominator
    lagrangian *= lagrangian
    lagrangian -= (0.5 * s)**2

    return lagrangian


#displaces an atom with the dq from the IRC data
#returns void
def displaceIRCStep(Molsys, dq, H, fq):
    # get norm |q| and unit vector in the step direction
    ircDqNorm = sqrt(np.dot(dq, dq))
    ircU = dq / ircDqNorm
    print_opt("\tNorm of target step-size %15.10f\n" % ircDqNorm)

    # get gradient and hessian in step direction
    ircG = np.dot(-1, np.dot(fq, ircU))  # gradient, not force
    ircH = np.dot(ircU, np.dot(H, ircU))

    #if op.Params.print_lvl > 1:
    #print_opt('\t|IRC target step|: %15.10f\n' % ircDqNorm)
    #print_opt('\tIRC gradient     : %15.10f\n' % ircG)
    #print_opt('\tIRC hessian      : %15.10f\n' % ircH)
    #DEprojected = stepAlgorithms.DE_projected('IRC', ircDqNorm, ircG, ircH)
    #print_opt("\tProjected energy change by quadratic approximation: %20.10lf\n" % DEprojected)

    # Scale fq into aJ for printing
    fq_aJ = intcosMisc.qShowForces(Molsys.intcos, fq)
    #print ("------------------------------")
    #print (Molsys._fragments[0].intcos)
    #print (Molsys._fragments[0].geom)
    #print (dq)
    #print (fq_aJ)
    #print ("--------------------------------")
    displace(Molsys._fragments[0].intcos, Molsys._fragments[0].geom, dq, fq_aJ)

    dq_actual = sqrt(np.dot(dq, dq))
    print_opt("\tNorm of achieved step-size %15.10f\n" % dq_actual)

    # Symmetrize the geometry for next step


# # symmetrize_geom()

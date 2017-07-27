from displace import displace
import intcosMisc
from addIntcos import linearBendCheck
from math import sqrt, fabs
from printTools import printArray, printMat, print_opt
from linearAlgebra import absMax, symmMatEig, asymmMatEig, symmMatInv, symmMatRoot
from history import History
import numpy as np

#Takes a half step from starting geometry along the gradient, then takes an additional half step as a guess
#returns dq
def takeHessianHalfStep(Molsys, H, B, s, direction = 'forward'):
    #Calculate G, G^-1/2 and G-1/2
    Gm = intcosMisc.Gmat(Molsys.intcos, Molsys.geom, Molsys.masses)
    GmInv = symmMatInv(Gm)
    GmRoot = symmMatRoot(Gm)
    GmRootInv = symmMatInv(GmRoot)

    Hq = intcosMisc.convertHessianToInternals(H, Molsys.intcos, Molsys.geom)
    HEigVals, HEigVects = symmMatEig(Hq)
    
    #get gradient from Psi4 in cartesian geom
    #gx = fgradient(geom)    
    #gq = np.dot(GmInv, np.dot(B, gx))
    
    #initial internal coordinates from intcosMisc
    qZero = intcosMisc.qValues(Molsys.intcos, Molsys.geom)

    #symmMatEig returns the Eigen Vectors as rows in order of increasing eigen values
    #first step from TS will be along the smallest eigenvector
    gk = np.zeros(len(HEigVects[0]), float)

    for col in range (len(HEigVects[0])):
        gk[col] = HEigVects[0, col]

    if (direction == 'backward'):
        for i in range (len(gk)):
            gk[i] = -1 * gk[i]

    #To solve N = (gk^t * G * gk)    
    N = np.dot(gk.T, np.dot(Gm, gk))
    N = 1/sqrt(N)

    #g*k+1 = qk - 1/2 * s * (N*G*gk)
    #N*G*gk
    qPivot = np.dot(N, np.dot(Gm, gk))

    #applying weight 1/2 s to product (N*G*gk)
    #then applies vector addition q -  1/2 * s* (N*G*gk)
    for i in range (len(gk)):
        qPivot[i] = 0.5 * s * qPivot[i]
        qPivot = np.subtract(qZero, qPivot)
    
    #displaceIRCStep(Molsys.intcos, Molsys.geom, np.subtract(qPivot, qZero), H, g)

    #To-Do add cartesian coordinate for pivot point 
    #qPrime = np.dot (N, np.dot(G, gQ))
    for i in range (len(gk)):
        qPrime = np.dot(2, np.subtract(qPivot[i], qZero[i]))
    
    dq = np.subtract(qPrime, qZero)
    dq = sqrt(np.dot(dq, dq))
    return qPivot, qPrime, dq

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
    for i in range (len(gQ)):
        qPivot[i] = 0.5 * s * qPivot[i]
        qPivot = np.subtract(qZero, qPivot)
     
    #displaceIRCStep(Molsys.intcos, Molsys.geom, np.subtract(qPivot, qZero), H, g)
     
    qPrime = np.dot (N, np.dot(G, gQ))
    for i in range (len(gQ)):
        qPrime[i] = 0.5 * s * qPrime[i]
        qPrime[i] = qPivot[i] - qPrime[i]   
   
     
    dq = np.subtract(qPrime, qZero) 
    dq = sqrt(np.dot(dq, dq))
    
    return qPivot, qPrime, Dq
    
#Before Dq_IRC is called, the goemetry must be updated to the guess point    
#Returns Dq from qk+1 to gprime.
def Dq(Molsys, E, Hq, B, s, g, qPrime, qPivot):

    GPrime = intcosMisc.Gmat(Molsys.intcos, Molsys.geom, Molsys.masses) 
    GPrimeRoot = symmMatRoot(GPrime)
    GPrimeRootInv = symmMatRoot(GPrime, True)
    
    #vectors nessecary to solve for Lambda, naming is taken from Gonzalez and Schlegel
    deltaQM = 0
    pPrime = np.subtract(qPrime, qPivot)
    gM = np.dot(GPrimeRoot, g)
    HM = np.dot(GPrimeRoot, np.dot(Hq, GPrimeRoot))
    pM = np.dot(GPrimeRootInv, pPrime) 
    HMEigValues, HMEigVects = symmMatEig(HM)
    
    #Variables for solving lagrangian function
    lowerBLagrangian = -100
    upperBLagrangian = 100
    lowerBLambda
    upperBLambda
    Lambda = 0.5 * HMEigValues[i]
    prevLambda = Lambda
    prevLagrangian = 1
    lagrangian = 1

    #Solves F(L) = Sum[(bj*pj - gj)/(bj-L)]^2 - 1/2s = 0
    #coarse search
    lagIter = 0
    while((prevLagrangian * lagrangian > 0) and lagIter < 1000):
        prevLagrangian = lagrangian
        Lambda -= 1
        lagrangian = calcLagrangian(Lambda, len(intcos), 0) 
        
        if (lagrangian < 0 and fabs(lagrangian) < fabs(lowerBLagrangian)):
            lowerBLagrangian = lagrangian
            lowerBLambda = Lambda
        
        if (lagrangian > 0 and fabs(lagrangian) < fabs(upperBLagrangian)):
            upperBLagrangian = lagrangian
            upperBLambda = Lambda
        lagIter += 1
        Lambda -= 1

    #fine search
    #calulates next lambda using Householder method

    dLagrangian = np.array([2, 6, 24, 120]) #array of lagrangian derivatives to solve Householder method with weights to solve derivative
    lagIter = 0

    while(Lagrangian - prevLagrangian > 10**-16):
        prevLagrangian = Lagrangian
        for i in range (4):
            dLagrangian[i] *= calcLagrangian(Lambda, len(intcos), i + 1)
    
        h_f = -lagrangian/ dLagrangian[1] #I dont know what this is in terms of the solution of the householder method
               
        if (lagrangian < 0 and fabs(lagrangian) < fabs(lowerBLagrangian)):
            lowerBLagrangian = lagrangian
            lowerBLambda = Lambda        
        
        elif (lagrangian > 0 and fabs(lagrangian) < fabs(upperBLagrangian)):
            upperBLagrangian = lagrangian
            upperBLambda = Lambda
    
        elif (Lagrangian * prevLagrangian < 0):
            Lagrangian = (Lagrangian + prevLagrangian)/2
            #Lagrangian found    
        else:
            prevLambda = Lambda
            Lambda += h_f * (24*dLagrangian[0] * 24*dLagrangian[1] * 8 * h_f + 4 *dLagrangian[2] * 8 * h_f**2) / (24*dLagrangian[0] + 36*h_f *dLagrangian[1] \
                + 6 * (dlagrangian[1] ** 2/ dLagrangian[0]) * 8 * h_f**2 + 8 * dLagrangian[2] * h_f**2 + dLagrangian[3] * h_f**3) 
        lagIter += 1

        if (lagIter > 50):
            prevLambda = Lambda
            Lambda = (lb_Lambda + ub_Lambda) / 2
        
        if (lagIter > 200):
            print("Exception should have been thrown")
            #needs to throw failure to converge exception
            
    #constructing Lambda * intcosxintcosI
    LambdaI = np.zeroes((len(fq), len(fq)), float)

    for i in range(len(fq)):
        LambdaI[i][i] = 1 * Lambda
 
    deltaQM = symmMatInv(-np.subtract(HM, LambdaI))  
    deltaQM = np.dot(deltaQM, np.subtract(gM, np.multiply(Lambda, pM)))    
    
    dq = np.dot(GPrimeRoot, deltaQM)
    dq = sqrt(np.dot(dq, dq))

    displaceIRCHistory(Molsys.intcos, Molsys.geom, dq, H, g)
     
    # save values in step data
    #History.appendRecord(DEprojected, dq, ircU, ircG, ircH)

    return dq    

# calculates Lagrangian function of Lambda
# returns value of lagrangian function
def calcLagrangian(Lambda, dim, derivativeOrder, HMEigValues, HMEigVects, gM, pM):

    for i in range (dim):
        numerator = np.subtract(np.dot(HMEigValues[i], np.dot(pM.T, HMEigVects[i])), np.dot(gM.T, HMEigValues))
        denominator = HMEigValues - Lambda
        lagrangian += numerator/(denominator**(i+1))
    lagrangian *= lagrangian
    lagrangian -= (0.5*s)**2

    return lagrangian

#displaces an atom with the dq from the IRC data
#returns void
def displaceIRCStep(intcos, geom, dq, H, gq):
    # get norm |q| and unit vector in the step direction
    ircDqNorm = sqrt(np.dot(dq,dq))
    ircU = dq.copy() / ircDqNorm
    print_opt("\tNorm of target step-size %15.10f\n" % ircDqNorm)

    # get gradient and hessian in step direction
    ircG = np.dot(gq, ircU) # gradient, not force
    ircH = np.dot( ircU, np.dot(H, ircU) )

    if op.Params.print_lvl > 1:
       print_opt('\t|IRC target step|: %15.10f\n' % ircDqNorm)
       print_opt('\tIRC gradient     : %15.10f\n' % ircG)
       print_opt('\tIRC hessian      : %15.10f\n' % ircH)
    DEprojected = DE_projected('IRC', ircDqNorm, ircG, ircH)
    print_opt("\tProjected energy change by quadratic approximation: %20.10lf\n" % DEprojected)

    # Scale fq into aJ for printing
    fq_aJ = qShowForces(Molsys.intcos, fq)
    displace(Molsys._fragments[0].intcos, Molsys._fragments[0].geom, dq, fq_aJ)

    dq_actual = sqrt( np.dot(dq,dq) )
    print_opt("\tNorm of achieved step-size %15.10f\n" % dq_actual)

    # Symmetrize the geometry for next step
    # symmetrize_geom()

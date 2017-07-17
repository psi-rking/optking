import Psi4
import displace
from intcosMisc import convertHessianToInternals, qValues
from addIntcos import linearBendCheck
from math import sqrt, fabs
from printTools import printArray, printMat, print_opt
from linearAlgebra import absMax, symmMatEig, asymmMatEig, symmMatInv, symmMatRoot
from history import History
import v3d
import numpy as np

#Params the gradient in cartesians and Hessian in cartesians calculated at the TS
def Dq_IRC(Molsys, intcos, geom, E, H, B, s, direction = 'forward', stepNumber, fgradient)
    #Calculate G, G^-1/2 and G-1/2
    Gm = intcosMisc.Gmat(intcos, geom, True)
    GmInv = symmMatIntc(Gm)
    GmRoot = symmMatRoot(Gm)
    GmRootInv = summMatInv(GmRoot)

    Hq = convertHessianToInternals(H, intcos, geom)
    HEigVals, HEigVects = symmMatEigH(H)
    
    #get gradient from Psi4 in cartesian geom
    gx = fgradient(geom)    
    gq = np.dot(GmInv, np.dot(B, gx)
    
    #initial internal coordinates from intcosMisc
    qZero = qValues(intcos, geom)

    #symmMatEig returns the Eigen Vector as a row in order of increasing eigen value
    #first step from TS will be along the smallest eigenvector
    gk = np.zeros(Molsys.nAtom, float)
    if (stepNumber == 0):
        for col in range (Molsys.Natom):
            gk[col] = HEigVects[0, col]

        if (direction == 'backward'):
            for i in range (len(gk)):
                gk[i] = -1 * gk[i]
    #depending on the loop structure, may need to recalc the gradient / forces at this point    
    else 
        for i in range len(g): #if not at the TS, set the pivot vecct equal to the gradient (- Force)
           gk = g[i]
    #To solve N = (gk^t * G * gk)    
    N = symmMatRoot(np.dot(gk.T, np.dot(Gm, gk)), True)

    #g*k+1 = qk - 1/2 * s * (N*G*gk)
    #N*G*gk
    qPivot = np.dot (N, np.dot(G, gk))

    #applying weight 1/2 s to product (N*G*gk)
    #then applies vector addition q -  1/2 * s* (N*G*gk)
    for i in range (len(gk)):
        qPivot[i] = 0.5 * s * qPivot[i]
        qPivot[i] = qZero[i] - qPivot[i]
    #To-Do add cartesian coordimate for pivot point
    #For initial guess of Qprime take another step 1/2s in same direction as qPivot 
    qPrime = np.dot (N, np.dot(G, gk))
    for i in range (len(gk)):
        qPrime[i] = 0.5 * s * qPrime[i]
        qPrime[i] = qPivot[i] - qPrime[i]    
    
    #before this happens we need to update the geometry of the molsys or fragment to qprime
    #recalculategradient at qPrime
    dq = np.subtract(qPrime, qZero)
    dislace(intcos, geom, dq, fq) #before I calculate the gmat need to move to the new geometry at qPrime
    GPrime = intcosMisc.Gmat(intcos, geom, True) 
    GPrimeRoot = symmMatRoot(GPrime)
    GPrimeRootInv = symmMatRoot(GPrime, True)

    #vectors nessecary to solve for Lambda, naming is taken from Gonzalez and Schlegel
    deltaQM = 0
    pPrime = np.subtract(qPrime, qPivot)
    gm = np.dot(GPrimeRoot, gPrime)
    HM = np.dot(GPrimeRoot, np.dot(Hq, GPrimeRoot)) #Does the hessian need to be recalculated? I dont think so
    pm = np.dot(GPrimeRootInv, pPrime) 
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
    while(prevLagrangian * lagrangian !< 0 && lagIter < 1000)
        prevLagrangian = lagrangian
        lambda -= 1
        lagrangian = calcLagrangian(lambda, len(intcos), 0) 
        
        if (lagrangian < 0 && fabs(lagrangian) < fabs(lowerBLagrangian)):
            lowerBLagrangian = lagrangian
            lowerBLambda = Lambda
        
        if (lagrangian > 0 && fabs(lagrangian) < fabs(upperBLagrangian)):
            upperBLagrangian = lagrangian
            upperBLambda = Lambda
        lagIter += 1
        lambda -1

    #fine search
    #calulates next lambda using Householder method

    dLagrangian = np.array([2, 6, 24, 120]) #array of lagrangian derivatives to solve Householder method with weights to solve derivative
    lagIter = 0
    while(Lagrangian - prevLagrangian < 1*10^-16 )
        prevLagrangian = Lagrangian
        for i in range (4):
            dLagrangian[i] *= calcLagrangian(lambda, len(intcos), i + 1)
    
        h_f = -lagrangian/ dLagrangian[1] #I dont know what this is in terms of the solution of the householder method
               
        if (lagrangian < 0 && fabs(lagrangian) < fabs(lowerBLagrangian)):
            lowerBLagrangian = lagrangian
            lowerBLambda = Lambda        
        
        elif (lagrangian > 0 && fabs(lagrangian) < fabs(upperBLagrangian)):
            upperBLagrangian = lagrangian
            upperBLambda = Lambda
    
        elif (Lagrangian * prevLagrangian < 0):
            Lagrangian = (Lagrangian + prevLagrangian)/2
            #Lagrangian found    
        else:
            prevLambda = Lambda
            lambda += h_f * (24*dLagrangian[0] * 24*dLagrangian[1] 8 h_f + 4 *dLagrangian[2] 8 h_f**2) / (24*dLagrangian[0] + 36*h_f *dLagrangian[1] + 6 * (dlagrangian[1] ** 2/ dLagrangian[0]) 8 h_f**2 + 8 * dLagrangian[2] * h_f**2 + dLagrangian[3] * h_f**3) 
        lagIter += 1

        if (lagIter > 50):
            prevLambda = Lambda
            Lambda = (lb_Lambda + ub_Lambda) / 2
        
        if (lagIter > 200)
            #needs to throw failure to converge exception            
    #constructing Lambda * intcosxintcosI
    LambdaI = np.zeroes((len(fq), len(fq)), float)
        for i in range (len(fq)):
        LambdaI[i][i] = 1 * Lambda
 
    deltaQM = symmMatInv(-np.subtract(HM, LambdaI))  
    deltaQM = np.dot(deltaQM, np.subtract(gm, np.multiply(Lambda, pm)))    

    dq = np.dot(GPrimeRoot, deltaQM)
        
    return dq    

# calculates Lagrangian function of Lambda
# params lambda
#        dim - number of internal coordinates
#        derivativeOrder - which derivative to calcualate
# returns value of lagrangian function
def calcLagrangian(lambda, dim, derivativeOrder):

    for i in range (dim):
        numerator = np.subtract(np.dot(HMEigValues[i], np.dot(pm.T, HMEigVects[i])), np.dot(gm.T, HMEigValues))
        denominator = HMEigValues - Lambda
        lagrangian += numerator/(denominator**(i+1))
    lagrangian *= lagrangian
    lagrangian -= (0.5*s)**2
           
    return lagrangian


import Psi4
import displace
from intcosMisc import convertHessianToInternals
from addIntcos import linearBendCheck
from math import sqrt, fabs
from printTools import printArray, printMat, print_opt
from linearAlgebra import absMax, symmMatEig, asymmMatEig, symmMatInv, symmMatRoot
from history import History
import v3d
import numpy as np

#Params the gradient in cartesians and Hessian in cartesians calculated at the TS
def Dq_IRC(Molsys, intcos, E, g, H, B, s, qZero, geom, direction = 'forward')
    #Calculate G, G^-1/2 and G-1/2
    Gm = intcosMisc.Gmat(intcos, geom, True)
    GmRoot = symmMatRoot(Gm)
    GmRootInv = summMatInv(GmRoot)
    
    Hq = convertHessianToInternals(H, intcos, geom)
    HEigVals, HEigVects = symmMatEigH(H)
    
    #symmMatEig returns the Eigen Vector as a row in order of increasing eigen value
    #first step from TS will be along the smallest eigenvector
    gk = np.zeros(Molsys.nAtom, float)
    if (at TS):
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
    gPrime = recalculategradient at qPrime
    GPrime = intcosMisc.Gmat(intcos, geom, True) 
    GPrimeRoot = symmMatRoot(GPrime)
    GPrimeRootInv = symmMatRoot(GPrime, True)
    #vectors nessecary to solve for Lambda
    deltaQM
    pPrime = np.subtract(qPrime, qPivot)
    gm = np.dot(GPrimeRoot, gPrime)
    HM = np.dot(GPrimeRoot, np.dot(Hq, GPrimeRoot)) #Does the hessian need to be recalculated? I dont think so
    pm = np.dot(GPrimeRootInv, pPrime) 
    HMEigValues, HMEigVects = symmMatEig(HM)
    
    Lambda = 0.5 * HMEigValues[i]
    previousF = 1
    FLambda = 1
    #Solves F(L) = Sum[(bj*pj - gj)/(bj-L)]^2 - 1/2s = 0
    #coarse search
    while (previousL * FLambda !< 0)
         for i in range (len(intcos)):
            Lambda -= 1
            previousL = FLambda
            numerator = np.subtract(np.dot(HMEigValues[i], np.dot(pm.T, HMEigVects[i])), np.dot(gm.T, HMEigValues))
            denominator = HMEigValues - Lambda
            FLambda += numerator/denominator
 
        FLambda *= FLambda
        FLabmda -= 0.5*s
    
          

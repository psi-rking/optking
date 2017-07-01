import numpy as np
from math import sqrt
import bend
import oofp
import tors
from linearAlgebra import symmMatInv
from printTools import printMat, printArray
import optParams as op

# Simple operations on internal :148
#coordinate sets.  For example,
# return values, Bmatrix or fix orientation.

# q    -> qValues
# DqDx -> Bmat
def qValues(intcos, geom):
    q  = np.zeros( (len(intcos)), float)
    for i, intco in enumerate(intcos):
        q[i] = intco.q(geom)
    return q

def qShowValues(intcos, geom):
    q  = np.zeros( (len(intcos)), float)
    for i, intco in enumerate(intcos):
        q[i] = intco.qShow(geom)
    return q

def updateDihedralOrientations(intcos, geom):
    for intco in intcos:
        if isinstance(intco, tors.TORS) or isinstance(intco, oofp.OOFP):
            intco.updateOrientation(geom)
    return

def fixBendAxes(intcos, geom):
    for intco in intcos:
        if isinstance(intco, bend.BEND):
            intco.fixBendAxes(geom)
    return

def unfixBendAxes(intcos):
    for intco in intcos:
        if isinstance(intco, bend.BEND):
            intco.unfixBendAxes()
    return

def Bmat(intcos, geom):
    Nint = len(intcos)
    Ncart = geom.size

    B = np.zeros( (Nint,Ncart), float)
    for i,intco in enumerate(intcos):
        intco.DqDx(geom, B[i])

    return B

def Gmat(intcos, geom, masses=None):
    B = Bmat(intcos, geom)

    if masses:
        for i in range(len(intcos)):
            for a in range(len(geom)):
                for xyz in range(3):
                    B[i][3*a+xyz] /= math.sqrt(masses[a]);

    return np.dot(B, B.T)
        

# Compute forces in internal coordinates in au, f_q = G_inv B u f_x
# if u is unit matrix, f_q = (BB^T)^(-1) * B f_x
def qForces(intcos, geom, gradient_x):
    if len(intcos) == 0 or len(geom) == 0: return np.zeros( (0), float)
    B = Bmat(intcos, geom)
    fx = -1.0 * gradient_x     # gradient -> forces
    temp_arr = np.dot(B,fx.T)
    del fx

    G    = np.dot(B, B.T)  
    del B
    Ginv = symmMatInv(G,redundant=True)
    del G

    fq = np.dot(Ginv,temp_arr.T)
    return fq

# Prints them, but does not recompute them.
def qShowForces(intcos, forces):
    qaJ = np.copy(forces)
    for i, intco in enumerate(intcos):
        qaJ[i] *= intco.fShowFactor
    return qaJ

def constraint_matrix(intcos):
    if not any( [coord.frozen for coord in intcos ]):
        return None
    C = np.zeros((len(intcos),len(intcos)), float)
    for i,coord in enumerate(intcos):
        if coord.frozen:
            C[i,i] = 1.0
    return C

# Project redundancies and constraints out of forces and Hessian.
def projectRedundanciesAndConstraints(intcos, geom, fq, H):
    dim = len(intcos)

    # compute projection matrix = G G^-1
    G = Gmat(intcos, geom)
    G_inv = symmMatInv(G, redundant=True)
    Pprime = np.dot(G, G_inv) 
    if op.Params.print_lvl >= 3:
        print "\tProjection matrix for redundancies."
        printMat(Pprime)

    # Add constraints to projection matrix
    C = constraint_matrix(intcos)

    if C is not None:
        print "Adding constraints for projection."
        print (C)
        P= np.zeros((len(intcos), len(intcos)), float)
        #print (np.dot(C, np.dot(Pprime, C)))
        CPC = np.zeros((len(intcos), len(intcos)), float) 
        CPC[:,:] = np.dot(C, np.dot(Pprime, C))
        CPC = symmMatInv(CPC, redundant = True)  
        P[:,:] = Pprime - np.dot(Pprime, np.dot(C, np.dot(CPC, np.dot(C, Pprime))))
        # Project redundancies out of forces.
        # fq~ = P fq
        fq[:] = np.dot(P, fq.T)

        if op.Params.print_lvl >= 3:
            print "\tInternal forces in au, after projection of redundancies and constraints."
            printArray(fq)
        # Project redundancies out of Hessian matrix.
        # Peng, Ayala, Schlegel, JCC 1996 give H -> PHP + 1000(1-P)
        # The second term appears unnecessary and sometimes messes up Hessian updating.
        tempMat = np.dot(H, P)
        H[:,:] = np.dot(P, tempMat)
        #for i in range(dim)
        #    H[i,i] += 1000 * (1.0 - P[i,i])
        #for i in range(dim)
        #    for j in range(i):
        #        H[j,i] = H[i,j] = H[i,j] + 1000 * (1.0 - P[i,j])
        if op.Params.print_lvl >= 3:
            print "Projected (PHP) Hessian matrix"
            printMat(H)

def applyFixedForces(Molsys, fq, H, stepNumber):
    x = Molsys.geom
    for iF,F in enumerate(Molsys._fragments):
        for i, intco in enumerate(F.intcos):
            if intco.fixed:
              location = Molsys.frag_1st_intco(iF) + i
              val   = intco.q(x)
              eqVal = intco.fixedEqVal

              # Increase force constant by 5% of initial value per iteration
              k = (1 + 0.05 * stepNumber) * op.Params.fixed_coord_force_constant;
              force = k * (eqVal - val)
              H[location][location] = k;

              print("\n\tAdding user-defined constraint: Fragment %d; Coordinate %d:" % (iF+1, i+1))
              print("\t\tValue = %12.6f; Fixed value    = %12.6f" % (val, eqVal))
              print("\t\tForce = %12.6f; Force constant = %12.6f" % (force, k))
              fq[location] = force

              # Delete coupling between this coordinate and others.
              print("\t\tRemoving off-diagonal coupling between coordinate %d and others." % (location+1))
              for j in range(len(H)): # gives first dimension length
                if j != location:
                  H[j][location] = H[location][j] = 0

    return


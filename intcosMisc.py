import numpy as np
from math import sqrt
import bend
import oofp
import tors
from linearAlgebra import symmMatInv
from printTools import printMat
import optParams as op

# Simple operations on internal coordinate sets.  For example,
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

#def project_dq(dq):
#  int Nintco = Ncoord();
#  int Ncart = 3*g_natom();
#
#  double *dq_orig; # only for printing
#  if Params.print_lvl >=2:
#    dq_orig = init_array(Nintco);
#    array_copy(dq, dq_orig, Ncoord());
#
#  double **B = compute_B();
#  opt_matrix_mult(B, 1, B, 0, G, 0, Ncart, Nintco, Ncart, 0);
#
#  # B dx = dq
#  # B^t B dx = B^t dq
#  # dx = (B^t B)^-1 B^t dq
#  double **G_inv = symmMatInv(G, Ncart, 1);
#  double **B_inv = init_matrix(Ncart, Nintco);
#  opt_matrix_mult(G_inv, 0, B, 1, B_inv, 0, Ncart, Ncart, Nintco, 0);
#
#  double **P = init_matrix(Nintco, Nintco);
#  opt_matrix_mult(B, 0, B_inv, 0, P, 0, Nintco, Ncart, Nintco, 0);
#
#  double * temp_arr = init_array(Nintco);
#  opt_matrix_mult(P, 0, &dq, 1, &temp_arr, 1, Nintco, Nintco, 1, 0);
#  array_copy(temp_arr, dq, Ncoord());
#
#  Params.print_lvl >=2:
#    print "Projection of redundancies out of step:\n");
#    print "\tOriginal dq     Projected dq     Difference\n");
#    for (int i=0; i<Nintco; ++i)
#      oprintf_out("\t%12.6lf    %12.6lf   %12.6lf\n", dq_orig[i], dq[i], dq[i]-dq_orig[i]);

def constraint_matrix(intcos):
    if not any( [coord.frozen for coord in intcos ]):
        return None
    C = np.zeros((len(intcos),len(intcos)), float)
    for i,coord in intcos:
        if coord.frozen:
            C[i,i] = 1.0
    return C

# Project redundancies and constraints out of forces and Hessian.
def project_redundancies(intcos, geom, fq, H):
    dim = len(intcos)

    # compute projection matrix = G G^-1
    G = Gmat(intcos, geom)
    G_inv = symmMatInv(G, redundant=True)
    P = np.dot(G, G_inv) 

    if op.Params.print_lvl >= 3:
        print "\tProjection matrix for redundancies."
        printMat(P)

    # Add constraints to projection matrix
    C = constraint_matrix(intcos)

    if C != None {
        print "Adding constraints for projection."

    """
  # P = P' - P' C (CPC)^-1 C P'
  if (constraints_present) {
    double **T = init_matrix(Nintco,Nintco);
    opt_matrix_mult(P, 0, C, 0,  T, 0, Nintco, Nintco, Nintco, 0);
    double **T2 = init_matrix(Nintco,Nintco);
    opt_matrix_mult(C, 0, T, 0, T2, 0, Nintco, Nintco, Nintco, 0);
    double **T3 = symm_matrix_inv(T2, Nintco, 1);

    opt_matrix_mult( C, 0,  P, 0,  T, 0, Nintco, Nintco, Nintco, 0);
    opt_matrix_mult(T3, 0,  T, 0, T2, 0, Nintco, Nintco, Nintco, 0);
    free_matrix(T);
    opt_matrix_mult( C, 0, T2, 0, T3, 0, Nintco, Nintco, Nintco, 0);
    opt_matrix_mult( P, 0, T3, 0, T2, 0, Nintco, Nintco, Nintco, 0);
    free_matrix(T3);
    for (int i=0; i<Nintco; ++i)
      for (int j=0; j<Nintco; ++j)
        P[i][j] -= T2[i][j];
    free_matrix(T2);
  }
  free_matrix(C);
    """


    # Project redundancies out of forces.
    # fq~ = P fq
    fq[:] = np.dot(P, fq.T)

    if op.Params.print_lvl >= 3:
        print "\tInternal forces in au, after projection of redundancies and constraints."

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


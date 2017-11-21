from .bend import *
import numpy as np
from .printTools import printMat


# print the Hessian in common spectroscopic units of aJ/Ang^2, aJ/deg^2 or aJ/(Ang deg)
def show(H, intcos):
    Hscaled = np.zeros( H.shape, H.dtype)
    for i, row in enumerate(intcos):
        for j, col in enumerate(intcos):
            Hscaled[i,j] = H[i,j] * pc.hartree2aJ / row.qShowFactor / col.qShowFactor
    print_opt("Hessian in aJ/Ang^2, etc.\n")
    printMat(Hscaled)

def guess(intcos, geom, Z, connectivity = False, guessType="SIMPLE"):
    """ Generates diagonal empirical Hessians in a.u. such as 
      Schlegel, Theor. Chim. Acta, 66, 333 (1984) and
      Fischer and Almlof, J. Phys. Chem., 96, 9770 (1992).
    """
    dim = len(intcos)

    H = np.zeros( (dim,dim), float)
    for i,intco in enumerate(intcos):
        H[i,i] = intco.diagonalHessianGuess(geom, Z, connectivity, guessType)

    return H


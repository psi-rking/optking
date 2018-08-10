from __future__ import print_function
import numpy as np
import logging
# from sys import stdout

def printMatString(M, Ncol=7, title="\n"):
    """Formats a Matrix for Logging or Printing

    Parameters
    ----------
    M : ndarray, list

    title : string, optional
        title to include

    Returns
    -------
    string
        numpy array as string
    
    """
    #if title != "\n":
    #    title = title + "\n"
    #return np.array2string(M, max_line_width=100, precision=6, prefix=title, suffix="\n")
    s = '\t'
    if title:
        s += title + '\n\t'
    for row in range(M.shape[0]):
        tab = 0
        for col in range(M.shape[1]):
            tab += 1
            s += " %10.6f" % M[row, col]
            if tab == Ncol and col != (M.shape[1] - 1):
                s += '\n'
                tab = 0
        s += '\n\t'
    return s

def printArrayString(M, Ncol=7, title="\n"):
    """Formats Arrays for Logging or Printing

    Parameters
    ----------
    M : ndarray, list

    title="\n" : string
        optional title to include

    Returns
    -------
    string
        numpy array as string
    """
    #M = np.asarray(M)
    #if title != "\n":
    #    title = title + "\n"
    #return np.array2string(M, max_line_width=100, precision=6, prefix=title, suffix="\n")
    s = ''
    if title:
        s = title + '\n'
    tab = 0
    for i, entry in enumerate(M):
        tab += 1
        s += " %10.6f" % entry
        if tab == Ncol and i != (len(M) - 1):
           s += '\n'
           tab = 0
    s += '\n'
    return s


def printGeomGrad(geom, grad):
    logger = logging.getLogger(__name__)
    Natom = geom.shape[0]
    geometry_str = "\tGeometry\n\n"
    gradient_str = "\tGradient\n\n"
    for i in range(Natom):
        geometry_str += ("\t%20.10f%20.10f%20.10f\n" % (geom[i, 0], geom[i, 1], geom[i, 2]))
    geometry_str += ("\n")
    for i in range(Natom):
        gradient_str += ("\t%20.10f%20.10f%20.10f\n" % (grad[3 * i + 0], grad[3 * i + 1],
                                                        grad[3 * i + 2]))
    logger.info(geometry_str)
    logger.info(gradient_str)


def welcome():
    welcome_string = """
    \t\t\t-----------------------------------------\n
    \t\t\t OPTKING 3.0: for geometry optimizations \n
    \t\t\t     By R.A. King, Bethel University     \n
    \t\t\t        with contributions from          \n
    \t\t\t    A.V. Copan, J. Cayton, A. Heide      \n
    \t\t\t-----------------------------------------\n
    """
    return welcome_string

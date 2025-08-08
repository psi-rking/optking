from __future__ import print_function

import logging
from itertools import chain

import numpy as np
import qcelemental as qcel

# from sys import stdout


def print_mat_string(M, Ncol=7, title=None):
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
    # if title != "\n":
    #    title = title + "\n"
    # return np.array2string(M, max_line_width=100, precision=6, prefix=title, suffix="\n")
    s = "\t"
    if title != None:
        s += title + "\n\t"
    for row in range(M.shape[0]):
        tab = 0
        for col in range(M.shape[1]):
            tab += 1
            s += " %10.6f" % M[row, col]
            if tab == Ncol and col != (M.shape[1] - 1):
                s += "\n\t"
                tab = 0
        s += "\n\t"
    s = s[:-1]
    return s


def print_array_string(M, Ncol=7, title=None, form=":10.6f"):
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
    # M = np.asarray(M)
    # if title != "\n":
    #    title = title + "\n"
    # return np.array2string(M, max_line_width=100, precision=6, prefix=title, suffix="\n")
    s = "\t"
    if title != None:
        s += title + "\n\t\t"
    tab = 0
    formstring = "{" + form + "}"
    for i, entry in enumerate(M):
        tab += 1
        s += formstring.format(entry)
        if tab == Ncol and i != (len(M) - 1):
            s += "\n\t\t"
            tab = 0
    s += "\n"
    return s


def print_geom_grad(geom, grad):
    Natom = geom.shape[0]
    geometry = [f"{'Geometry (au)':>22}"]
    geometry += [
        "{:15.10f}{:15.10f}{:15.10f}".format(geom[i, 0], geom[i, 1], geom[i, 2])
        for i in range(Natom)
    ]

    gradient = [f"{' ':39}Gradient (au)\n"]
    gradient += [
        "\t\t{:15.10f}{:15.10f}{:15.10f}\n".format(
            grad[3 * i + 0], grad[3 * i + 1], grad[3 * i + 2]
        )
        for i in range(Natom)
    ]

    return "".join(list(chain(*zip(geometry, gradient))))


def print_geom_string(symbols, geom, unit=None):
    if unit == "Angstrom" or unit == "Angstroms":
        geom_str = "\n\tCartesian Geometry (in Angstroms)\n"
        for i in range(geom.shape[0]):
            geom_str += "\t%5s%20.10f%20.10f%20.10f\n" % (
                symbols[i],
                qcel.constants.bohr2angstroms * geom[i, 0],
                qcel.constants.bohr2angstroms * geom[i, 1],
                qcel.constants.bohr2angstroms * geom[i, 2],
            )
    else:
        geom_str = "\n\tCartesian Geometry\n"
        for i in range(geom.shape[0]):
            geom_str += "\t%5s%20.10f%20.10f%20.10f\n" % (
                symbols[i],
                geom[i, 0],
                geom[i, 1],
                geom[i, 2],
            )
    return geom_str


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

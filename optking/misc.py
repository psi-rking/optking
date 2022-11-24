import logging
import math

import numpy as np
import qcelemental as qcel

from .exceptions import OptError
from . import log_name

logger = logging.getLogger(f"{log_name}{__name__}")


def import_psi4(mesg=""):
    """Attempt psi4 import. Print mesg as indicator for why psi4 is required to user"""
    try:
        import psi4
    except ImportError as error:
        mesg = "Cannot import psi4" + mesg
        raise OptError(mesg + "conda install psi4 psi4-rt -c psi4") from error


def delta(i, j):
    if i == j:
        return 1
    else:
        return 0


def is_dq_symmetric(oMolsys, Dq):
    # TODO add symmetry check
    logger.debug("\tTODO add is_dq_symmetric\n")
    return True


def symmetrize_xyz(XYZ):
    # TODO add symmetrize function
    logger.debug("\tTODO add symmetrize XYZ\n")
    return XYZ


# "Average" bond length given two periods
# Values below are from Lindh et al.
# Based on DZP RHF computations, I suggest: 1.38 1.9 2.53, and 1.9 2.87 3.40
def average_r_from_periods(perA, perB):
    if perA == 1:
        if perB == 1:
            return 1.35
        elif perB == 2:
            return 2.1
        else:
            return 2.53
    elif perA == 2:
        if perB == 1:
            return 2.1
        elif perB == 2:
            return 2.87
        else:
            return 3.40
    else:
        if perB == 1:
            return 2.53
        else:
            return 3.40


# Return Lindh alpha value from two periods
def hguess_lindh_alpha(perA, perB):
    if perA == 1:
        if perB == 1:
            return 1.000
        else:
            return 0.3949
    else:
        if perB == 1:
            return 0.3949
        else:
            return 0.2800


# rho_ij = e^(alpha (r^2,ref - r^2))
def hguess_lindh_rho(ZA, ZB, RAB):
    perA = qcel.periodictable.to_period(ZA)
    perB = qcel.periodictable.to_period(ZB)

    alpha = hguess_lindh_alpha(perA, perB)
    r_ref = average_r_from_periods(perA, perB)

    return np.exp(-alpha * (RAB * RAB - r_ref * r_ref))


def tokenize_input_string(inString):
    """
    params: string of integers corresponding to internal coordinates
    returns: a list of integers correspoding to an atom
    removes spaces or non integer characters from string of internal coordinates to be frozen
    """
    outString = inString.replace("(", "").replace(")", "")
    return outString.split()


# Organize a single input list into a new list, each entry
# of which is Nint integers.
def int_list(inList, Nint=1):
    if len(inList) % Nint != 0:
        raise OptError("List does not have {}*n entries as expected".format(Nint))
    outList = []
    for i in range(0, len(inList), Nint):
        entry = []
        for I in range(Nint):
            entry.append(int(inList[i + I]))
        outList.append(entry)
    return outList


# Organize a single input list into a new list, each entry
# of which is Nint integers and Nfloat floats.
def int_float_list(inList, Nint=1, Nfloat=1):
    entry_length = Nint + Nfloat
    if len(inList) % entry_length != 0:
        raise OptError("List does not have {}*n entries as expected".format(entry_length))
    outList = []
    for i in range(0, len(inList), entry_length):
        entry = []
        for I in range(Nint):
            entry.append(int(inList[i + I]))
        for F in range(Nfloat):
            entry.append(float(inList[i + Nint + F]))
        outList.append(entry)
    return outList


# Organize a single input list into a new list, each entry
# of which is Nint integers, Nxyz lists (probably 1), and Nfloat floats
# e.g., ['2', 'xz', '1', '3'] => [2, ['x','z'], 1.0, 3.0]
def int_xyz_float_list(inList, Nint=1, Nxyz=1, Nfloat=1):
    entry_length = Nint + Nxyz + Nfloat
    if len(inList) % entry_length != 0:
        raise OptError("List does not have {}*n entries as expected".format(entry_length))
    outList = []
    for i in range(0, len(inList), entry_length):
        entry = []
        for I in range(Nint):
            entry.append(int(inList[i + I]))
        for X in range(Nxyz):
            cart_string = str(inList[i + Nint + X]).upper()
            if len(cart_string) > 3 or len(cart_string) < 1:
                raise OptError("Could not decipher xyz coordinate string")
            for c in cart_string:
                if c not in ("X", "Y", "Z"):
                    raise OptError("Could not decipher xyz coordinate string")
            cart_string = sorted(cart_string)  # x , then y, then z
            entry.append(cart_string)
        for F in range(Nfloat):
            entry.append(float(inList[i + Nint + Nxyz + F]))
        outList.append(entry)
    return outList


class string_math_fx(object):
    allowed_ops = {
        "sin": "math.sin",
        "cos": "math.cos",
        "log": "math.log",
        "ln": "math.log",
        "log10": "math.log10",
        "exp": "math.exp",
        "pow": "pow",
        "abs": "abs",
    }

    def __init__(self, formula_string_in):
        s = formula_string_in.lower()
        self.formula_string = s
        self.check_allowed_ops(s)
        self.make_fx(s)

    # Ensure nothing nefarious; only allowed functions.
    def check_allowed_ops(self, s):
        white = ["[", "(", "]", ")", "+", "-", "*", "/", "."]
        for c in s:
            if c in white:
                s = s.replace(c, " ")
        words = s.split()
        for w in words:
            if w.isdigit():
                # print('number')
                pass
            elif w in self.allowed_ops:
                # print('operation')
                pass
            elif w == "x":
                # print('variable x')
                pass
            else:
                raise Exception("Could not identify substring: {}".format(w))
        return

    def make_fx(self, s):
        for key, val in self.allowed_ops.items():
            s = s.replace(key, val)

        logger.info("make_fx: formula to evaluate: " + s)
        self.fx = eval("lambda x: " + s)

    def evaluate(self, x):
        return self.fx(x)


# Process input string (not yet tokenized) containing atoms, then formula.
# Formula will take x in Angstroms or degrees, return force in au
def int_fx_string(inString, Nint=1):
    if len(inString) == 0:
        return []
    logger.debug(inString)

    # split out the formulae string
    if "'" in inString:
        words = inString.split("'")
    elif '"' in inString:
        words = inString.split('"')
    else:
        raise OptError("Cannot find embedded string format function.")
    words = [w for w in words if len(w)]  # remove empty string at end

    if len(words) % 2 != 0:
        raise OptError("Input is not string of atoms containing string of formulae.")

    outList = []
    for i in range(0, len(words), 2):
        entry = []
        atoms = words[i].replace("(", "").replace(")", "").split(" ")  # allow ()
        atoms = [at for at in atoms if len(at)]  # remove empty string at end
        if len(atoms) != Nint:
            raise OptError("Not finding {:d} atoms for formula".format(Nint))

        for at in atoms:
            entry.append(int(at))
        entry.append(string_math_fx(words[i + 1]))

        outList.append(entry)
        logger.debug(entry)

    return outList


# Process input string (not yet tokenized), into entries in list,
# each of which has  [atom, [x,y,z] (probably only 1 of), formula]
# Formula may not contain spaces
def int_xyz_fx_string(inString, Nint=1):
    if len(inString) == 0:
        return []
    logger.debug(inString)

    # split out the formulae string
    if "'" in inString:
        words = inString.split("'")
    elif '"' in inString:
        words = inString.split('"')
    else:
        raise OptError("Cannot find embedded string format function.")
    words = [w for w in words if len(w)]  # remove empty string at end

    if len(words) % 2 != 0:
        raise OptError('Input string does not contain:  atom x/y/z "formulae".')

    outList = []
    for i in range(0, len(words), 2):
        entry = []
        cart = words[i].replace("(", "").replace(")", "").split(" ")  # allow ()
        cart = [at for at in cart if len(at)]  # remove empty string at end
        if len(cart) != Nint + 1:
            raise OptError("Not finding {:d} atom + x/y/z for formula".format(Nint))
        for at in cart[:-1]:
            entry.append(int(at))

        cart_string = cart[1].upper()
        if len(cart_string) > 3 or len(cart_string) < 1:
            raise OptError("Could not decipher xyz coordinate string")
        for c in cart_string:
            if c not in ("X", "Y", "Z"):
                raise OptError("Could not decipher xyz coordinate string")
        cart_string = sorted(cart_string)  # x , then y, then z
        entry.append(cart_string)

        entry.append(string_math_fx(words[i + 1]))

        outList.append(entry)
        logger.debug(entry)

    return outList

from math import fabs
import numpy as np

def delta(i, j):
    if i == j:
        return 1
    else:
        return 0

def isDqSymmetric(intcos, geom, Dq):
    print '\tTODO add isDqSymmetric'
    return True

def symmetrizeXYZ(XYZ):
    print '\tTODO add symmetrize XYZ'
    return XYZ

# return period from atomic number
def ZtoPeriod(Z):
    if   Z <=  2: return 1
    elif Z <= 10: return 2
    elif Z <= 18: return 3
    elif Z <= 36: return 4
    else:         return 5

# "Average" bond length given two periods
# Values below are from Lindh et al.
# Based on DZP RHF computations, I suggest: 1.38 1.9 2.53, and 1.9 2.87 3.40
def AverageRFromPeriods(perA, perB):
    if perA == 1:
        if   perB == 1:
            return 1.35
        elif perB == 2:
            return 2.1
        else:
            return 2.53
    elif perA == 2:
        if   perB == 1:
            return 2.1
        elif perB == 2:
            return 2.87
        else:
            return 3.40
    else:
        if   perB == 1:
            return 2.53
        else:
            return 3.40

# Return Lindh alpha value from two periods
def HguessLindhAlpha(perA, perB):
    if perA == 1:
        if (perB == 1):
            return 1.000
        else:
            return 0.3949
    else:
        if perB == 1:
            return 0.3949
        else:
            return 0.2800

# rho_ij = e^(alpha (r^2,ref - r^2))
def HguessLindhRho(ZA, ZB, RAB):
    perA = ZtoPeriod( ZA )
    perB = ZtoPeriod( ZB )

    alpha = HguessLindhAlpha(perA, perB)
    r_ref = AverageRFromPeriods(perA, perB)

    return np.exp(-alpha * (RAB*RAB - r_ref*r_ref))


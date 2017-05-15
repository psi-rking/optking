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
        if   perB == 1: return 1.35
        elif perB == 2: return 2.1
        else:           return 2.53
    elif perA == 2:
        if   perB == 1: return 2.1
        elif perB == 2: return 2.87
        else:           return 3.40
    else:
        if   perB == 1: return 2.53
        else:           return 3.40


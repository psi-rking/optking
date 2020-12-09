from math import acos, cos, sin, sqrt

import numpy as np

from optking import v3d

from .exceptions import AlgError

""" Tools for analytic rotation and orientation """


def rotate_vector(rot_axis, phi, v):
    """ rotate_vecs(): Rotate a set of vectors around an arbitrary axis

    Parameters
    ----------
    rot_axis : numpy array float[3]
        axis to rotate around - gets normalized here
    phi : float
        magnitude of rotation in radians
    v :  numpy array of floats
        n points to rotate ; overwritten on exit
        vectors are rows.

    Returns
    -------
    None, but v is overwritten with new vectors
    """
    # normalize rotation axis
    norm = sqrt(rot_axis[0] ** 2 + rot_axis[1] ** 2 + rot_axis[2] ** 2)
    rot_axis /= norm

    wx = rot_axis[0]
    wy = rot_axis[1]
    wz = rot_axis[2]
    cos_phi = cos(phi)
    sin_phi = sin(phi)
    cp = 1.0 - cos_phi

    R = np.ndarray((3, 3), float)
    R[0, 0] = cos_phi + wx * wx * cp
    R[0, 1] = -wz * sin_phi + wx * wy * cp
    R[0, 2] = wy * sin_phi + wx * wz * cp
    R[1, 0] = wz * sin_phi + wx * wy * cp
    R[1, 1] = cos_phi + wy * wy * cp
    R[1, 2] = -wx * sin_phi + wy * wz * cp
    R[2, 0] = -wy * sin_phi + wx * wz * cp
    R[2, 1] = wx * sin_phi + wy * wz * cp
    R[2, 2] = cos_phi + wz * wz * cp

    v_new = np.dot(R, v.T)  # vectors came in as rows!
    v[:] = v_new.T
    return


def zmat_point(A, B, C, R_CD, theta_BCD, phi_ABCD):
    """ zmat_point(): Given the xyz coordinates for three points and
        R, theta, and phi, as traditionally understood in a Z-matrix,
        returns the location of a 4th point.

    Parameters
    ----------
    A : numpy array float[3]
        Cartesian coordinates of atom A
    B : numpy array float[3]
        Cartesian coordinates of atom B
    C : numpy array float[3]
        Cartesian coordinates of atom C
    R_CD : float 
        Distance between atoms C and d
    theta_BCD :
        Angle between atoms B, C and d
    phi_ABCD :
        Dihedral Angle between atoms A, B, C and d

    Returns
    ----------
    d : numpy array float[3]
        Cartesian coordinates of atom d
    """

    eAB = v3d.eAB(A, B)  # vector A->B
    eBC = v3d.eAB(B, C)  # vector B->C
    cosABC = -v3d.dot(eAB, eBC)

    sinABC = sqrt(1 - (cosABC * cosABC))
    if (sinABC - 1.0e-14) < 0.0:
        raise AlgError("Z-matrix (reference) points cannot be colinear.")

    eY = v3d.cross(eAB, eBC) / sinABC
    eX = v3d.cross(eY, eBC)
    D = C + R_CD * (-eBC * cos(theta_BCD) + eX * sin(theta_BCD) * cos(phi_ABCD) + eY * sin(theta_BCD) * sin(phi_ABCD))
    return D


# axis  = np.array( [0,0,1],float )
# v     = np.array( [2,0,0],float )
# angle = acos(-1)
# print('v:', v)
# print('axis:',axis)
# print('angle: %15.10f' % angle)
# rotate_vector(axis, angle, v)
# print('v:', v)

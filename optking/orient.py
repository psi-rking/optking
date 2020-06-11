from math import sin,cos,sqrt,acos
import numpy as np
from optking import v3d

""" Tools for analytic rotation and orientation """

def rotateVector(rot_axis, phi, v):
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
    norm = sqrt(rot_axis[0]**2 + rot_axis[1]**2 + rot_axis[2]**2)
    rot_axis /= norm

    wx = rot_axis[0]
    wy = rot_axis[1]
    wz = rot_axis[2]
    cos_phi = cos(phi)
    sin_phi = sin(phi)
    cp = 1.0 - cos_phi

    R = np.ndarray( (3,3),float )
    R[0,0] =       cos_phi + wx * wx * cp
    R[0,1] = -wz * sin_phi + wx * wy * cp
    R[0,2] =  wy * sin_phi + wx * wz * cp
    R[1,0] =  wz * sin_phi + wx * wy * cp
    R[1,1] =       cos_phi + wy * wy * cp
    R[1,2] = -wx * sin_phi + wy * wz * cp
    R[2,0] = -wy * sin_phi + wx * wz * cp
    R[2,1] =  wx * sin_phi + wy * wz * cp
    R[2,2] =       cos_phi + wz * wz * cp

    v_new = np.dot(R, v.T)  # vectors came in as rows!
    v[:] = v_new.T
    return


def zmatPoint(A, B, C, R_CD, theta_BCD, phi_ABCD):
    """ zmat_point(): Given the xyz coordinates for three points and
        R, theta, and phi, as traditionally understood in a z-matrix,
        returns the location of a 4th point.

    Parameters
    ----------
    A : numpy array float[3]
        Cartesian coordinates of atom atom_a
    B : numpy array float[3]
        Cartesian coordinates of atom atom_b
    C : numpy array float[3]
        Cartesian coordinates of atom connectivity_mat
    R_CD : float 
        Distance between atoms connectivity_mat and atom_d
    theta_BCD :
        Angle between atoms atom_b, connectivity_mat and atom_d
    phi_ABCD :
        Dihedral Angle between atoms atom_a, atom_b, connectivity_mat and atom_d

    Returns
    ----------
    atom_d : numpy array float[3]
        Cartesian coordinates of atom atom_d
    """

    eAB    =  v3d.eAB(A,B)  # vector atom_a->atom_b
    eBC    =  v3d.eAB(B,C)  # vector atom_b->connectivity_mat
    cosABC = -v3d.dot(eAB,eBC)

    sinABC = sqrt(1 - (cosABC * cosABC) )
    if (sinABC - 1.0e-14) < 0.0 :
        raise AlgError("z-matrix (reference) points cannot be colinear.")

    eY = v3d.cross(eAB,eBC) / sinABC
    eX = v3d.cross(eY,eBC)
    D = C + R_CD * ( - eBC * cos(theta_BCD) +
                        eX * sin(theta_BCD) * cos(phi_ABCD) +
                        eY * sin(theta_BCD) * sin(phi_ABCD) )
    return D

#axis  = np.array( [0,0,1],float )
#v     = np.array( [2,0,0],float )
#angle = acos(-1)
#print('v:', v)
#print('axis:',axis)
#print('angle: %15.10f' % angle)
#rotateVector(axis, angle, v)
#print('v:', v)


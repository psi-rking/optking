import psi4
import optking
import qcelemental as qcel
au2kcal = qcel.constants.hartree2kcalmol
import numpy as np
import pytest
import matplotlib.pyplot as plt

# Potential energy scan with two benzenes.
benz_xyz = """
  C   -1.2025450   -0.6942916    0.0
  C   -1.2025450    0.6942916    0.0
  C    0.0000000   -1.3886100    0.0
  C   -0.0000000    1.3886100    0.0
  C    1.2025450   -0.6942916    0.0
  C    1.2025450    0.6942916    0.0
  H   -2.1397760   -1.2353675    0.0
  H   -2.1397760    1.2353675    0.0
  H    0.0000000   -2.4708011    0.0
  H   -0.0000000    2.4708011    0.0
  H    2.1397760   -1.2353675    0.0
  H    2.1397760    1.2353675    0.0
  --
  C   -1.2025450   -0.6942916   10.0
  C   -1.2025450    0.6942916   10.0
  C    0.0000000   -1.3886100   10.0
  C   -0.0000000    1.3886100   10.0
  C    1.2025450   -0.6942916   10.0
  C    1.2025450    0.6942916   10.0
  H   -2.1397760   -1.2353675   10.0
  H   -2.1397760    1.2353675   10.0
  H    0.0000000   -2.4708011   10.0
  H   -0.0000000    2.4708011   10.0
  H    2.1397760   -1.2353675   10.0
  H    2.1397760    1.2353675   10.0
"""
 
@pytest.mark.dimers
def test_dimers_benzene_pes():
    dimerMol = psi4.geometry(benz_xyz)

    Axyz = dimerMol.geometry().np[0:12,]
    Bxyz = dimerMol.geometry().np[12:,]

    # Define some reference atoms
    ArefAtoms = [ [0,1,2,3,4,5], # COM (between Carbon's)
                  [2], # carbon on x-axis
                  [1]] # another carbon
    BrefAtoms = [ [0,1,2,3,4,5], # COM (between Carbon's)
                  [2], # carbon on x-axis
                  [1]] # another carbon

    dimerCoord = optking.dimerfrag.DimerFrag(0, ArefAtoms, 1, BrefAtoms)

    # Here are the dimer coordinates defined, with their values in the
    # default of au or radians.:
    # name  = value         # description
    R       =   6.0         # Distance A1 to B1 (in this case, between COM's)
    theta_A =  np.pi/2      # Angle,          A2-A1-B1
    theta_B =  np.pi/2      # Angle,          A1-B1-B2
    tau     =   0.0         # Dihedral angle, A2-A1-B1-B2
    phi_A   =  np.pi/2      # Dihedral angle, A3-A2-A1-B1
    phi_B   = -np.pi/2      # Dihedral angle, A1-B1-B2-B3
    # To see starting values:
    # dimerCoord.update_reference_geometry(Axyz, Bxyz)
    #q = dimerCoord.q_array()
    #print(q)

    # Choose a theory.
    psi4.core.clean_options()
    psi4_options = { 'basis':'sto-3g', 'd_convergence':'9' }
    psi4.set_options(psi4_options)

    # Vary tau, spinning one benzene ring above another parallel one at a distance of 6
    # Angstroms.  Use degrees.
    R       =   3.0         # Distance A1 to B1 (in this case, between COM's)
    theta_A =  90.0
    theta_B =  90.0
    phi_A   =  90.0
    phi_B   = -90.0
    E_tau   = []
    for tau in range(0,181,30):
        q_target = np.array([R, theta_A, theta_B, tau, phi_A, phi_B])
        Bxyz[:] = dimerCoord.orient_fragment(Axyz, Bxyz, q_target, unit_angle='deg', unit_length='Angstrom')
        xyz = psi4.core.Matrix.from_array( np.concatenate( (Axyz,Bxyz) ) )
        dimerMol.set_geometry( xyz )
        E_tau.append( [tau, xyz.to_array(), psi4.energy('scf')] )

    E_tau = np.array(E_tau)
    Emin = min(E_tau[:,2])
    E_tau[:,2] = au2kcal*(E_tau[:,2] - Emin)

    print('Relative energy vs. tau')
    for pts in E_tau:
        print('{:10.3f}  {:10.3f}'.format(pts[0],pts[2]))

    # To get geometry at tau = 30.0
    # row = np.where(E_tau[:,0]==30.0)[0][0]
    # dimerMol.set_geometry( psi4.core.Matrix.from_array(E_tau[row,1]) )

    """ To plot
    fig = plt.figure()
    ax = fig.add_axes( [0.10,0.10,0.80,0.80] )
    ax.set_title('Benzene dimer')
    ax.set_xlabel('tau angle')
    ax.set_ylabel('Energy')
    ax.plot( E_tau[:,0], E_tau[:,2], 'g+-')
    plt.xlim(0,180)
    plt.show()
    """

    # Vary theta_A this time.  Start farther apart.
    R       =   6.0
    theta_B =  90.0
    tau     =   0.0
    phi_A   =  90.0
    phi_B   = -90.0
    E_theta_A   = []
    for theta_A in range(1,182,30):
        q_target = np.array([R, theta_A, theta_B, tau, phi_A, phi_B])
        Bxyz[:] = dimerCoord.orient_fragment(Axyz, Bxyz, q_target, unit_angle='deg', unit_length='Angstrom')
        xyz = psi4.core.Matrix.from_array( np.concatenate( (Axyz,Bxyz) ) )
        dimerMol.set_geometry( xyz )
        E_theta_A.append( [theta_A, xyz.to_array(), psi4.energy('scf')] )

    E_theta_A = np.array(E_theta_A)
    Emin = min(E_theta_A[:,2])
    E_theta_A[:,2] = au2kcal*(E_theta_A[:,2] - Emin)

    print('Relative energy vs. theta_A')
    for pts in E_theta_A:
        print('{:10.3f}  {:10.3f}'.format(pts[0],pts[2]))

    # To get Geometry at theta_A = 30.0
    # row = np.where(E_theta_A[:,0]==31.0)[0][0]
    # dimerMol.set_geometry( psi4.core.Matrix.from_array(E_theta_A[row,1]) )

    """ To plot
    fig = plt.figure()
    ax = fig.add_axes( [0.10,0.10,0.80,0.80] )
    ax.set_title('Benzene dimer')
    ax.set_xlabel('theta_A angle')
    ax.set_ylabel('Energy')
    ax.plot( E_theta_A[:,0], E_theta_A[:,2], 'g+-')
    plt.xlim(0,180)
    plt.show()
    """

    # Vary phi_A this time.  Start farther apart.
    R       =   6.0
    theta_A =  90.0
    theta_B =  90.0
    tau     =   0.0
    phi_B   = -90.0
    E_phi_A   = []
    for phi_A in range(0,181,30):
        q_target = np.array([R, theta_A, theta_B, tau, phi_A, phi_B])
        Bxyz[:] = dimerCoord.orient_fragment(Axyz, Bxyz, q_target, unit_angle='deg', unit_length='Angstrom')
        xyz = psi4.core.Matrix.from_array( np.concatenate( (Axyz,Bxyz) ) )
        dimerMol.set_geometry( xyz )
        E_phi_A.append( [phi_A, xyz.to_array(), psi4.energy('scf')] )

    E_phi_A = np.array(E_phi_A)
    Emin = min(E_phi_A[:,2])
    E_phi_A[:,2] = au2kcal*(E_phi_A[:,2] - Emin)

    print('Relative energy vs. phi_A')
    for pts in E_phi_A:
        print('{:10.3f}  {:10.3f}'.format(pts[0],pts[2]))

    # Geometry at phi_A = 30.0
    # row = np.where(E_phi_A[:,0]==30.0)[0][0]
    # dimerMol.set_geometry( psi4.core.Matrix.from_array(E_phi_A[row,1]) )

    """ To plot
    fig = plt.figure()
    ax = fig.add_axes( [0.10,0.10,0.80,0.80] )
    ax.set_title('Benzene dimer')
    ax.set_xlabel('phi_A angle')
    ax.set_ylabel('Energy')
    ax.plot( E_phi_A[:,0], E_phi_A[:,2], 'g+-')
    plt.xlim(0,180)
    plt.show()
    """

    ref_E_tau_60 = 0.343
    row = np.where(E_tau[:,0]==60.0)[0][0]
    assert psi4.compare_values(ref_E_tau_60, E_tau[row,2], 3, "Tau=60.0 Rel. Energy")

    ref_E_theta_A_91 = 0.147
    row = np.where(E_theta_A[:,0]==91.0)[0][0]
    assert psi4.compare_values(ref_E_theta_A_91, E_theta_A[row,2], 3, "Theta_A=91.0 Rel. Energy")

    ref_E_phi_A_91 = 0.131
    row = np.where(E_phi_A[:,0]==90.0)[0][0]
    assert psi4.compare_values(ref_E_phi_A_91, E_phi_A[row,2], 3, "Phi_A=90.0 Rel. Energy")


#! Test gradient and Hessian transformations
import psi4
import qcelemental as qcel
import numpy as np

import optking
from optking import bend, stre, tors

# psi4.core.set_output_file('psi-output.dat')


def test_stationary_forces_h2o():
    mol = psi4.geometry(
        """
        0 1
        O   0.0000000000  -0.0000000000   0.0025968676
        H   0.0000000000  -0.7487897072   0.5811909492
        H  -0.0000000000   0.7487897072   0.5811909492
        unit Angstrom
    """
    )

    psi4_options = {"basis": "cc-pvdz", "scf_type": "pk"}
    psi4.set_options(psi4_options)
    mol.update_geometry()
    Natom = mol.natom()
    xyz = mol.geometry().to_array()

    # Make an optking molecule manually, and you can choose your specific
    # desired internal coordinates.
    coords = [
        stre.Stre(0, 1),  # from-zero indexed atoms in geometry
        stre.Stre(0, 2),
        bend.Bend(0, 1, 2),
    ]
    Z = [mol.Z(i) for i in range(0, Natom)]
    masses = [mol.mass(i) for i in range(0, Natom)]
    f1 = optking.frag.Frag(Z, xyz, masses, intcos=coords, frozen=False)
    OptMol = optking.molsys.Molsys([f1])
    # psi4.core.print_out(str(OptMol))

    grad_x = psi4.gradient("hf")  # returns an (N,3) psi4 matrix
    # rms = grad_x.rms()
    # Print gradient as psi4 matrix.
    # psi4.core.print_out(f"Cartesian Gradient (RMS={rms:.3e})\n")
    # grad_x.print_out()
    # Print gradient as numpy array.
    # print(f"Cartesian Gradient (RMS={rms:.3e})")
    # print(grad_x.to_array())

    grad_x = grad_x.to_array()

    grad_q = OptMol.gradient_to_internals(grad_x.flatten(), use_masses=False)  # returns ndarray
    grad_x2 = OptMol.gradient_to_cartesians(grad_q).reshape(Natom, 3)
    grad_x2 = psi4.core.Matrix.from_array(grad_x2)
    # psi4.core.print_out("Internal Coordinate Gradient:\n"+str(grad_q)+"\n")
    # rms = grad_x2.rms()
    # psi4.core.print_out(f"Cartesian Gradient (RMS={rms:.3e})\n")
    # grad_x2.print_out()
    # print(f"Cartesian Gradient (RMS={rms:.3e})")
    # print(grad_x2.to_array())
    # rms_diff = np.sqrt(np.mean((grad_x - grad_x2)**2))
    # print(f"RMS diff gradient, cart->int->cart: {rms_diff:8.4e}")
    assert psi4.compare_values(grad_x, grad_x2, 10, "Diff grad. CART->int->CART")

    grad_q = OptMol.gradient_to_internals(grad_x.flatten(), use_masses=True)
    grad_x2 = OptMol.gradient_to_cartesians(grad_q).reshape(Natom, 3)
    grad_x2 = psi4.core.Matrix.from_array(grad_x2)
    assert psi4.compare_values(grad_x, grad_x2, 10, "Diff grad. CART->int->CART with u=1/mass")


def test_stationary_hessian_h2o():
    mol = psi4.geometry(
        """
        0 1
        O   0.0000000000  -0.0000000000   0.0025968676
        H   0.0000000000  -0.7487897072   0.5811909492
        H  -0.0000000000   0.7487897072   0.5811909492
        unit Angstrom
    """
    )

    psi4_options = {"basis": "cc-pvdz", "scf_type": "pk"}
    psi4.set_options(psi4_options)
    mol.update_geometry()
    Natom = mol.natom()
    xyz = mol.geometry().to_array()
    coords = [
        stre.Stre(0, 1),
        stre.Stre(0, 2),
        bend.Bend(0, 1, 2),
    ]
    Z = [mol.Z(i) for i in range(0, Natom)]
    masses = [mol.mass(i) for i in range(0, Natom)]
    f1 = optking.frag.Frag(Z, xyz, masses, intcos=coords, frozen=False)
    OptMol = optking.molsys.Molsys([f1])

    # Compute the Cartesian Hessian with psi4
    H_xy = psi4.hessian("hf")  # returns a (3N,3N) psi4 matrix
    # psi4.core.print_out("Calculated Cartesian hessian\n")
    # H_xy.print_out()

    # Transform hessian to internals with optking
    H_q = OptMol.hessian_to_internals(H_xy.to_array())  # returns ndarray
    # psi4.core.print_out(f"Hessian transformed into internal coordinates\n")
    # psi4.core.Matrix.from_array(H_q).print_out()

    ## Transform hessian to Cartesians with optking
    H_xy2 = OptMol.hessian_to_cartesians(H_q)  # returns ndarray
    # print("Hessian transformed back into Cartesian coordinates")
    # print(H_xy2)
    assert psi4.compare_values(H_xy, H_xy2, 7, "Diff hessian CART->int->CART")
    H_q = OptMol.hessian_to_internals(H_xy.to_array(), use_masses=True)
    H_xy2 = OptMol.hessian_to_cartesians(H_q)
    assert psi4.compare_values(H_xy, H_xy2, 7, "Diff hessian CART->int->CART with u=1/mass")


def test_nonstationary_forces_h2o():
    mol = psi4.geometry(
        """
        0 1
        O   0.0  -0.00  0.00
        H   0.0  -0.75  0.58
        H   0.0   0.75  0.58
        unit Angstrom
    """
    )

    psi4_options = {"basis": "cc-pvdz", "scf_type": "pk"}
    psi4.set_options(psi4_options)
    mol.update_geometry()
    Natom = mol.natom()
    xyz = mol.geometry().to_array()
    coords = [stre.Stre(0, 1), stre.Stre(0, 2), bend.Bend(0, 1, 2)]
    Z = [mol.Z(i) for i in range(0, Natom)]
    masses = [mol.mass(i) for i in range(0, Natom)]
    f1 = optking.frag.Frag(Z, xyz, masses, intcos=coords, frozen=False)
    OptMol = optking.molsys.Molsys([f1])

    grad_x = psi4.gradient("hf").to_array()

    grad_q = OptMol.gradient_to_internals(grad_x.flatten(), use_masses=False)
    grad_x2 = OptMol.gradient_to_cartesians(grad_q).reshape(Natom, 3)
    grad_x2 = psi4.core.Matrix.from_array(grad_x2)
    assert psi4.compare_values(grad_x, grad_x2, 10, "Diff grad. CART->int->CART")

    grad_q = OptMol.gradient_to_internals(grad_x.flatten(), use_masses=True)
    grad_x2 = OptMol.gradient_to_cartesians(grad_q).reshape(Natom, 3)
    grad_x2 = psi4.core.Matrix.from_array(grad_x2)
    assert psi4.compare_values(grad_x, grad_x2, 10, "Diff grad. CART->int->CART with u=1/mass")


def test_nonstationary_hessian_h2o():
    mol = psi4.geometry(
        """
        0 1
        O   0.0  -0.00  0.00
        H   0.0  -0.75  0.58
        H   0.0   0.75  0.58
        unit Angstrom
    """
    )

    psi4_options = {"basis": "cc-pvdz", "scf_type": "pk"}
    psi4.set_options(psi4_options)
    mol.update_geometry()
    Natom = mol.natom()
    xyz = mol.geometry().to_array()
    coords = [
        stre.Stre(0, 1),
        stre.Stre(0, 2),
        bend.Bend(0, 1, 2),
    ]
    Z = [mol.Z(i) for i in range(0, Natom)]
    masses = [mol.mass(i) for i in range(0, Natom)]
    f1 = optking.frag.Frag(Z, xyz, masses, intcos=coords, frozen=False)
    OptMol = optking.molsys.Molsys([f1])

    grad_x = psi4.gradient("hf").to_array().flatten()
    H_xy = psi4.hessian("hf")

    H_q = OptMol.hessian_to_internals(H_xy.to_array(), grad_x, use_masses=False)
    grad_q = OptMol.gradient_to_internals(grad_x, use_masses=False)
    H_xy2 = OptMol.hessian_to_cartesians(H_q, grad_q)
    assert psi4.compare_values(H_xy, H_xy2, 8, "Diff hessian CART->int->CART")

    H_q = OptMol.hessian_to_internals(H_xy.to_array(), grad_x, use_masses=True)
    grad_q = OptMol.gradient_to_internals(grad_x, use_masses=True)
    H_xy2 = OptMol.hessian_to_cartesians(H_q, grad_q)
    assert psi4.compare_values(H_xy, H_xy2, 8, "Diff hessian CART->int->CART with u=1/masses")


def test_stationary_forces_hooh():
    mol = psi4.geometry(
        """
        0 1
        H   0.9047154509   0.7748902860   0.4679224940
        O   0.1020360382   0.6887430144  -0.0294829672
        O  -0.1020360382  -0.6887430144  -0.0294829672
        H  -0.9047154509  -0.7748902860   0.4679224940
    """
    )

    mol.update_geometry()
    psi4_options = {"basis": "cc-pvdz", "scf_type": "pk"}
    psi4.set_options(psi4_options)
    xyz = mol.geometry().to_array()
    Natom = mol.natom()
    coords = [stre.Stre(0, 1), stre.Stre(2, 3), bend.Bend(0, 1, 2), bend.Bend(1, 2, 3), tors.Tors(0, 1, 2, 3)]
    Z = [mol.Z(i) for i in range(0, Natom)]
    masses = [mol.mass(i) for i in range(0, Natom)]
    f1 = optking.frag.Frag(Z, xyz, masses, intcos=coords, frozen=False)
    OptMol = optking.molsys.Molsys([f1])

    grad_x = psi4.gradient("hf").to_array()
    grad_q = OptMol.gradient_to_internals(grad_x.flatten(), use_masses=False)
    grad_x2 = OptMol.gradient_to_cartesians(grad_q).reshape(Natom, 3)
    assert psi4.compare_values(grad_x, grad_x2, 8, "Diff grad. CART->int->CART")

    grad_q = OptMol.gradient_to_internals(grad_x.flatten(), use_masses=True)
    grad_x2 = OptMol.gradient_to_cartesians(grad_q).reshape(Natom, 3)
    assert psi4.compare_values(grad_x, grad_x2, 8, "Diff grad. CART->int->CART with u=1/mass")


def test_stationary_hessian_hooh():
    mol = psi4.geometry(
        """
        0 1
        H   0.9047154509   0.7748902860   0.4679224940
        O   0.1020360382   0.6887430144  -0.0294829672
        O  -0.1020360382  -0.6887430144  -0.0294829672
        H  -0.9047154509  -0.7748902860   0.4679224940
    """
    )

    mol.update_geometry()
    Natom = mol.natom()
    psi4_options = {"basis": "cc-pvdz", "scf_type": "pk"}
    psi4.set_options(psi4_options)
    xyz = mol.geometry().to_array()
    coords = [stre.Stre(0, 1), stre.Stre(2, 3), bend.Bend(0, 1, 2), bend.Bend(1, 2, 3), tors.Tors(0, 1, 2, 3)]
    Z = [mol.Z(i) for i in range(0, Natom)]
    masses = [mol.mass(i) for i in range(0, Natom)]
    f1 = optking.frag.Frag(Z, xyz, masses, intcos=coords, frozen=False)
    OptMol = optking.molsys.Molsys([f1])

    H_xy = psi4.hessian("hf").to_array()

    H_q = OptMol.hessian_to_internals(H_xy, use_masses=False)
    H_xy2 = OptMol.hessian_to_cartesians(H_q)
    H_q2 = OptMol.hessian_to_internals(H_xy2, use_masses=False)
    assert psi4.compare_values(H_q, H_q2, 7, "Diff hessian cart->INT->cart->INT")

    # Cartesians do not agree due to implicit rotations
    # assert psi4.compare_values(H_xy, H_xy2, 7, "Diff hessian cart->int->cart")


def test_nonstationary_hessian_hooh():
    mol = psi4.geometry(
        """
        0 1
        H   0.90    0.77    0.46 
        O   0.10    0.68   -0.02 
        O  -0.10   -0.68   -0.02 
        H  -0.90   -0.77    0.46 
    """
    )

    mol.update_geometry()
    Natom = mol.natom()
    psi4_options = {"basis": "cc-pvdz", "scf_type": "pk"}
    psi4.set_options(psi4_options)
    xyz = mol.geometry().to_array()
    coords = [stre.Stre(0, 1), stre.Stre(2, 3), bend.Bend(0, 1, 2), bend.Bend(1, 2, 3), tors.Tors(0, 1, 2, 3)]
    Z = [mol.Z(i) for i in range(0, Natom)]
    masses = [mol.mass(i) for i in range(0, Natom)]
    f1 = optking.frag.Frag(Z, xyz, masses, intcos=coords, frozen=False)
    OptMol = optking.molsys.Molsys([f1])

    grad_x = psi4.gradient("hf").to_array().flatten()
    H_xy = psi4.hessian("hf").to_array()

    grad_q = OptMol.gradient_to_internals(grad_x)
    H_q = OptMol.hessian_to_internals(H_xy, grad_x)
    H_xy2 = OptMol.hessian_to_cartesians(H_q, grad_q)
    H_q2 = OptMol.hessian_to_internals(H_xy2, grad_x)
    assert psi4.compare_values(H_q, H_q2, 7, "Diff hessian cart->INT->cart->INT")

    # Cartesians do not agree due to implicit rotations
    # assert psi4.compare_values(H_xy, H_xy2, 7, "Diff hessian CART->int->CART")

import psi4
import optking
import pytest

#! Various constrained energy minimizations of HOOH with cc-pvdz RHF.
#! Cartesian-coordinate constrained optimizations of HOOH in Cartesians.
#! 1. Cartesian optimization.
#! 2. Cartesian optimization with frozen H's.
#! 3. Cartesian optimization with frozen O's.
HOOH_E             = -150.7866742 # TEST
HOOH_E_frozen_H_xyz = -150.7866491 # TEST
HOOH_E_frozen_O_xyz = -150.7866390 # TEST

f0 = ''''''
# Freeze H xyz in HOOH.
f1 = """ 1 Xyz 4 xYz """
# Freeze O xyz in HOOH.
f2 = """ 2 xyz 3 xyz """ 
# Freeze H xyz with individual input entries in HOOH.
f3 = """
     1 x
     1 y
     1 Z
     4 x
     4 Y
     4 z """

opt0 = {'frozen_cartesian': f0}
opt1 = {'frozen_cartesian': f1}
opt2 = {'frozen_cartesian': f2}
opt3 = {'frozen_cartesian': f3}
opt4 = {'frozen_cartesian': f1, 'opt_coordinates': 'redundant'}

freeze_params = [(opt0, HOOH_E), (opt1, HOOH_E_frozen_H_xyz), (opt2, HOOH_E_frozen_O_xyz), (opt3, HOOH_E_frozen_H_xyz), 
                 (opt4, HOOH_E_frozen_H_xyz)]

@pytest.mark.parametrize("options, expected", freeze_params, ids=["Only backstep", "freeze H", "freeze O", 
                                                                 "freeze individual x,y,z", "freeze then change coord"])
def test_hooh_freeze_xyz_Hs(options, expected):    

    hooh = psi4.geometry("""
      H  0.90  0.80  0.5
      O  0.00  0.70  0.0
      O  0.00 -0.70  0.0
      H -0.90 -0.80  0.5
    """)

    psi4.core.clean_options()   
    psi4_options = {
        'basis': 'cc-pvdz',
        'opt_coordinates': 'cartesian',
        'g_convergence': 'gau_tight',
        'geom_maxiter': 20,
        'consecutive_backsteps': 1
    }
    psi4.set_options(psi4_options)

    psi4.set_module_options("OPTKING", options)

    json_output = optking.optimize_psi4('hf')

    thisenergy = json_output['energies'][-1] #TEST
    assert psi4.compare_values(expected, thisenergy, 6)  #TEST

#! test if we can keep oxygen atom from moving off of the point (1,1,1)
def test_frozen_cart_h2o():

    h2o = psi4.geometry("""
        O   1.000000   1.000000   1.000000
        H   2.000000   1.000000   1.000000
        H   1.000000   2.000000   1.000000
        units angstrom
        no_com
        no_reorient
    """)

    psi4.core.clean_options()   
    psi4_options = {
        'basis': 'cc-pvdz',
        'reference': 'rhf',
        'scf_type': 'df',
        'max_energy_g_convergence': 7
    }
    psi4.set_options(psi4_options)
    psi4.set_module_options("OPTKING", {'frozen_cartesian': '''1 xyz'''} )

    json_output = optking.optimize_psi4('hf')

    thisenergy = json_output['energies'][-1]
    assert psi4.compare_values(-76.0270327834836, thisenergy, 6, "RHF Energy")
    assert psi4.compare_values( h2o.x(0), 1.88972613289, 6, "X Frozen coordinate")
    assert psi4.compare_values( h2o.y(0), 1.88972613289, 6, "Y Frozen coordinate")
    assert psi4.compare_values( h2o.z(0), 1.88972613289, 6, "Z Frozen coordinate")


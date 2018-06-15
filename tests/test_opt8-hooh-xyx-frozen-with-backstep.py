#! Various constrained energy minimizations of HOOH with cc-pvdz RHF.
#! Cartesian-coordinate constrained optimizations of HOOH in Cartesians.
#! 1. Cartesian optimization.
#! 2. Cartesian optimization with fixed H's.
#! 3. Cartesian optimization with fixed O's.

import psi4
import Psi4Opt

HOOH_E             = -150.7866742 # TEST
HOOH_E_fixed_H_xyz = -150.7866491 # TEST
HOOH_E_fixed_O_xyz = -150.7866390 # TEST

def test_hooh_full_opt():
    # Full optimization
    hooh = psi4.geometry("""
      H  0.90  0.80  0.5
      O  0.00  0.70  0.0
      O  0.00 -0.70  0.0
      H -0.90 -0.80  0.5
      no_com
      no_reorient
    """)
    
    psi4.set_options({
     'basis': 'cc-pvdz',
     'opt_coordinates': 'cartesian',
     'g_convergence': 'gau_tight',
     'geom_maxiter': 20,
     'consecutive_backsteps': 1  # TEST backsteps 
    })
    
    Psi4Opt.calcName = 'hf'
    thisenergy = Psi4Opt.Psi4Opt()
    assert psi4.compare_values(HOOH_E, thisenergy, 6, "Cart. Coord. RHF opt of HOOH, energy")  #TEST

def test_hooh_freeze_xyz_Hs():    
    # Freeze H xyz in HOOH.
    hooh = psi4.geometry("""
      H  0.90  0.80  0.5
      O  0.00  0.70  0.0
      O  0.00 -0.70  0.0
      H -0.90 -0.80  0.5
      no_com
      no_reorient
    """)
    
    psi4.set_options({
     'basis': 'cc-pvdz',
     'opt_coordinates': 'cartesian',
     'g_convergence': 'gau_tight',
     'geom_maxiter': 20,
     'consecutive_backsteps': 1  # TEST backsteps 
    })
    
    freeze_list = """ 1 Xyz 4 xYz """
    
    psi4.set_module_options('Optking', {'frozen_cartesian': freeze_list})
    #set optking frozen_cartesian $freeze_list
    
    Psi4Opt.calcName = 'hf'
    thisenergy = Psi4Opt.Psi4Opt()
    assert psi4.compare_values(HOOH_E_fixed_H_xyz, thisenergy, 6, 
                        "Cart. Coord. RHF opt of HOOH with H's xyz frozen, energy")  #TEST

def test_hooh_freeze_xyz_Os():    
    # Freeze O xyz in HOOH.
    hooh = psi4.geometry("""
      H  0.90  0.80  0.5
      O  0.00  0.70  0.0
      O  0.00 -0.70  0.0
      H -0.90 -0.80  0.5
      no_com
      no_reorient
    """)
    
    psi4.set_options({
     'basis': 'cc-pvdz',
     'opt_coordinates': 'cartesian',
     'g_convergence': 'gau_tight',
     'geom_maxiter': 20,
     'consecutive_backsteps': 1  # TEST backsteps 
    })
    
    
    
    freeze_list = """
      2 xyz
      3 xyz
    """
    
    
    psi4.set_module_options('Optking', {'frozen_cartesian': freeze_list})
    
    #set optking frozen_cartesian $freeze_list
    #what happens if we just dont set any options right here
    #if we dont give new options. the old options are still a global variable here.
    
    #reload(Psi4Opt)
    Psi4Opt.calcName = 'hf'
    thisenergy = Psi4Opt.Psi4Opt()
    assert psi4.compare_values(HOOH_E_fixed_O_xyz, thisenergy, 6, 
                              "Cart. Coord. RHF opt of HOOH with O's xyz frozen, energy")  #TEST

def test_hooh_individual_freezes_x_y_z():    
    # Freeze H xyz with individual input entries in HOOH.
    psi4.geometry("""
      H  0.90  0.80  0.5
      O  0.00  0.70  0.0
      O  0.00 -0.70  0.0
      H -0.90 -0.80  0.5
      no_com
      no_reorient
    """)
    
    psi4.set_options({
     'basis': 'cc-pvdz',
     'opt_coordinates': 'cartesian',
     'g_convergence': 'gau_tight',
     'geom_maxiter': 20,
     'consecutive_backsteps': 1  # TEST backsteps 
    })
    
    freeze_list = """
      1 x
      1 y
      1 Z
      4 x
      4 Y
      4 z
    """
    #set optking frozen_cartesian $freeze_list
    
    psi4.set_module_options('Optking', {'frozen_cartesian': freeze_list})
    
    Psi4Opt.calcName = 'hf'
    thisenergy = Psi4Opt.Psi4Opt()
    assert psi4.compare_values(HOOH_E_fixed_H_xyz, thisenergy, 6, "Cart. Coord. RHF opt of HOOH with H's x y z frozen, energy")  #TEST

def test_hooh_freeze_xyz_change_opt_coord():    
    # Freeze H xyz in HOOH.
    hooh = psi4.geometry("""
      H  0.90  0.80  0.5
      O  0.00  0.70  0.0
      O  0.00 -0.70  0.0
      H -0.90 -0.80  0.5
      no_com
      no_reorient
    """)
    
    psi4.set_options({
     'basis': 'cc-pvdz',
     'opt_coordinates': 'cartesian',
     'g_convergence': 'gau_tight',
     'geom_maxiter': 20,
     'consecutive_backsteps': 1  # TEST backsteps 
    })
    
    freeze_list = """
     1 xyz 
     4 xyz 
    """
    #set optking frozen_cartesian $freeze_list
    
    psi4.set_module_options('Optking', {'frozen_cartesian': freeze_list, 'opt_coordinates': 'redundant'})
    #set optking opt_coordinates redundant
    
    Psi4Opt.calcName = 'hf'
    thisenergy = Psi4Opt.Psi4Opt()
    assert psi4.compare_values(HOOH_E_fixed_H_xyz, thisenergy, 6, "Int. Coord. RHF opt of HOOH with H's xyz frozen, energy")  #TEST
    
#    # Freeze O xyz in HOOH.
#    hooh = psi4.geometry("""
#      H  0.90  0.80  0.5
#      O  0.00  0.70  0.0
#      O  0.00 -0.70  0.0
#      H -0.90 -0.80  0.5
#      no_com
#      no_reorient
#    """)
#    
#    psi4.set_options({
#     'basis': 'cc-pvdz',
#     'opt_coordinates': 'cartesian',
#     'g_convergence': 'gau_tight',
#     'geom_maxiter': 20,
#     'consecutive_backsteps': 1  # TEST backsteps 
#    })
#    
#    freeze_list = """
#     2 xyz 
#     3 xyz 
#    """
#    #set optking frozen_cartesian $freeze_list
#    
#    psi4.set_module_options('Optking', {'frozen_cartesian': freeze_list})
#    
#    Psi4Opt.calcName = 'hf'
#    thisenergy = Psi4Opt.Psi4Opt()
#    psi4.compare_values(HOOH_E_fixed_O_xyz, thisenergy, 6, "Int. Coord. RHF opt of HOOH with O's xyz frozen, energy")  #TEST
#

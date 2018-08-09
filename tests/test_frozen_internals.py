#! Various constrained energy minimizations of HOOH with cc-pvdz RHF
#! Internal-coordinate constraints in internal-coordinate optimizations.

import psi4
import optking

OH_frozen_stre_rhf       = -150.781130357 #TEST
OOH_frozen_bend_rhf      = -150.786372411 #TEST
HOOH_frozen_dihedral_rhf = -150.786766848 #TEST

def test_frozen_stre():
    # Constrained minimization with O-H bonds frozen.
    hooh = psi4.geometry("""
      H
      O 1 0.90
      O 2 1.40 1 100.0 
      H 3 0.90 2 100.0 1 115.0
    """)
   
    psi4options = {
      'diis': 'false',
      'basis': 'cc-PVDZ',
      'g_convergence': 'gau_verytight',
      'scf_type': 'pk',
    }
 
    psi4.set_options(psi4options)
    
    frozen_stre = ("""
        1  2
        3  4
    """)
    
    psi4.set_module_options('OPTKING', {'frozen_distance': frozen_stre, "g_convergence": "MOLPRO", "print": 4})
    json_output = optking.Psi4Opt('hf', psi4options)
    thisenergy = json_output['properties']['return_energy']
    assert psi4.compare_values(OH_frozen_stre_rhf, thisenergy, 7, 
                        "Int. Coord. RHF opt of HOOH with O-H frozen, energy")  #TEST

def test_frozen_bend():    
    # Constrained minimization with O-O-H angles frozen.
    hooh = psi4.geometry("""
      H
      O 1 0.90
      O 2 1.40 1 100.0
      H 3 0.90 2 100.0 1 115.0
    """)

    psi4options = {
      'diis': 'false',
      'basis': 'cc-PVDZ',
      'g_convergence': 'gau_verytight',
      'scf_type': 'pk',
    }

    psi4.set_options(psi4options)

    frozen_angles = ("""
        1 2 3
        2 3 4
    """)
    
    psi4.set_module_options('OPTKING', {'frozen_bend': frozen_angles, 'g_convergence': 'CFOUR', "print": 4}) 
    json_output = optking.Psi4Opt('hf', psi4options)
    thisenergy = json_output['properties']['return_energy']
    assert psi4.compare_values(OOH_frozen_bend_rhf, thisenergy, 7,
                        "Int. Coord. RHF opt of HOOH with O-O-H frozen, energy") #TEST

def test_frozen_tors():    
    # Constrained minimization with H-O-O-H dihedral frozen.
    hooh = psi4.geometry("""
      H
      O 1 0.90
      O 2 1.40 1 100.0 
      H 3 0.90 2 100.0 1 115.0
    """)

    psi4options = {
      'diis': 'false',
      'basis': 'cc-PVDZ',
      'g_convergence': 'gau_verytight',
      'scf_type': 'pk',
      'consecutive_backsteps': 1
    }

    psi4.set_options(psi4options)

    frozen_tors = ("1 2 3 4")
    
    psi4.set_module_options('OPTKING', {'frozen_dihedral': frozen_tors, 'g_convergence': "gau_verytight", "print": 4}) 
    json_output = optking.Psi4Opt('hf', psi4options)
    thisenergy = json_output['properties']['return_energy']
    assert psi4.compare_values(HOOH_frozen_dihedral_rhf, thisenergy, 7, 
                        "Int. Coord. RHF opt of HOOH with H-O-O-H frozen, energy") #TEST


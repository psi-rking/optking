#  IRC for HOOH from cis confirmation.
import psi4
import optking
psi4.set_memory('2 GB')

def test_hooh_irc():
    energy_5th_IRC_pt = -150.812913276783  # TEST
    h2o2 = psi4.geometry("""
      H     0.0000000000   0.9803530335  -0.8498671785
      O     0.0000000000   0.6988545188   0.0536419016
      O     0.0000000000  -0.6988545188   0.0536419016
      H     0.0000000000  -0.9803530335  -0.8498671785
      noreorient
    """)
    # Necessary since IRC will break C2h.
    h2o2.reset_point_group('c2')
    
    psi4_options = { 'basis': 'dzp',
                     'scf_type': 'pk' }
    
    psi4.set_options(psi4_options)
    
    psi4.set_module_options( "OPTKING", {
      "g_convergence": "gau_verytight",
      'opt_type': 'irc',
      'geom_maxiter': 20,
      #'irc_direction': "backward"
      } )
    
    json_output = optking.Psi4Opt('hf', psi4_options)
    
    IRC = json_output['properties']['IRC']
    
    print("%15s%15s%20s%15s" % ('Step Number', 'Arc Distance', 'Energy', 'HOOH dihedral'))
    for step in IRC:
        print("%15d%15.5f%20.10f%15.5f" % (step['Step Number'], step['Arc Distance'],
                                           step['Energy'],      step['Intco Values'][5] ))

    assert psi4.compare_values(energy_5th_IRC_pt, IRC[5]['Energy'], 6, "Energy of 5th IRC point.") #TEST



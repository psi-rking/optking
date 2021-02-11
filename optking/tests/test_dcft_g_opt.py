import psi4
import optking

#! DC-06 calculation for the O2 molecule (triplet ground state). This performs 
#! geometry optimization using two-step and simultaneous solution of the 
#! response equations for the analytic gradient.
def test_dcft_O2():
    o2 = psi4.geometry("""
      0 3
      O
      O 1 R
      R = 1.230
    """)

    psi4.core.clean_options()
    psi4_options = {
      'r_convergence': 10,
      'algorithm': 'twostep',
      'response_algorithm': 'twostep',
      'basis': 'dzp',
      'max_disp_g_convergence': 1e-6,
      'rms_force_g_convergence':  1e-6,
      'max_energy_g_convergence': 1e-7,
      'reference': 'uhf',
      'dct_functional': 'dc-06'
    }
    psi4.set_options(psi4_options)

    result = optking.optimize_psi4('dct')

    this_uhf = result['trajectory'][-1]['properties']['scf_total_energy'] #TEST
    this_mp2 = result['trajectory'][-1]['properties']['mp2_total_energy'] #TEST
    this_dct = result['energies'][-1] #TEST
    REF_uhf    = -149.6520519320 #TEST
    REF_mp2    = -150.0109986566 #TEST
    REF_dct    = -150.0227937862 #TEST
    assert psi4.compare_values(REF_uhf, this_uhf, 6, "UHF Energy") #TEST
    assert psi4.compare_values(REF_mp2, this_mp2, 6, "MP2 Energy") #TEST
    assert psi4.compare_values(REF_dct, this_dct, 6, "DC-06 Energy (two-step response)") #TEST

    # Psi4 should test this; so optking shouldn't need to.
    # Now try alternative response
    #psi4.set_options( {'response_algorithm': 'simultaneous'} )
    #o2.R = 1.232
    #
    #result = optking.optimize_psi4('dct')
    #
    #this_uhf = result['trajectory'][-1]['properties']['scf_total_energy']
    #this_mp2 = result['trajectory'][-1]['properties']['mp2_total_energy']
    #this_dct = result['energies'][-1]
    #assert psi4.compare_values(REF_uhf, this_uhf, 6, "UHF Energy");
    #assert psi4.compare_values(REF_mp2, this_mp2, 6, "MP2 Energy");
    #assert psi4.compare_values(REF_dct, this_dct, 6, "DC-06 Energy (simultaneous response)");


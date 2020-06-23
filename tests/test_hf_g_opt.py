#! SCF STO-3G geometry optimzation, with Z-matrix input
import psi4
import optking

def test_hf_g_h2o():
    refnucenergy = 8.9064890670                                                                     #TEST
    refenergy = -74.965901192                                                                    #TEST
    
    h2o = psi4.geometry("""
         O
         H 1 1.0
         H 1 1.0 2 104.5
    """)

    psi4.core.clean_options()

    psi4_options = {
      'diis': False,
      'basis': 'sto-3g',
      'e_convergence': 10,
      'd_convergence': 10,
      'scf_type': 'pk'
    }    

    psi4.set_options(psi4_options)
    
    json_output = optking.optimize_psi4('hf')
    E = json_output['energies'][-1]
    nucenergy = json_output['trajectory'][-1]['properties']['nuclear_repulsion_energy']
    
    assert psi4.compare_values(refnucenergy, nucenergy, 3, "Nuclear repulsion energy")    #TEST
    assert psi4.compare_values(refenergy, E, 6, "Reference energy")                                #TEST

#! SCF cc-pVDZ geometry optimzation, Z-matrix input, tight convergence
def test_hf_g_h2o_tight():
    REF_energy = -76.027053513

    h2o = psi4.geometry("""
        O
        H 1 1.0
        H 1 1.0 2 104.5
    """)

    psi4.core.clean_options()

    psi4_options = {
        'diis': False,
        'basis': 'cc-pvdz',
        'e_convergence': 10,
        'd_convergence': 10,
        'g_convergence': 'gau_tight',
        'scf_type': 'pk'
    }

    psi4.set_options(psi4_options)

    json_output = optking.optimize_psi4('hf')
    E = json_output['energies'][-1]

    assert psi4.compare_values(REF_energy, E, 9, "RHF energy")


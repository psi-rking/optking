#! SCF STO-3G geometry optimzation, with Z-matrix input
import psi4
import optking

def test_hf_g_h2o():
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
      'scf_type': 'pk',
    }    
    psi4.set_options(psi4_options)
    
    json_output = optking.optimize_psi4('hf')

    E = json_output['energies'][-1] #TEST
    nucenergy = json_output['trajectory'][-1]['properties']['nuclear_repulsion_energy'] #TEST
    refnucenergy =   8.9064983474  #TEST
    refenergy    = -74.9659011923  #TEST
    assert psi4.compare_values(refnucenergy, nucenergy, 3, "Nuclear repulsion energy") #TEST
    assert psi4.compare_values(refenergy, E, 6, "Reference energy")                    #TEST


#! SCF cc-pVDZ geometry optimzation, Z-matrix input, tight convergence
def test_hf_g_h2o_tight():
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

    E = json_output['energies'][-1] #TEST
    REF_energy = -76.02705351276    #TEST
    assert psi4.compare_values(REF_energy, E, 8, "RHF energy") #TEST


def test_hf_g_h2o_large():
    h2o = psi4.geometry("""
        O
        H 1 1.0
        H 1 1.0 2 104.5
    """)

    psi4.core.clean_options()
    psi4_options = {
        'basis': 'cc-pvtz',
        'e_convergence': 10,
        'd_convergence': 10,
        'g_convergence': 'gau_tight',
        'scf_type': 'pk'
    }
    psi4.set_options(psi4_options)

    json_output = optking.optimize_psi4('hf')

    E = json_output['energies'][-1] #TEST
    REF_energy = -76.05776970191 #TEST
    assert psi4.compare_values(REF_energy, E, 8, "RHF energy") #TEST

#! SCF cc-pVDZ geometry optimzation of ketene, starting from bent structure
def test_hf_g_ketene():

    ketene = psi4.geometry("""
        0 1
        H
        C 1 1.1
        C 2 1.3 1 130.0
        H 2 1.1 3 110.0 1 180.0
        O 3 1.1 4 120.0 1 180.0
    """)

    psi4.core.clean_options()
    psi4_options = {
        'basis': 'cc-pvdz',
        'g_convergence': 'gau_tight',
        'scf_type': 'pk'
    }
    psi4.set_options(psi4_options)

    result = optking.optimize_psi4('scf')

    E = result['energies'][-1] #TEST
    REF_energy  = -151.7410313803 #TEST
    assert psi4.compare_values(REF_energy, E, 8, "RHF energy") #TEST


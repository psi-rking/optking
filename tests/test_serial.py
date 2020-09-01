#! optimization of water using optHelper and explicit loop.
#  1. this scheme could be used to modify gradient explicitly.
#     by modifying opt._gX
#  2. optking still has a module level parameters and history,
#       that could be eliminated, so not yet parallel safe.
#  3. Have not yet restarted using serialization, but should be close.
import psi4
import optking

def test_optHelper():
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
    
    opt = optking.optHelper('hf')
    opt.build_coordinates()
    opt.show()
    
    for step in range(30):
        opt.energy_gradient_hessian()
        opt.step()
        conv = opt.testConvergence()
        if conv == True:
            print("Job is converged.")
            print("Optimization SUCCESS:\n" + opt._history.summary_string())
            break
    else:
        print("Calculation did not converge.")
        print("Optimization FAILURE:\n" + opt._history.summary_string())
     
    json_output = opt.close()

    E = json_output['energies'][-1] #TEST
    nucenergy = json_output['trajectory'][-1]['properties']['nuclear_repulsion_energy'] #TEST
    refnucenergy =   8.9064983474  #TEST
    refenergy    = -74.9659011923  #TEST
    assert psi4.compare_values(refnucenergy, nucenergy, 3, "Nuclear repulsion energy") #TEST
    assert psi4.compare_values(refenergy, E, 6, "Reference energy")                    #TEST


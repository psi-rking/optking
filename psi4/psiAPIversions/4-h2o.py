#! SCF CC-PVTZ geometry optimzation, with Z-matrix input

finalEnergy = -76.05776970 #TEST
import psi4
import psi4optwrapper

h2o = psi4.geometry("""
 O
 H 1 1.0
 H 1 1.0 2 104.5
""")

psi4_options = {
  'basis': 'cc-pvtz',
  'e_convergence': '10',
  'd_convergence': '10',
  'scf_type': 'pk',  
}

psi4.set_options(psi4_options)
psi4.set_module_options('Optking', {'step_type': 'rfo', 'print': 3})
E, nuc_energy = psi4optwrapper.Psi4Opt('hf', psi4_options)
psi4.compare_values(finalEnergy, E, 6, "RFO Step Final Energy")                                #TEST

#h2o = psi4.geometry("""
# O
# H 1 1.0
# H 1 1.0 2 104.5
#""")
#
#psi4options = {
#  'basis': 'cc-pvtz',
#  'e_convergence': '10',
#  'd_convergence': '10',
#  'scf_type': 'pk'
#}  
#
#psi4.set_options(psi4options)
#psi4.set_module_options('Optking', {'step_type': 'nr'})
#E, nucenergy = psi4optwrapper.Psi4Opt('hf', psi4options)
#psi4.compare_values(finalEnergy, E, 6, "NR Step Final Energy")                                #TEST
#
#molecule h2o {
# O
# H 1 1.0
# H 1 1.0 2 104.5
#}
#
#psi4.set_options({
#  'basis': 'cc-pvtz',
#  'e_convergence': '10',
#  'd_convergence': '10',
#  'scf_type': 'pk',
#}
#
#psi4.set_module_options('Optking', {'step_type': 'SD'})
#
#Psi4Opt.calcName = 'hf'
#E = Psi4Opt.Psi4Opt()
#psi4.compare_values(finalEnergy, E, 6, "SD Step Final Energy")                                #TEST
#


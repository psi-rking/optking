#! SCF CC-PVDZ geometry optimzation, with Z-matrix input
import psi4
import runpsi4API

nucrefenergy =   38.06177      #TEST
refenergy = -150.786766850  #TEST

hooh = psi4.geometry("""
  H
  O 1 0.9
  O 2 1.4 1 100.0
  H 3 0.9 2 100.0 1 114.0
""")


psi4_options = { 'basis': 'cc-pvdz',
  'g_convergence': 'gau_tight',
  'scf_type': 'pk',
}

psi4.set_options(psi4_options)

this_energy, nuc_energy = runpsi4API.Psi4Opt('hf', psi4_options)

psi4.compare_values(nucrefenergy, nuc_energy, 4, "Nuclear repulsion energy")    #TEST
psi4.compare_values(refenergy, this_energy, 8, "Reference energy")                                #TEST


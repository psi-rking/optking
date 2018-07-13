#! SCF STO-3G geometry optimzation, with Z-matrix input
import psi4
# These values are from a tightly converged QChem run
refnucenergy = 8.9064890670                                                                     #TEST
refenergy = -74.965901192                                                                    #TEST

h2o = psi4.geometry("""
O
H 1 1.88972
H 1 1.88972 2 104.5
units bohr
""")

#h2o.set_units(psi4.core.GeometryUnits.Bohr)
psi4options = {
  'diis': False,
  'basis': 'sto-3g',
  'e_convergence': 10,
  'd_convergence': 10,
  'scf_type': 'pk'
}

psi4.set_options(psi4options)

import runpsi4API
thisenergy, nucenergy = runpsi4API.Psi4Opt('hf', psi4options)

psi4.compare_values(refnucenergy, nucenergy, 3, "Nuclear repulsion energy")    #TEST
psi4.compare_values(refenergy, thisenergy, 6, "Reference energy")  #TEST


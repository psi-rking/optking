#! SCF DZ allene geometry optimization, with Cartesian input, first in c2v symmetry,
#! then in Cs symmetry from a starting point with a non-linear central bond angle.

import psi4
import importlib
nucenergy =   59.2532646680161                                                                 #TEST
refenergy = -115.8302823663                                                                    #TEST

# starting point is D2d/c2v
allene = psi4.geometry("""
 H  0.0  -0.92   -1.8
 H  0.0   0.92   -1.8
 C  0.0   0.00   -1.3
 C  0.0   0.00    0.0
 C  0.0   0.00    1.3
 H  0.92  0.00    1.8
 H -0.92  0.00    1.8
""")

psi4.set_options({
  'basis': 'DZ',
  'e_convergence': 10,
  'd_convergence': 10,
  'scf_type': 'pk',
})

import Psi4Opt
Psi4Opt.calcName = 'hf'
thisenergy = Psi4Opt.Psi4Opt()

psi4.compare_values(nucenergy, allene.nuclear_repulsion_energy(), 2, "Nuclear repulsion energy")    #TEST
psi4.compare_values(refenergy, thisenergy, 6, "Reference energy")                                   #TEST

# central C-C-C bond angle starts around 170 degrees to test the dynamic addition
# of new linear bending coordinates, and the redefinition of dihedrals.
allene = psi4.geometry("""
 H  0.0  -0.92   -1.8
 H  0.0   0.92   -1.8
 C  0.0   0.00   -1.3
 C  0.0   0.10    0.0
 C  0.0   0.00    1.3
 H  0.92  0.00    1.8
 H -0.92  0.00    1.8
""")

#importlib.reload(Psi4Opt) This reload no longer is nessecary since Psi4Opt no longer contains any global variables
Psi4Opt.calcName = 'hf'
thisenergy = Psi4Opt.Psi4Opt()

psi4.compare_values(nucenergy, allene.nuclear_repulsion_energy(), 2, "Nuclear repulsion energy")    #TEST
psi4.compare_values(refenergy, thisenergy, 6, "Reference energy")                                   #TEST


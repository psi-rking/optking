#! water dimer
import psi4
import optking

refenergy = -149.9414053335

dimer = psi4.geometry("""
 0 1
  H   0.53297836  -1.10088263  -2.17523351
  O   0.67046349  -1.17150926  -0.32413149
  H  -0.87285505  -0.32827188   0.18313336
 --
 0 1
  H   -0.75224517  -2.04662631  -6.55895403
  O   -0.20225739  -0.70744543  -5.42642983
  H    0.62391765   0.47566674  -6.56493124
 nocom
 units au
""")

psi4_options = {
  'basis': 'sto-3g',
  'geom_maxiter':30,
  'frag_mode':'MULTI',
  'frag_ref_atoms': [ 
    [[2],[1],[3]], #atoms for reference pts. on frag1 
    [[2],[1],[3]]  #atoms for reference pts. on frag 2
   ]
}

psi4.set_options(psi4_options)
json_output = optking.optimize_psi4('hf')

print('Number of iterations: %5d' % len(json_output['energies']))
print('Start energy: %15.10f' % json_output['energies'][0])
print('Final energy: %15.10f' % json_output['energies'][-1])

assert psi4.compare_values(refenergy, json_output['energies'][-1], 6, "Reference energy")


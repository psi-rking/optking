#! SCF CC-PVDZ geometry optimzation, with Z-matrix input
#
#import psi4
#
#def test_1_hooh():
#    nucenergy =   38.06177      #TEST
#    refenergy = -150.786766850  #TEST
#    
#    hooh = psi4.geometry("""
#      H
#      O 1 0.9
#      O 2 1.4 1 100.0
#      H 3 0.9 2 100.0 1 114.0
#    """)
#    
#    
#    psi4.set_options({
#      'basis': 'cc-pvdz',
#      'g_convergence': 'gau_tight',
#      'scf_type': 'pk',
#    })
#    
#    import Psi4Opt
#    Psi4Opt.calcName = 'hf'
#    thisenergy = Psi4Opt.Psi4Opt()
#   
#    assert psi4.compare_values(nucenergy, hooh.nuclear_repulsion_energy(), 4, "Nuclear repulsion energy")    #TEST
#    assert psi4.compare_values(refenergy, thisenergy, 8, "Reference energy")

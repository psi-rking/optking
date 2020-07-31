# Constrained optimization of tyrosine side group with methylthiophene.
# The relative orientation of the two monomers is fixed.  The monomers
# themselves and the distance between them are optimized.
import psi4
import optking
import numpy as np
import pytest

@pytest.mark.dimers
def test_dimers_mt_tyr_frozen_orientation():
    # Starting at R ~ 5 Angstroms
    # Had to choose geometry that would converge with 321G in <10 min for
    # TravisCI.  This very truncated optimization doesn't converge perfectly.
    init_xyz = """
        C  -1.2590959511   0.5355429724   0.4279186757
        H  -0.6910257912   1.1718902366   1.0934105249
        C  -0.6617862995  -0.3502689598  -0.4129046767
        C  -2.6990432131   0.5491391782   0.3593951325
        H  -3.3210084944   1.1917446222   0.9665019526
        C  -3.2196342504  -0.3234697363  -0.5384170295
        H  -4.2513479106  -0.5175322554  -0.7646817036
        S  -1.9125603285  -1.2514578208  -1.3784809513
        C   0.7925639416  -0.6471958752  -0.5978475957
        H   1.3855548585   0.0171924708   0.0386364493
        H   1.1050383529  -0.4966388972  -1.6375706422
        H   1.0252450859  -1.6832079353  -0.3235361360
        --
        C  -6.5704021253   0.0046458790  -1.4917663224
        C  -6.5961441385  -0.0415685440   1.3282358390
        C  -6.7021466699   1.1890940261  -0.7669317571
        C  -6.4863845709  -1.2105824958  -0.8023464841
        C  -6.4988249468  -1.2284077296   0.5864899947
        C  -6.7057800734   1.1624489957   0.6282194382
        H  -6.7821287197   2.1380218475  -1.2867282203
        H  -6.3988232778  -2.1213876069  -1.3796909994
        H  -6.4165976834  -2.1761604173   1.1067116140
        H  -6.7940837887   2.0928949711   1.1778141053
        O  -6.5235706161  -0.0509640100  -2.8760575285
        H  -6.5560764998   0.8664318319  -3.2520857835
        C  -6.5629804123  -0.0749551624   2.8430616051
        H  -7.3214118106  -0.7608362302   3.2373319981
        H  -6.7523147846   0.9193723913   3.2586831988
        H  -5.5827540136  -0.4133299217   3.2002276077
    nocom
    unit angstrom
    """
    # Note that nocom is needed so psi4 does not move the fragment COM's.
    
    # Define the reference atoms for each fragment, as a linear combination
    # of the positions of one or more atoms of the fragment.
    # If no weights are given, then the atoms are equally weighted.
    MTdimer = {
      'Natoms per frag': [12, 16],
      'A Frag'     : 1,
      'A Ref Atoms': [[1,3,4,6,8], [8], [11]],
      'A Label'    : 'methylthiophene',
      'B Frag'     : 2,
      'B Ref Atoms': [[13,14,15,16,17,18], [13], [15]],
      'B Label'    : 'tyrosine',
      'Frozen'     : ['theta_A','theta_B','tau','phi_A','phi_B']
    }
    # Here are the dimer coordinates that are used with their definitions.
    #  R         : Distance A1 to B1
    #  theta_A   : Angle,          A2-A1-B1
    #  theta_B   : Angle,          A1-B1-B2
    #  tau       : Dihedral angle, A2-A1-B1-B2
    #  phi_A     : Dihedral angle, A3-A2-A1-B1
    #  phi_B     : Dihedral angle, A1-B1-B2-B3
    
    # Build the psi4 molecule.
    MTmol = psi4.geometry(init_xyz)
    
    # To see the values of the interfragment coordinates, do this:
    #MTdimerCoord = optking.dimerfrag.DimerFrag.fromUserDict(MTdimer)
    #Axyz = MTmol.geometry().np[0:12,]
    #Bxyz = MTmol.geometry().np[12:,]
    #MTdimerCoord.update_reference_geometry(Axyz, Bxyz)
    #print( MTdimerCoord )
    #quit()
    
    # Choose a theory
    psi4.set_memory('4000.0 MB')
    psi4.core.clean_options()
    psi4_options = {
    #  'basis':'6-31G(d)',
      'basis':'3-21g',
      'd_convergence': 9,
      'frag_mode':'multi',
    }
    psi4.set_options(psi4_options)
    
    # For the moment, 'interfrag_coords' is a non-standard keyword and so
    # must be passed like this.
    # Optimize fragments and R but not interfragment angular coordinates.
    result = optking.optimize_psi4('b3lyp-d3mbj', XtraOptParams={'interfrag_coords': str(MTdimer)})
    
    E = result['energies'][-1]

    REF_631Gd_Energy = -939.169521
    REF_321G_Energy  = -934.237170
    assert psi4.compare_values(REF_321G_Energy, E, 4, "B3LYP-D3MBJ energy")  #TEST
    

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
    # TravisCI
    init_xyz = """
        C  -1.2593016170   0.5371998665   0.4278220579
        H  -0.6930240986   1.1743120734   1.0942743459
        C  -0.6604755174  -0.3481772244  -0.4112260179
        C  -2.6997212225   0.5501047727   0.3585876580
        H  -3.3215344864   1.1922854537   0.9663104893
        C  -3.2194057853  -0.3242524060  -0.5373030400
        H  -4.2506132487  -0.5212710107  -0.7635152265
        S  -1.9117745434  -1.2493145842  -1.3810959219
        C   0.7930615698  -0.6479331771  -0.5977352635
        H   1.3884369925   0.0141883931   0.0389407023
        H   1.1041610825  -0.4965544502  -1.6376797646
        H   1.0230908744  -1.6848497066  -0.3249560190
        --
        C  -6.5775144152   0.0093354912  -1.4944498683
        C  -6.6056046122  -0.0409951376   1.3252015873
        C  -6.7094268461   1.1925065106  -0.7681832705
        C  -6.4946788206  -1.2072710171  -0.8069064653
        C  -6.5079074824  -1.2271325258   0.5816361330
        C  -6.7133867581   1.1638971278   0.6272409826
        H  -6.7877295734   2.1425527449  -1.2861751109
        H  -6.4063957501  -2.1169363052  -1.3859985785
        H  -6.4247194426  -2.1751912468   1.1010433289
        H  -6.7998879214   2.0940312984   1.1777473928
        O  -6.5319155689  -0.0447650257  -2.8789609183
        H  -6.5636764336   0.8730478324  -3.2536037126
        C  -6.5800606054  -0.0772890241   2.8402959767
        H  -7.3905200438  -0.7041052605   3.2298432782
        H  -6.6964168350   0.9281396562   3.2563720305
        H  -5.6303619453  -0.4891439265   3.2015664089
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

    REF_631Gd_Energy = -939.16952108
    REF_321G_Energy  = -934.23717038
    assert psi4.compare_values(REF_321G_Energy, E, 6, "B3LYP-D3MBJ energy")  #TEST
    

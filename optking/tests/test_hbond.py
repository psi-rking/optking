import pytest
import numpy as np

import optking.stre
import optking.bend
import optking.tors
import optking.intcosMisc
import optking.addIntcos
import optking.displace
import optking.optparams as op
import optking.caseInsensitiveDict


# stretch displacement
# bend displacement
# whether h bond should be detected
@pytest.mark.skip
@pytest.mark.parametrize('options, expected', [
 ((     0,        0), True),
 ((-0.375,        0), False),
 ((     0, -np.pi/8), False),
 ((   0.8,        0), False)])
def test_hydrogen_bonds(options, expected):

    # Manually construct water dimer in single frag mode
    # Tests fail when run with entire suite due to op.Params missing
    op.Params = op.OptParams(optking.caseInsensitiveDict.CaseInsensitiveDict({}))

    # Geometry in Bohr
    geom = np.array([[- 1.2824641813, - 2.3421356740, 0.0913301558],
                     [- 0.0314167971, - 3.6520400907, 0.0340038272],
                     [- 1.9858966815, - 2.4709212198, 1.7566192883],
                     [1.3419482749, 2.4585411267, - 0.1017720514],
                     [0.4210461244, 0.8871180358, - 0.1345980555],
                     [0.6522116213, 3.3884042683, - 1.4903045984]])

    Z = [8, 1, 1, 8, 1, 1]
    masses = [15.999, 1.008, 1.008, 15.999, 1.008, 1.008]

    s1 = optking.stre.Stre(0, 1)
    s2 = optking.stre.Stre(0, 2)
    s3 = optking.stre.Stre(0, 4)
    s4 = optking.stre.Stre(3, 4)
    s5 = optking.stre.Stre(3, 5)

    b1 = optking.bend.Bend(1, 0, 2)
    b2 = optking.bend.Bend(1, 0, 4)
    b3 = optking.bend.Bend(2, 0, 4)
    b4 = optking.bend.Bend(0, 4, 3)
    b5 = optking.bend.Bend(4, 3, 5)

    t1 = optking.bend.Bend(1, 0, 4, 3)
    t2 = optking.bend.Bend(2, 0, 4, 3)
    t3 = optking.bend.Bend(0, 4, 3, 5)

    intcos = [s1, s2, s3, s4, s5, b1, b2, b3, b4, b5, t1, t2, t3]

    o_frag = optking.frag.Frag(Z, geom, masses, intcos)

    dq = np.zeros(13)
    dq[2] = options[0]
    dq[8] = options[1]

    # Displace in Bohr and degrees. Do four smaller displacements
    optking.displace.displace_frag(o_frag, dq)
    o_frag.add_h_bonds()

    assert (optking.stre.HBond(0, 4) in o_frag.intcos) == expected

#test_hydrogen_bonds( (-0.5,0), True)


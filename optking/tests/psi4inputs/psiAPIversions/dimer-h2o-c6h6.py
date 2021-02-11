import psi4
import optking

refenergy = -303.25404554

dimer = psi4.geometry(
    """
0 1
 O -0.5026452583       -0.9681078610       -0.4772692868
 H -2.3292990446       -1.1611084524       -0.4772692868
 H -0.8887241813        0.8340933116       -0.4772692868
 --
0 1
 C    0.8853463281       -5.2235996493        0.5504918473
 C    1.8139169342       -2.1992967152        3.8040686146
 C    2.8624456357       -4.1143863257        0.5409035710
 C   -0.6240195463       -4.8153482424        2.1904642137
 C   -0.1646305764       -3.3031992532        3.8141619690
 C    3.3271056135       -2.6064153737        2.1669340785
 H    0.5244823836       -6.4459192939       -0.7478283184
 H    4.0823309159       -4.4449979205       -0.7680411190
 H   -2.2074914566       -5.7109913627        2.2110247636
 H   -1.3768100495       -2.9846751653        5.1327625515
 H    4.9209603634       -1.7288723155        2.1638694922
 H    2.1923374156       -0.9964630692        5.1155773223
 nocom
 units au
"""
)

psi4_options = {
    "basis": "sto-3g",
    "frag_mode": "MULTI",
    "frag_ref_atoms": [
        [[1], [2], [3]],  # reference atoms are O, H, H for h2o,frag1
        [[1, 2, 3, 4, 5, 6], [2], [6]],  # reference atoms are center of ring, C, C frag2
    ],
}
psi4.set_options(psi4_options)

json_output = optking.optimize_psi4("mp2")

print("Number of iterations: %5d" % len(json_output["energies"]))
print("Start energy: %15.10f" % json_output["energies"][0])
print("Final energy: %15.10f" % json_output["energies"][-1])

assert psi4.compare_values(refenergy, json_output["energies"][-1], 6, "Reference energy")

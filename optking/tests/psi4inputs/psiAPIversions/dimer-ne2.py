#! neon dimer; test interfragment coordinates for minimal system
import psi4
import optking

refenergy = -253.209145

dimer = psi4.geometry(
    """
 0 1
 Ne  0.0  0.0  0.0
 --
 0 1
 Ne  1.5  0.0  0.0
 nocom
 units au
"""
)

psi4_options = {
    "basis": "sto-3g",
    "geom_maxiter": 30,
    "frag_mode": "MULTI",
    "frag_ref_atoms": [
        [[1]],
        [[1]],
    ],  # atoms for reference pts. on frag1  # atoms for reference pts. on frag 2
    "g_convergence": "gau_tight",
}
psi4.set_options(psi4_options)

json_output = optking.optimize_psi4("hf")

print("\nNumber of iterations: %5d" % len(json_output["energies"]))
print("Start energy: %15.10f" % json_output["energies"][0])
print("Final energy: %15.10f" % json_output["energies"][-1])

assert psi4.compare_values(refenergy, json_output["energies"][-1], 6, "Reference energy")

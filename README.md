<p align="center">
<br>
<a href="https://travis-ci.org/psi-rking/optking"><img src="https://travis-ci.org/psi-rking/optking.svg?branch=master"></a>
<a href="https://codecov.io/gh/psi-rking/optking"> <img src="https://codecov.io/gh/psi-rking/optking/branch/master/graph/badge.svg" /></a>
<a href="https://opensource.org/licenses/BSD-3-Clause"><img src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg" /></a>
<br>
<a href="#"> <img src="https://img.shields.io/github/release/psi-rking/optking.svg" /></a>
<a href="#"> <img src="https://img.shields.io/github/commits-since/psi-rking/optking/latest.svg" /></a>
<a href="#"> <img src="https://img.shields.io/github/release-date/psi-rking/optking.svg" /></a>
<a href="#"> <img src="https://img.shields.io/github/commit-activity/y/psi-rking/optking.svg" /></a>
<a href="#"> <img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<br>
</p>


# optking
Python version of the PSI geometry optimization program by R.A. King

Optking works with the quantum chemistry code of your choice through MolSSI's QCEngine.

Optking is primarily written to perform minimizations but does have transition state and IRC capabilities as in the previous version of optking

## Installation
To do a local install of the Python directory,
```
pip install -e .
```

## Running
### QCSchema
Optking can be run through a MolSSI qcSchema by calling `optking.optimize_qcengine`

```
import qcelemental as qcel
# Basic OptimizationInput
input_dict = {
  "schema_name": "qcschema_optimization_input",
  "schema_version": 1,
  "keywords": {
    "program": "psi4"
  },
  "initial_molecule": {
  "geometry": [
    0.90,  0.80,  0.5,
    0.00,  0.70,  0.0,
    0.00, -0.70,  0.0,
    -0.90, -0.80,  0.5],
  "symbols": [
    "H",
    "O",
    "O",
    "H"
  ],
  },
  "input_specification": {
    "schema_name": "qcschema_input",
    "schema_version": 1,
    "driver": "gradient",
    "model": {
      "method": "HF",
      "basis": "sto-3g"
    },
    "keywords": {
      "soscf": True
    }
  }
}

input_Schema = qcel.models.OptimizationInput(**input_dict)  # Use model to fill out OptimizationInput
output_dict = optking.optimize_qcengine(input_dict)
```
Calling optking through QCEngine is not currently supported officially

### Psi4
Alternatively, old psi4 optimization inputs can be easily altered to work with the new optimizer by adding import optking and changing `psi4.optimize` to `optking.optimize_psi4` Please note that `psi4.optimize` currently calls the c++ optking optimizer but this will be changed in a future, undetermined, release. At that time calling `psi4.optimize` will be recommended.

```
import optking
molecule mol {
  pubchem:hydrogen peroxide
}

set {
  basis sto-3g
  g_convergence gau_tight
  e_convergence 1e-10
  soscf true
  intrafrag_step_limit_max 0.3
}

opt_result = optking.optimize_psi4('hf')
print(len(opt_result['energies']))
print(opt_result['energies'][-1])
```

## Testing
Testing is done through py.test module. To run the entire test suite execute
```
py.test -v
```

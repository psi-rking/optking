# wrapper class for optimize in order to get JSON input/output fully functioning.
import os
import json
import uuid
import logging
import optking

from .molsys import Molsys
from . import qcdbjson


def run_json(json_file):
    """ wrapper for run_json_dict to read json input file and create json output file 
    formattted according to the MolSSI QCSchema

    Parameters
    ----------
    json_file : file
        json input file: qc_schema_input

    Notes
    -----
    optimization summary is added to the original json input file.
    For more information about the optimization see the .out logging file
    """

    with open(json_file) as input_file:
        json_dict = json.load(input_file)
    
    json_out = run_json_dict(json_dict)
    
    with open(json_file, "r+") as input_file:
        input_file.seek(0)
        input_file.truncate()
        json.dump(json_out, input_file, indent=2)


def run_json_dict(json_dict):
    """Wrapper to optking.optimize() will perform an optimization based
    
    Paramters
    ---------
    json_dict: dict
        must comply with MolSSI qcSchema
    
    Returns
    -------

    dict
    
    """

    if json_dict['driver'] != "optimize":
        logger.error('optking is not meant to run this input please use your favorite QC program')
 
    o_json = qcdbjson.jsonSchema(json_dict)
    optking_options = o_json.find_optking_options()
    oMolsys = Molsys.from_JSON_molecule(json.dumps(json_dict['molecule']))
    json_output = optking.optimize(oMolsys, optking_options, o_json)

    return json_output

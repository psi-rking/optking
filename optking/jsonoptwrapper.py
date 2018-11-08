# wrapper class for optimize in order to get JSON input/output fully functioning.
import os
import json
import uuid
import logging
import optking

from .molsys import Molsys
from . import qcdbjson


def run_json(json_file):
    """ wrapper for running optking's optimize() with json input 

    Parameters
    ----------
    json_file : file
        json input file: qc_schema_input

    Notes
    -----
    optimization summary is added to the original json input file. Final optimized geometry [a0] 
    replaces the input geometry. For more information about the optimization see the .out logging
    file
    """

    logger = logging.getLogger(__name__)
    with open(json_file) as input_data:
        json_dict = json.load(input_data)
    if json_dict['driver'] != 'optimize':
        logger.error('optking is not meant to run this input please use your favorite QM' +
                     'program (psi4) ;)')
        quit()

    o_json = qcdbjson.jsonSchema(json_dict)
    optking_options = o_json.find_optking_options()
    oMolsys = Molsys.from_JSON_molecule(json.dumps(json_dict['molecule']))
    json_output = optking.optimize(oMolsys, optking_options, o_json)

    with open(json_file, "r+") as input_data:
        input_data.seek(0)
        input_data.truncate()
        json.dump(json_output, input_data, indent=2)


def run_json_dict(json_dict):
    """Wrapper to run_json to pass and return as dictionary"""

    tempfile = str(uuid.uuid4())
    with open(tempfile, 'w') as handle:
        json.dump(json_dict, handle)

    run_json(tempfile)

    with open(tempfile) as handle:
        json_output = json.load(handle)

    os.unlink(tempfile)

    return json_output

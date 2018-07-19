# wrapper class for optimize in order to get JSON input/output fully functioning.
import json
import logging
import optking

from molsys import Molsys
import qcdbjson


def run_optking_json(json_file):
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
        json_dict = json.load(input_data)
        json_dict.update(json_output)
        input_data.seek(0)
        input_data.truncate()
        json.dump(json_dict, input_data, indent=2)

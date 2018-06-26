#wrapper class for optimize in order to get JSON input/output fully functioning.
#overhaul to optimize may be required if this is just the way we want optimize to run
import json
import optking
import psi4

from optking import printTools
printTools.printInit(psi4.core.print_out)
from optking import molsys

def run_optking_json(json_file):
    
    with open(json_file) as input_data:
        json_dict = json.load(input_data) #this might actually want to be just  a load 
    if json_dict['driver'] != 'optimize': #or something similar
        print_out('optking is not meant to run this input Please use your favorite QM program (psi4) ;)')
        quit()
    print(json_dict)
    o_json = optking.qcdbjson.jsonSchema(json_dict)
    optking_options = o_json.find_optking_options()
    oMolsys = molsys.Molsys.from_JSON_molecule(json.dumps(json_dict['molecule']))

    return optking.optimize(oMolsys, optking_options, o_json) 

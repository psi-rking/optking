#JSON class for json_object storage and manipulation. also contains methods for interacting with json files
#but which are not part of the class
   
import numpy as np
import json
import copy
from . import printTools
from . import history
from . import optparams as op


class jsonSchema:
    """Class for handling molSSI json input/output. Designed for "schema_name": "qc_schema_input", "schema_version": 1
    optking does not store qc_schema_output files from QM programs. Once this class is used to make a jsonSchema object
    optking simply updates the object when requesting a calculation. 
    """
    
    def __init__(self, JSON_dict):
        if JSON_dict['schema_name'] == 'qc_schema_input':
            self.optking_json = copy.deepcopy(JSON_dict)
            self.optking_json['molecule']['fix_com'] = True
            self.optking_json['molecule']['fix_orientation'] = True
        else:
            raise ValueError("JSON file must match...")

    def __str__(self):
        """Returns a string representation of a jsonSchema, ie the current version of
        the original json input file"""        

        return str(self.optking_json) 
        
    def update_geom_and_driver(self, geom, driver='gradient'):
        """Updates the geometry and driver in optkings json dictionary in order to
        request a calculation be performed. Also politely requests psi4 not to reorient
        any coordinates
        """
        self.optking_json['molecule']['geometry'] = geom
        json_for_input = copy.deepcopy(self.optking_json)
        json_for_input['driver'] = driver

        return json_for_input
    
    def find_optking_options(self):
        """This is meant to look for any options specifically for optking in a qcdb format. I'm assumming a sub
        dictionary of optking options will be provided.  We'll see if someone who actually can
        make decisions wants to do something different
        """

        optking_options = {}
        if 'optimizer' in self.optking_json['keywords']:
            for i in self.optking_json['keywords']['optimizer']:
                optking_options[i] = self.optking_json['keywords']['optimizer'][i]         

        del self.optking_json['keywords']['optimizer'] 
        return optking_options

    def generate_json_output(self, geom):
        import os
        """Untested method for creating a JSON output file"""
        json_output = {'schema_name': 'qc_schema_output'}
        json_output['provenance'] = {'creator': 'optking', 'version': '3.0?', \
            'routine': 'runoptkingjson'} 
        json_output['return_result'] = to_JSON_geom(geom)
        json_output['success'] = 'true'
        json_output['properties'] = {'return_energy': history.oHistory[-1].E, \
                'nuclear_repulsion_energy': history.oHistory.nuclear_repulsion_energy}
        json_output['properties']['steps'] = history.oHistory.summary()
        return json_output

def to_JSON_geom(geom):
    """Takes in optkings molecular systems geometry and converts to a 1D list to can be appended to
    a JSON_file. Returns a string.
    """
    j_geom = []
    for i in geom.flat:
        j_geom.append(i)

    return j_geom #Do I actually want to return a string here??

def get_JSON_result(json_data, calc_type, nuc=False):
    """Reads in the properties list of a qc_schmea json output file to get any data needed in
    addition to the return_result field. Meant to be called by optimize.get_x()
    Input:
        json_data: qc json calculation output - type qc JSON format dict
        calc_type: driver from calculation (gradient, hessian, etc..) - type string
    returns: gradient and energy (+nuc), hessian, or energy (+ nuc) as dictated by calc_type
         
    """   
    if json_data['schema_name'] == 'qc_schema_output':
        if calc_type == 'gradient':
            return_result = json_data['return_result']
            return_energy = json_data['properties']['return_energy']
            if nuc == True:
                nuc_energy = json_data['properties']['nuclear_repulsion_energy']
                return return_energy, return_result, nuc_energy     
            return return_energy, return_result    
        elif calc_type == 'hessian':
            return_result = json_data['return_result']
            return return_result
        elif calc_type == 'energy':
            return_result = json_data['return_result']
            if nuc == True:
                return_nuc = json_data['properties']['nuclear_repulsion_energy']
                return return_result, return_nuc
            return return_result

def make_qcschema(geom, symbols, QM_method, basis, keywords):
    """Creates a qc_schmea input.
    Parameters -
        geom - cartesian geometry - type 1D list
        symbols - atomic symbols - type 1D list
        QM_method - type string
        basis - type string
        keywords - Python Dict of strings
    """ 
    qcschema = {"schema_name": "qc_schema_input", "schema_version": 1, "molecule": \
        {"geometry": geom, "symbols": symbols}, "driver": "", "model": 
        {"method": QM_method, "basis": basis}, "keywords": keywords}  
    
    return qcschema

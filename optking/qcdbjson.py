import copy
import math
import numpy as np

from . import history


class jsonSchema:
    """ An implementation of MolSSI's qc schema

    Parameters
    ----------
    JSON_dict : dict
        should match qcschema_input format. version 1

    """

    def __init__(self, JSON_dict):
        if JSON_dict['schema_name'] == 'qcschema_input':
            self.optking_json = copy.deepcopy(JSON_dict)
            self._original = copy.deepcopy(JSON_dict)
            self.optking_json['molecule']['fix_com'] = True
            self.optking_json['molecule']['fix_orientation'] = True
        else:
            raise ValueError("JSON file must match the qcschema_input")

    def __str__(self):
        return str(self.optking_json)

    def _get_original(self, geom, driver = 'gradient'):
        self._original['molecule']['geometry'] = self.to_JSON_geom(geom)
        return self._original

    def update_geom_and_driver(self, geom, driver='gradient'):
        """Updates jsonSchema for requesting calculation

        Parameters
        ----------
        geom : list of float
            cartesian geometry 1D list
        driver : str, optional
            deafult is gradient. Other options: hessian or energy

        Returns
        -------
        json_for_input : dict
        """
        self.optking_json['molecule']['geometry'] = geom
        json_for_input = copy.deepcopy(self.optking_json)
        json_for_input['driver'] = driver

        return json_for_input

    # TODO revist once options for optimizer is finalized
    def find_optking_options(self):
        """ Parse JSON dict for optking specific options"""

        if 'optimizer' in self.optking_json['keywords']:
            optking_options = self.optking_json['keywords']['optimizer']
            del self.optking_json['keywords']['optimizer']  # remove to preserve json file for QM
            return optking_options
        else:
            return {}

    # TODO turn off logging_file if using json
    # TODO error output to json_output file
    def generate_json_output(self, geom, g_x):
        """ Creates json style dictionary to summarize optimization

        Parameters
        ----------
        geom : ndarray
            (nat, 3) cartesian geometry

        Returns
        -------
        json_output : dict
        """
        json_output = {'schema_name': 'qcschema_output'}
        json_output['provenance'] = {'creator': 'optking', 'version': '3.0?',
                                     'routine': 'runoptkingjson'}
        json_output['return_result'] = {'geometry': self.to_JSON_geom(geom)}
        json_output['success'] = 'true'
        json_output['properties'] = {'return_energy': history.oHistory[-1].E,
                                     'nuclear_repulsion_energy':
                                         history.oHistory.nuclear_repulsion_energy}
        json_output['properties']['steps'] = history.oHistory.summary()
        json_output['return_result']['gradient'] = [i for i in g_x.flat]
        return json_output

    @staticmethod
    def to_JSON_geom(geom):
        """ Converts optking geom to list for JSON

        Parameters
        ----------
        geom : ndarray
            cartesian geometry

        Returns
        -------
        list
            1D geometry
        """
        j_geom = [i for i in geom.flat]
        return j_geom

    @staticmethod
    def get_JSON_result(json_data, driver, nuc=False):
        """ Parse JSON file from QM program for result of calculation

        Parameters
        ----------
        json_data : dict
        driver : str
            gradient, hessian, or energy
        nuc : boolean, optional
            return nuclar repulsion energy as well

        Returns
        -------
        return_result : float or ndarray
            float if energy. ndarray if gradient or hessian
        return_energy : float
        return_nuc : float
        """

        if json_data['schema_name'] == 'qcschema_output':
            if driver == 'gradient':
                return_result = np.asarray(json_data['return_result'])
                return_energy = json_data['properties']['return_energy']
            elif driver == 'hessian':
                return_result = np.asarray(json_data['return_result'])
                nat_3 = int(math.sqrt(len(return_result)))
                return_result.shape = (nat_3, nat_3)
            elif driver == 'energy':
                return_result = json_data['return_result']

            return_nuc = json_data['properties']['nuclear_repulsion_energy']
            if driver == 'gradient' and nuc:
                return return_energy, return_result, return_nuc
            elif driver == 'gradient':
                return return_energy, return_result
            elif nuc:
                return return_result, return_nuc
            else:
                return return_result

    @classmethod
    def make_qcschema(cls, geom, symbols, QM_method, basis, keywords):
        """ Creates a qcschema according to MolSSI qcschema_input version 1

        Parameters
        ----------
        geom : list of float
            cartesian geom (1D list)
        symbols : list of str
             atomic symbols (1D list)
        QM_method: str
        basis : str
        keywords : dict of str
            all options
        """
        qcschema = {"schema_name": "qcschema_input", "schema_version": 1, "molecule":
                    {"geometry": geom, "symbols": symbols, "fix_com": True,
                     "fix_orientation": True},
                    "driver": "", "model": {"method": QM_method, "basis": basis},
                    "keywords": keywords}

        return cls(qcschema)

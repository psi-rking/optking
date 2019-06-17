import logging.config
import logging
import os
import sys

from . import lj_functions
from . import loggingconfig
from .psi4optwrapper import Psi4Opt
from .jsonoptwrapper import run_json, run_json_dict
from .stre import Stre
from .bend import Bend
from .tors import Tors
from .oofp import Oofp
from .frag import Frag 
from .molsys import Molsys
from .history import History
from .qcdbjson import jsonSchema
from .displace import displace
from . import optparams as op
from .exceptions import OptError, AlgError
from .optimize import optimize

# this was on my TODO list, but now im not sure its working
op.Params = op.OptParams({})

try:
    with open(os.path.join(os.getcwd(), sys.argv[0][:-3]) + ".out", "r+") as output_file:
        print(os.path.join(os.getcwd(), sys.argv[0][:-3]) + ".out")
        output_file.seek(0)
        output_file.truncate()
except FileNotFoundError:
    pass

logging.config.dictConfig(loggingconfig.logging_configuration)
logger = logging.getLogger(__name__)

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

import logging.config
import logging
import os
import sys

from .optimize import optimize
from .optimize import get_gradient
from .optimize import get_hessian
from .optimize import get_energy
from . import lj_functions
from . import loggingconfig
from .psi4optwrapper import Psi4Opt
from .jsonoptwrapper import run_json_file, run_qcschema
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
from . import run_json

# this was on my TODO list, but now im not sure its working
op.Params = op.OptParams({})

with open(os.path.join(os.getcwd(), 'opt_log.out'), "w") as output_file:
    output_file.truncate(0)
    print(os.path.join(os.getcwd(), 'opt_log.out'))

logging.config.dictConfig(loggingconfig.logging_configuration)
logger = logging.getLogger(__name__)

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

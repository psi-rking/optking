import os, sys
import logging
import logging.config

from .optimize import optimize
from . import loggingconfig
from . import lj_functions
from .optwrapper import optimize_psi4, optimize_qcengine
#from .stre import Stre
#from .bend import Bend
#from .tors import Tors
#from .oofp import Oofp
from . import frag
#from .molsys import Molsys
#from .history import History
#from .displace import displace
#from .exceptions import OptError, AlgError

from . import optparams as op
op.Params = op.OptParams({})

with open(os.path.join(os.getcwd(), 'opt_log.out'), "w") as output_file:
    output_file.truncate(0)
    print(os.path.join(os.getcwd(), 'opt_log.out'))

logging.config.dictConfig(loggingconfig.logging_configuration)
logger = logging.getLogger(__name__)

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
_optking_provenance_stamp = {"creator": "optking", "routine": None, "version": __version__}

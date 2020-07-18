import os

import logging.config

from .optimize import optimize, make_internal_coords
from . import loggingconfig
from . import optparams as op
from ._version import get_versions
from .optimize import optimize
from .opt_helper import OptHelper
from . import lj_functions
from .optwrapper import optimize_psi4, optimize_qcengine

with open(os.path.join(os.getcwd(), 'opt_log.out'), "w") as output_file:
    output_file.truncate(0)
    print(os.path.join(os.getcwd(), 'opt_log.out'))

logging.config.dictConfig(loggingconfig.logging_configuration)
logger = logging.getLogger(__name__)

op.Params = op.OptParams({})

__version__ = get_versions()['version']
del get_versions
_optking_provenance_stamp = {"creator": "optking", "routine": None, "version": __version__}




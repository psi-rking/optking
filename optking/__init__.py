import os, sys
import logging
import logging.config
with open(os.path.join(os.getcwd(), 'opt_log.out'), "w") as output_file:
    output_file.truncate(0)
    print(os.path.join(os.getcwd(), 'opt_log.out'))

from . import loggingconfig
logging.config.dictConfig(loggingconfig.logging_configuration)
logger = logging.getLogger(__name__)

from . import optparams as op
op.Params = op.OptParams({})

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
_optking_provenance_stamp = {"creator": "optking", "routine": None, "version": __version__}

from .optimize import optimize
from . import lj_functions
from .optwrapper import optimize_psi4, optimize_qcengine


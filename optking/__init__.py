import logging
import os
import sys
from logging.config import dictConfig
from . import lj_functions, loggingconfig

if "psi4" in sys.argv[0] or "psi4" in sys._getframe(1).f_globals.get("__name__") or "psi4" in sys.modules:

    opt_log = logging.getLogger("psi4.optking")
    log_name = "psi4."
    opt_log.propagate = True
else:
    logger = logging.getLogger()
    dictConfig(loggingconfig.logging_configuration)
    logger = logging.getLogger(__name__)
    log_name = ""

from . import optparams as op
from ._version import get_versions
from .opt_helper import EngineHelper, CustomHelper
from .optimize import make_internal_coords, optimize
from .optwrapper import optimize_psi4, optimize_qcengine
from .stre import Stre
from .bend import Bend
from .tors import Tors
from .frag import Frag
from .molsys import Molsys
from .history import History

op.Params = op.OptParams({})

__version__ = get_versions()["version"]
del get_versions
_optking_provenance_stamp = {
    "creator": "optking",
    "routine": None,
    "version": __version__,
}

import logging
import os
import sys

from packaging.version import Version
from logging.config import dictConfig
from . import lj_functions, loggingconfig

loggers = [name for name in logging.root.manager.loggerDict]
if "psi4" in loggers:
    opt_log = logging.getLogger("psi4.optking")
    log_name = "psi4."
    opt_log.propagate = True
else:
    logger = logging.getLogger()
    dictConfig(loggingconfig.logging_configuration)
    logger = logging.getLogger(__name__)
    log_name = ""

from packaging.version import Version

import logging
from . import log_name

logger = logging.getLogger(f"{log_name}{__name__}")

# Allows modules to import op without having to check which optparams to import
try:
    import pydantic
except ImportError as e:
    logger.error("Could not import pydantic. Please install with pip or conda")
    raise e
else:
    if Version(pydantic.__version__) < Version("2"):
        from .v1 import optparams as op
    else:
        from .v2 import optparams as op
        logger.warning(
            "Some dependencies such as QCElemental have not yet finished migration to pydantic v2. "
            "If issues are encountered please downgrade pydantic or upgrade QCElemental as appropriate"
        )

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

__version__ = get_versions()["version"]
del get_versions
_optking_provenance_stamp = {
    "creator": "optking",
    "routine": None,
    "version": __version__,
}

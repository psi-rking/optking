import logging

from . import log_name
optimize_log = logging.getLogger(f"{log_name}{__name__}")

# We don't catch this one internallyclass OptFail(Exception):
class OptError(Exception):
    def __init__(self, mesg="None given", err_type="Not specified"):
        # optimize_log.critical("Error message: %s", mesg)
        # optimize_log.critical("OptError: Optimization has failed.")
        self.mesg = mesg
        self.err_type = err_type
        # Exception.__init__(self, mesg)


class AlgError(Exception):
    # maybe generalize later def __init__(self, *args, **kwargs):
    def __init__(self, mesg="None given", newLinearBends=None):
        optimize_log.error(f"AlgError: Exception created. Mesg: {mesg}")
        if newLinearBends:
            optimize_log.error("AlgError: New bends detected.\n")
        self.linearBends = newLinearBends
        self.mesg = mesg

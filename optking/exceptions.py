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
    def __init__(self, mesg="None given", new_linear_bends=[], new_linear_torsion=[], oofp_failures=[]):
        optimize_log.error(f"AlgError: Exception created. Mesg: {mesg}")
        if new_linear_bends:
            optimize_log.error("AlgError: Linear bends detected.\n%s", '\n'.join(map(str, new_linear_bends)))
        if new_linear_torsion:
            optimize_log.error("AlgError: Linear Torsions Detected\n%s", '\n'.join(map(str, new_linear_torsion)))
        if oofp_failures:
            optimize_log.error("AlgError: Linear Oofps Detected\n%s", '\n'.join(map(str, oofp_failures)))
        self.linear_bends = new_linear_bends
        self.linear_torsions = new_linear_torsion
        self.oofp_failures = oofp_failures
        self.mesg = mesg

import logging


# We don't catch this one internallyclass OptFail(Exception):
class OptFail(Exception):
    def __init__(self, mesg='None given'):
        optimize_log = logging.getLogger(__name__)
        optimize_log.critical('OptFail: Optimization has failed.')
        # Exception.__init__(self, mesg)


class AlgFail(Exception):
    # maybe generalize later def __init__(self, *args, **kwargs):
    def __init__(self, mesg='None given', newLinearBends=None):
        optimize_log = logging.getLogger(__name__)
        optimize_log.error('AlgFail: Exception created.\n')
        if newLinearBends:
            optimize_log.error('AlgFail: New bends detected.\n')
        self.linearBends = newLinearBends
        self.mesg = mesg

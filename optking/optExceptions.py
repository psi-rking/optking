from printTools import print_opt

# We don't catch this one internallyclass OptFail(Exception):
class OptFail(Exception):
    def __init__(self, mesg='None given'):
        print_opt('OptFail: Optimization has failed.')
        #Exception.__init__(self, mesg)


class AlgFail(Exception):
    #maybe generalize later def __init__(self, *args, **kwargs):
    def __init__(self, mesg='None given', newLinearBends=None):
        print_opt('AlgFail: Exception created.\n')
        if newLinearBends:
            print_opt('AlgFail: New bends detected.\n')
        self.linearBends = newLinearBends
        self.mesg = mesg

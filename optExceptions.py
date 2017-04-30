

class BAD_STEP_EXCEPT(Exception):
    def __init__(self,mesg='None given'):
        print 'A bad step exception has been raised.'
        Exception.__init__(self, mesg)


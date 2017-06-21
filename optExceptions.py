

class BAD_STEP_EXCEPT(Exception):
    def __init__(self, mesg='None given'):
        print 'A BAD_STEP exception has been created.'
        Exception.__init__(self, mesg)


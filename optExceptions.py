from printTools import print_opt

class BAD_STEP_EXCEPT(Exception):
    def __init__(self, mesg='None given'):
        print_opt('A BAD_STEP exception has been created.\n')
        Exception.__init__(self, mesg)


from . import optParams as op

#Object class that stores the pivot point in internals, converged geometry, and overall stepNumber for the step
#Meant to be entered into a list to track the optimization

class IRCStep(object):
    #accepts 
    def __int__(self,p, xyzGeom, stepNumber):
        self.p = p
        self.geom = xyzGeom
        self.stepNumber = stepNumber

#Place holder for future way to check whether or not this step is more equivalent
#previous step may be better to perform this check in terms of internal coordinates
    def __eq__(self, other):
        if self.xyzGeom == other.xyzGeom
            return True
        else 
            return False

    def __repr__(self):
        print ("For step number: %d" (%self.stepNumber)
        print ("Pivot point is:")
        print (self.p)
        print ("Final cartesian geometry:")
        print (self.xyzGeom)            
        
    def __str__(self):
        print ("Final geometry for step %d is:" (% self.stepNumber))
        print (self.xyzGeom)
    
        if op.PrintLevel > 2:
            print ("IRC step length is: %d"  % self.stepLength)        

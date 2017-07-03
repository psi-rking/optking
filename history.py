import misc
import numpy as np
import optParams as op
import intcosMisc
from math import fabs
from linearAlgebra import absMax,rms,signOfDouble
from printTools import printMat,printMatString,printArrayString,print_opt

class STEP(object):
    def __init__(self,geom,E,forces):
        self.geom         = geom.copy()  # Store as 2D object
        self.E            = E
        self.forces       = forces.copy()
        self.projectedDE  = None
        self.Dq           = None
        self.followedUnitVector = None
        self.oneDgradient = None
        self.oneDhessian  = None

    def record(self,projectedDE,Dq,followedUnitVector,oneDgradient,oneDhessian):
        self.projectedDE  = projectedDE
        self.Dq           = Dq.copy()
        self.followedUnitVector       = followedUnitVector.copy()
        self.oneDgradient = oneDgradient
        self.oneDhessian  = oneDhessian

    def __str__(self):
        s = "Step Info\n"
        s += "Geometry     = \n"
        s += printMatString(self.geom)
        s += "Energy       = %15.10f\n" % self.E
        s += "forces       = "
        s += printArrayString(self.forces)
        s += "Projected DE = %15.10f\n" % self.projectedDE
        s += "Dq           = "
        s += printArrayString(self.Dq)
        s += "followedUnitVector       = "
        s += printArrayString(self.followedUnitVector)
        s += "oneDgradient = %15.10f\n" % self.oneDgradient
        s += "oneDhessian  = %15.10f\n" % self.oneDhessian
        return s

class HISTORY(object):
    stepsSinceLastHessian = 0
    consecutiveBacksteps = 0

    def __init__(self):
        self.steps = []
        HISTORY.stepsSinceLastHessian = 0

    def __str__(self):
        s = "History of length %d\n" % len(self)
        for i,step in enumerate(self.steps):
            s += "Step %d\n" % (i+1)
            s += step.__str__()
        return s

    def __len__(self):
        return len(self.steps)

    def __getitem__(self,index):
        return self.steps[index]

    def __setitem__(self,index,item):
        self.steps[index] = item

    def __delitem__(self,index):
        del self.steps[index]

    # Add new step.  We will store geometry as 1D in history.
    def append(self, geom, E, forces):
        s = STEP( geom, E, forces)
        self.steps.append(s)
        HISTORY.stepsSinceLastHessian += 1

    # Fill in details of new step.
    def appendRecord(self, projectedDE, Dq, followedUnitVector, oneDgradient,oneDhessian):
        self.steps[-1].record(projectedDE, Dq, followedUnitVector, oneDgradient, oneDhessian)

    def summary(self):
        print_opt("\n  ==> Optimization Summary <==\n\n")
        print_opt("  Measures of convergence in internal coordinates in au.\n")
        print_opt("  --------------------------------------------------------------------------------------------------------------- ~\n")
        print_opt("   Step         Total Energy             Delta E       MAX Force       RMS Force        MAX Disp        RMS Disp  ~\n")
        print_opt("  --------------------------------------------------------------------------------------------------------------- ~\n")

        for i in range(len(self.steps)):
      
            if i == 0: DE = self.steps[0].E
            else:      DE = self.steps[i].E - self.steps[i-1].E
      
            f = self.steps[i].forces
            max_force = absMax(f)
            rms_force = rms(f)
      
            # For the summary Dq, we do not want to +2*pi for example for the angles, so we read old Dq used during step.
            Dq = self.steps[i].Dq
            max_disp = absMax(Dq)
            rms_disp = rms(Dq)

            print_opt("   %4d %20.12lf  %18.12lf    %12.8lf    %12.8lf    %12.8lf    %12.8lf  ~\n" % ( (i+1), self.steps[i].E, \
              DE, max_force, rms_force, max_disp, rms_disp))
        print_opt("  --------------------------------------------------------------------------------------------------------------- ~\n\n")
  
    # Report on performance of last step
    # Eventually might have this function return False to reject a step
    def currentStepReport(self):
        print_opt("\tCurrent energy   : %20.10lf\n" % self.steps[-1].E)
  
        if len(self.steps) < 2:
            return True
  
        dontBackupYet = True if len(self.steps) < 5 else False

        energyChange    = self.steps[-1].E - self.steps[-2].E
        projectedChange = self.steps[-2].projectedDE
      
        print_opt("\tEnergy change for the previous step:\n")
        print_opt("\t\tProjected    : %20.10lf\n" % projectedChange)
        print_opt("\t\tActual       : %20.10lf\n" % energyChange)
      
        Energy_ratio = energyChange / projectedChange
      
        if op.Params.print_lvl >= 1:
            print_opt("\tEnergy ratio = %10.5lf\n" % Energy_ratio)
      
        if op.Params.opt_type == 'MIN':
            # Predicted up. Actual down.  OK.  Do nothing.
            if projectedChange > 0 and Energy_ratio < 0.0:
                return True
            # Actual step is  up.
            elif energyChange > 0:
                # Throw exception if bad step and not too close to start.
                if (op.Params.dynamic_level != 0) and not dontBackupYet:
                    raise BAD_STEP_EXCEPT("Energy has increased in a minimization.")
                # Not dynamic.  Do limited backsteps only upon request.  Otherwise, keep going.
                elif HISTORY.consecutiveBacksteps < op.Params.consecutiveBackstepsAllowed:
                    raise BAD_STEP_EXCEPT("Energy has increased in a minimization.")
                op.Params.decreaseTrustRadius()
            # Predicted down.  Actual down.
            elif Energy_ratio < 0.25:
                op.Params.decreaseTrustRadius()
            elif Energy_ratio > 0.75:
                op.Params.increaseTrustRadius()

        return True

    # Use History to update Hessian
    def hessianUpdate(self, H, intcos):
        if op.Params.hess_update == 'NONE': return
        print_opt("\tPerforming %s update.\n" % op.Params.hess_update)
        Nintco = len(intcos) # working dimension

        f = np.zeros(Nintco,float)
        x = np.zeros(self.steps[-1].geom.shape,float)
        q = np.zeros(Nintco, float)

        currentStep = self.steps[-1]
        f[:] = currentStep.forces
        x[:] = currentStep.geom
        q[:] = intcosMisc.qValues(intcos,x)

        # Fix configuration of torsions and out-of-plane angles,
        # so that Dq's are reasonable
        intcosMisc.updateDihedralOrientations(intcos, x)

        dq = np.zeros(Nintco, float)
        dg = np.zeros(Nintco, float)
        q_old = np.zeros(Nintco, float)

        # Don't go further back than the last Hessian calculation 
        numToUse = min( op.Params.hess_update_use_last, len(self.steps)-1, HISTORY.stepsSinceLastHessian )
        print_opt("\tUsing last %d steps for update.\n" % numToUse)

        # Make list of old geometries to update with.
        # Check each one to see if it is too close for stable denominators.
        checkStart = len(self.steps) - numToUse - 1
        use_steps = []
        for i in range(len(self.steps)-2,checkStart-1,-1):
            oldStep = self.steps[i]
            f_old = oldStep.forces
            x_old = oldStep.geom
            q_old[:] = intcosMisc.qValues(intcos, x_old)
            dq[:] = q - q_old
            dg[:] = f_old - f   # gradients -- not forces!
            gq = np.dot(dq, dg)
            qq = np.dot(dq, dq)

            # If there is only one left, take it no matter what.
            if len(use_steps) == 0 and i == (len(self.steps)-1): 
                use_steps.append(i)

            if fabs(gq) < op.Params.hess_update_den_tol or fabs(qq) < op.Params.hess_update_den_tol:
                print_opt("\tDenominators (dg)(dq) or (dq)(dq) are very small.\n")
                print_opt("\t Skipping Hessian update for step %d.\n" % (i+1))
                continue
  
            max_change = absMax(dq)
            if max_change > op.Params.hess_update_dq_tol:
                print_opt("\tChange in internal coordinate of %5.2e exceeds limit of %5.2e.\n" % \
                       (max_change, op.Params.hess_update_dq_tol))
                print_opt("\t Skipping Hessian update for step %d.\n" % (i+1))
                continue

            use_steps.append(i)

        print_opt("\tSteps to be used in Hessian update:\n\t")
        for i in use_steps:
          print_opt(" %d" % (i+1))
        print_opt("\n")

        H_new = np.zeros( H.shape, float)
        for i_step in use_steps:
            oldStep = self.steps[i_step]

            f_old = oldStep.forces
            x_old = oldStep.geom
            q_old[:] = intcosMisc.qValues(intcos, x_old)
            dq[:] = q - q_old
            dg[:] = f_old - f   # gradients -- not forces!
            gq = np.dot(dq, dg)
            qq = np.dot(dq, dq)

            # See  J. M. Bofill, J. Comp. Chem., Vol. 15, pages 1-11 (1994)
            #  and Helgaker, JCP 2002 for formula.
            if op.Params.hess_update == 'BFGS':
                for i in range(Nintco):
                    for j in range(Nintco):
                        H_new[i,j] = H[i,j] + dg[i] * dg[j] / gq
 
                Hdq = np.dot(H, dq)
                dqHdq = np.dot(dq, Hdq)

                for i in range(Nintco):
                    for j in range(Nintco):
                        H_new[i,j] -=  Hdq[i] * Hdq[j] / dqHdq
          
            elif op.Params.hess_update == 'MS':
                Z = -1.0 * np.dot(H, dq) + dg
                qz = np.dot(dq, Z)
          
                for i in range(Nintco):
                    for j in range(Nintco):
                        H_new[i,j] = H[i,j] + Z[i] * Z[j] / qz

            elif op.Params.hess_update == 'POWELL':
                Z = -1.0 * np.dot(H, dq) + dg
                qz = np.dot(dq, Z)

                for i in range(Nintco):
                    for j in range(Nintco):
                        H_new[i,j] = H[i,j] - qz/(qq*qq)*dq[i]*dq[j] + (Z[i]*dq[j] + dq[i]*Z[j])/qq
           
            elif op.Params.hess_update == 'BOFILL':
                #Bofill = (1-phi) * MS + phi * Powell
                Z = -1.0 * np.dot(H, dq) + dg
                qz = np.dot(dq, Z)
                zz = np.dot(Z, Z)
           
                phi = 1.0 - qz*qz/(qq*zz)
                if   phi < 0.0: phi = 0.0
                elif phi > 1.0: phi = 1.0
           
                for i in range(Nintco): # (1-phi)*MS
                    for j in range(Nintco):
                        H_new[i,j] = H[i,j] + (1.0-phi) * Z[i] * Z[j] / qz
           
                for i in range(Nintco): # (phi * Powell)
                    for j in range(Nintco):
                        H_new[i,j] += phi * (-1.0*qz/(qq*qq)*dq[i]*dq[j] + \
                                             (Z[i]*dq[j] + dq[i]*Z[j])/qq)

            if op.Params.hess_update_limit: # limit changes in H
                 # Changes to the Hessian from the update scheme are limited to the larger of
                 # (hess_update_limit_scale)*(the previous value) and hess_update_limit_max.
                 max_limit   = op.Params.hess_update_limit_max;
                 scale_limit = op.Params.hess_update_limit_scale;
                
                 # Compute change in Hessian
                 H_new[:,:] = H_new - H
               
                 for i in range(Nintco):
                   for j in range(Nintco):
                     val = fabs(scale_limit * H[i,j])
                     maximum = max(val, max_limit)
              
                     if fabs(H_new[i,j]) < maximum:
                         H[i,j] += H_new[i,j]
                     else: # limit change to max
                         H[i,j] += maximum * signOfDouble(H_new[i,j])

            else:  # only copy H_new into H
                H[:,:] = H_new
              
            H_new[:,:] = 0  # zero for next step
            # end loop over old geometries

        if op.Params.print_lvl >= 2:
            print_opt("Updated Hessian (in au)\n")
            printMat(H)
        return

History = HISTORY()


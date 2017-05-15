# Functions for step algorithms: Newton-Raphson, Rational Function Optimization,
# Steepest Descent. 
import numpy as np
from optParams import Params
from displace import displace
from intcosMisc import qShowForces
from addIntcos import linearBendCheck
from math import sqrt, fabs
from printTools import printArray, printMat
from misc import symmetrizeXYZ, isDqSymmetric
from linearAlgebra import absMax, symmMatEig, asymmMatEig, symmMatInv
import v3d
from history import History
import optExceptions

# This function and its components:
# 1. Computes Dq, the step in internal coordinates.
# 2. Calls displace and attempts to take the step.
# 3. Updates history with results.
def Dq(Molsys, E, qForces, H, stepType=None):
    if len(H) == 0 or len(qForces) == 0: return np.zeros( (0), float)

    if not stepType:
      stepType = Params.step_type

    if stepType == 'NR':
        return Dq_NR(Molsys, E, qForces, H)
    elif stepType == 'RFO':
        return Dq_RFO(Molsys, E, qForces, H)
    elif stepType == 'SD':
        return Dq_SD(Molsys, E, qForces)
    elif stepType == 'BACKSTEP':
        return Dq_BACKSTEP(Molsys) 
    else:
        raise ValueError('Dq: step type not yet implemented')

# Apply crude maximum step limit by scaling.
def applyIntrafragStepScaling(dq):
    trust = Params.intrafrag_trust
    if sqrt(np.dot(dq,dq)) > trust:
        scale = trust / sqrt(np.dot(dq,dq))
        print "\tStep length exceeds trust radius of %10.5f." % trust
        print "\tScaling displacements by %10.5f" % scale
        dq *= scale
    return

# Compute energy change along one dimension
def DE_projected(model, step, grad, hess):
    if model == 'NR':
        return (step * grad + 0.5 * step * step * hess)
    elif model == 'RFO':
        return (step * grad + 0.5 * step * step * hess)/(1 + step*step)
    else:
        raise ValueError("DE_projected does not recognize model.")

# geometry and E are just for passing
# at present we are not storing the ACTUAL dq but the attempted
def Dq_NR(intcos, geom, E, fq, H):
    print "\tTaking NR optimization step."

    # Hinv fq = dq
    Hinv = symmMatInv(H, redundant=True)
    dq = np.dot(Hinv, fq)

    # applies maximum internal coordinate change
    applyIntrafragStepScaling(dq);

    # get norm |q| and unit vector in the step direction
    nr_dqnorm = sqrt( np.dot(dq,dq) )
    nr_u = dq.copy() / nr_dqnorm
    print "\tNorm of target step-size %15.10f" % nr_dqnorm

    # get gradient and hessian in step direction
    nr_g = -1 * np.dot(fq, nr_u) # gradient, not force
    nr_h = np.dot( nr_u, np.dot(H, nr_u) )

    if Params.print_lvl > 1:
       print '\t|NR target step|: %15.10f' % nr_dqnorm
       print '\tNR_gradient     : %15.10f' % nr_g
       print '\tNR_hessian      : %15.10f' % nr_h
    DEprojected = DE_projected('NR', nr_dqnorm, nr_g, nr_h)
    print "\tProjected energy change by quadratic approximation: %20.10lf" % DEprojected

    # Scale fq into aJ for printing
    fq_aJ = qShowForces(intcos, fq)
    displace(intcos, geom, dq, fq_aJ)

    dq_actual = sqrt( np.dot(dq,dq) )
    print "\tNorm of achieved step-size %15.10f" % dq_actual

    # Symmetrize the geometry for next step
    # symmetrize_geom()

    # save values in step data
    History.appendRecord(DEprojected, dq, nr_u, nr_g, nr_h)

    # Can check full geometry, but returned indices will correspond then to that.
    linearList = linearBendCheck(Molsys.intcos, Molsys.geom, dq)
    if linearList:
        raise INTCO_EXCEPT("New linear angles", linearList)

    return dq


# Take Rational Function Optimization step
def Dq_RFO(Molsys, E, fq, H):
    print "\tTaking RFO optimization step."
    dim = len(fq)
    dq = np.zeros( (dim), float)         # To be determined and returned.
    trust = Params.intrafrag_trust  # maximum step size
    max_projected_rfo_iter = 25          # max. # of iterations to try to converge RS-RFO
    rfo_follow_root = Params.rfo_follow_root  # whether to follow root
    rfo_root = Params.rfo_root  # if following, which root to follow

    # Determine the eigenvectors/eigenvalues of H.
    Hevals, Hevects = symmMatEig(H)

    # Build the original, unscaled RFO matrix.
    RFOmat = np.zeros( (dim+1,dim+1), float)
    for i in range(dim):
        for j in range(dim):
            RFOmat[i,j] = H[i,j]
        RFOmat[i,dim]= RFOmat[dim,i] = -fq[i]

    if Params.print_lvl >= 4:
        print "Original, unscaled RFO matrix:"
        printMat(RFOmat)

    symm_rfo_step = False
    SRFOmat = np.zeros( (dim+1,dim+1), float) # For scaled RFO matrix.
    converged = False
    dqtdq = 10             # square of norm of step
    alpha = 1.0            # scaling factor for RS-RFO, scaling matrix is sI

    last_iter_evect = np.zeros( (dim), float)
    if rfo_follow_root and len(History.steps) > 1:
        last_iter_evect[:] = History.steps[-2].followedUnitVector  # RFO vector from previous geometry step

    # Iterative sequence to find alpha
    alphaIter = -1
    while not converged and alphaIter < max_projected_rfo_iter:
        alphaIter += 1

        # If we exhaust iterations without convergence, then bail on the
        #  restricted-step algorithm.  Set alpha=1 and apply crude scaling instead.
        if alphaIter == max_projected_rfo_iter:
            print "\tFailed to converge alpha. Doing simple step-scaling instead."
            alpha = 1.0
        elif Params.simple_step_scaling:
            # Simple_step_scaling is on, not an iterative method.
            # Proceed through loop with alpha == 1, and then continue
            alphaIter = max_projected_rfo_iter

        # Scale the RFO matrix.
        for i in range(dim+1):
            for j in range(dim):
                SRFOmat[j,i] = RFOmat[j,i] / alpha
            SRFOmat[dim,i] = RFOmat[dim,i]

        if Params.print_lvl >= 4:
            print "\nScaled RFO matrix."
            printMat(SRFOmat)

        # Find the eigenvectors and eigenvalues of RFO matrix.
        SRFOevals, SRFOevects = asymmMatEig(SRFOmat)

        if Params.print_lvl >= 4:
            print "Eigenvectors of scaled RFO matrix."
            printMat(SRFOevects)

        if Params.print_lvl >= 2:
            print "Eigenvalues of scaled RFO matrix."
            printArray(SRFOevals)
            print "First eigenvector (unnormalized) of scaled RFO matrix."
            printArray(SRFOevects[0])

        # Do intermediate normalization.  RFO paper says to scale eigenvector
        # to make the last element equal to 1. Bogus evect leads can be avoided
        # using root following.
        for i in range(dim+1):
            # How big is dividing going to make the largest element?
            # Same check occurs below for acceptability.
            if fabs(SRFOevects[i][dim]) > 1.0e-10:
                tval = absMax(SRFOevects[i]/ SRFOevects[i][dim])
                if tval < Params.rfo_normalization_max:
                    for j in range(dim+1):
                        SRFOevects[i,j] /= SRFOevects[i,dim]

        if Params.print_lvl >= 4:
            print "All scaled RFO eigenvectors (rows)."
            printMat(SRFOevects)

        # Use input rfo_root
        # If root-following is turned off, then take the eigenvector with the rfo_root'th lowest eigvenvalue.
        # If its the first iteration, then do the same.  In subsequent steps, overlaps will be checked.
        if not rfo_follow_root or len(History.steps) < 2:

            rfo_root = Params.rfo_root
            if alphaIter == 0:
                print "\tChecking RFO solution %d." % (rfo_root+1)

            for i in range(rfo_root, dim+1):
                # Check symmetry of root.
                dq[:] = SRFOevects[i,0:dim]
                if not Params.accept_symmetry_breaking:
                    symm_rfo_step = isDqSymmetric(Molsys.intcos, Molsys.geom, dq)
    
                    if not symm_rfo_step:  # Root is assymmetric so reject it.
                        if alphaIter == 0:
                            print "\tRejecting RFO root %d because it breaks the molecular point group."\
                                    % (rfo_root+1)
                        continue
               
                # Check normalizability of root.
                if fabs(SRFOevects[i][dim]) < 1.0e-10:  # don't even try to divide
                    if alphaIter == 0:
                        print "\tRejecting RFO root %d because normalization gives large value." % (rfo_root+1)
                    continue
                tval = absMax(SRFOevects[rfo_root]/SRFOevects[rfo_root][dim])
                if tval > Params.rfo_normalization_max: # matching test in code above
                    if alphaIter == 0:
                        print "\tRejecting RFO root %d because normalization gives large value." % (rfo_root+1)
                    continue

                rfo_root = i   # This root is acceptable.
                break 
            else:
                rfo_root = Params.rfo_root; # no good one found, use the default

            # Save initial root.
            # 'Follow' during the RS-RFO iterations.
            rfo_follow_root = True

        else: # Do root following.
            # Find maximum overlap. Dot only within H block.
            dots = np.array ( [v3d.dot(SRFOevects[i], last_iter_evect, dim) for i in range(dim)], float)
            rfo_root = np.argmax(dots)

        if alphaIter == 0:
            print "\tUsing RFO solution %d." % (rfo_root+1)
        last_iter_evect[:] = SRFOevects[rfo_root][0:dim]  # omit last column on right

        # Print only the lowest eigenvalues/eigenvectors
        if Params.print_lvl >= 2:
            print "\trfo_root is %d" % (rfo_root+1)
            for i in range(dim+1):
                if SRFOevals[i] < -1e-6 or i < rfo_root:
                    print "Scaled RFO eigenvalue %d: %15.10lf (or 2*%-15.10lf)" % (i+1, SRFOevals[i], SRFOevals[i]/2)
                    print "eigenvector:"
                    printArray(SRFOevects[i])

        dq[:] = SRFOevects[rfo_root][0:dim] # omit last column

        # Project out redundancies in steps.
        # Added this projection in 2014; but doesn't seem to help, as f,H are already projected.
        # project_dq(dq);

        # zero steps for frozen coordinates?

        dqtdq = np.dot(dq,dq)
        # If alpha explodes, give up on iterative scheme
        if fabs(alpha) > Params.rsrfo_alpha_max:
            converged = False
            alphaIter = max_projected_rfo_iter - 1
        elif sqrt(dqtdq) < (trust+1e-5):
            converged = True

        if alphaIter == 0 and not Params.simple_step_scaling:
            print "\n\tDetermining step-restricting scale parameter for RS-RFO."

        if alphaIter == 0:
            print "\tMaximum step size allowed %10.5lf" % trust
            print "\t Iter      |step|        alpha        rfo_root  "
            print "\t------------------------------------------------"
            print "\t%5d%12.5lf%14.5lf%12d" % (alphaIter+1, sqrt(dqtdq), alpha, rfo_root+1)

        elif alphaIter > 0 and not Params.simple_step_scaling:
            print "\t%5d%12.5lf%14.5lf%12d" % (alphaIter+1, sqrt(dqtdq), alpha, rfo_root+1)

        # Find the analytical derivative, d(norm step squared) / d(alpha)
        Lambda = -1 * v3d.dot(fq, dq, dim)
        if Params.print_lvl >= 2:
            print "dq:"
            printArray(dq, dim)
            print "fq:"
            printArray(fq, dim)
            print "\tLambda calculated by (dq^t).(-f)     = %20.10lf" % Lambda

        # Calculate derivative of step size wrt alpha.
        # Equation 20, Besalu and Bofill, Theor. Chem. Acc., 1999, 100:265-274
        tval = 0
        for i in range(dim):
            tval += ( pow(v3d.dot(Hevects[i], fq, dim),2) ) / ( pow(( Hevals[i]-Lambda*alpha ),3) )

        analyticDerivative = 2*Lambda / (1 + alpha * dqtdq ) * tval
        if Params.print_lvl >= 2:
            print "\tAnalytic derivative d(norm)/d(alpha) = %20.10lf" % analyticDerivative

        # Calculate new scaling alpha value.
        # Equation 20, Besalu and Bofill, Theor. Chem. Acc., 1999, 100:265-274
        alpha += 2*(trust * sqrt(dqtdq) - dqtdq) / analyticDerivative

    # end alpha RS-RFO iterations

    print "\t------------------------------------------------"

    # Crude/old way to limit step size if RS-RFO iterations
    if not converged or Params.simple_step_scaling:
        applyIntrafragStepScaling(dq)

    if Params.print_lvl >= 3:
        print "\tFinal scaled step dq:"
        printArray(dq)

    # Get norm |dq|, unit vector, gradient and hessian in step direction
    # TODO double check Hevects[i] here instead of H ? as for NR
    rfo_dqnorm = sqrt( np.dot(dq,dq) )
    print "\tNorm of target step-size %15.10f" % rfo_dqnorm
    rfo_u = dq.copy() / rfo_dqnorm
    rfo_g = -1 * np.dot(fq, rfo_u)
    rfo_h = np.dot( rfo_u, np.dot(H, rfo_u) )
    DEprojected = DE_projected('RFO', rfo_dqnorm, rfo_g, rfo_h)
    if Params.print_lvl > 1:
       print '\t|RFO target step|: %15.10f' % rfo_dqnorm
       print '\tRFO_gradient       : %15.10f' % rfo_g
       print '\tRFO_hessian        : %15.10f' % rfo_h
    print "\tProjected energy change by RFO approximation: %20.10lf" % DEprojected

    # Scale fq into aJ for printing
    fq_aJ = qShowForces(Molsys.intcos, fq)

    # this won't work for multiple fragments yet until dq and fq get cut up.
    for F in Molsys._fragments:
        displace(F.intcos, F.geom, dq, fq_aJ)

    # For now, saving RFO unit vector and using it in projection to match C++ code,
    # could use actual Dq instead.
    dqnorm_actual = sqrt( np.dot(dq,dq) )
    print "\tNorm of achieved step-size %15.10f" % dqnorm_actual

    # To test step sizes
    #x_before = original geometry
    #x_after = new geometry
    #masses
    #change = 0.0;
    #for i in range(Natom):
    #  for xyz in range(3):
    #    change += (x_before[3*i+xyz] - x_after[3*i+xyz]) * (x_before[3*i+xyz] - x_after[3*i+xyz])
    #             * masses[i]
    #change = sqrt(change);
    #print "Step-size in mass-weighted cartesian coordinates [bohr (amu)^1/2] : %20.10lf" % change

    #print "\tSymmetrizing new geometry"
    #geom = symmetrizeXYZ(geom)

    History.appendRecord(DEprojected, dq, rfo_u, rfo_g, rfo_h)

    linearList = linearBendCheck(Molsys.intcos, Molsys.geom, dq)
    if linearList:
        raise INTCO_EXCEPT("New linear angles", linearList)

    # Before quitting, make sure step is reasonable.  It should only be
    # screwball if we are using the "First Guess" after the back-transformation failed.
    if sqrt(np.dot(dq, dq)) > 10 * trust:
        raise optExceptions.BAD_STEP_EXCEPT("opt.py: Step is far too large.")

    return dq

# Take Steepest Descent step
def Dq_SD(intcos, geom, E, fq):
    print "\tTaking SD optimization step."
    dim = len(fq)
    sd_h = Params.sd_hessian # default value

    if len(History.steps) > 1:
        previous_forces = History.steps[-2].forces
        previous_dq     = History.steps[-2].Dq

        # Compute overlap of previous forces with current forces.
        previous_forces_u = previous_forces.copy()/np.linalg.norm(previous_forces)
        forces_u = fq.copy() / np.linalg.norm(fq)
        overlap = np.dot(previous_forces_u, forces_u)
        print "\tOverlap of current forces with previous forces %8.4lf" % overlap
        previous_dq_norm = np.linalg.norm(previous_dq)

        if overlap > 0.50:
            # Magnitude of current force
            fq_norm = np.linalg.norm(fq)
            # Magnitude of previous force in step direction
            previous_forces_norm = v3d.dot(previous_forces, fq, dim)/fq_norm
            sd_h = (previous_forces_norm - fq_norm) / previous_dq_norm

    print "\tEstimate of Hessian along step: %10.5e" % sd_h
    dq = fq / sd_h

    applyIntrafragStepScaling(dq)

    sd_dqnorm = np.linalg.norm(dq)
    print "\tNorm of target step-size %10.5f\n" % sd_dqnorm

    # unit vector in step direction
    sd_u = dq.copy() / np.linalg.norm( dq )
    sd_g = -1.0 * sd_dqnorm

    DEprojected = DE_projected('NR', sd_dqnorm, sd_g, sd_h);
    print "\tProjected energy change by quadratic approximation: %20.10lf" % DEprojected

    fq_aJ = qShowForces(intcos, fq) # for printing
    displace(intcos, geom, dq, fq_aJ)

    dqnorm_actual = np.linalg.norm(dq)
    print "\tNorm of achieved step-size %15.10f" % dqnorm_actual

    # Symmetrize the geometry for next step
    # symmetrize_geom()

    History.appendRecord(DEprojected, dq, sd_u, sd_g, sd_h)

    linearList = linearBendCheck(Molsys.intcos, Molsys.geom, dq)
    if linearList:
        raise INTCO_EXCEPT("New linear angles", linearList)

    return dq

# Take partial backward step.  Update current step in history.
# Divide the last step size by 1/2 and displace from old geometry.
# HISTORY contains:
#   consecutiveBacksteps : increase by 1
#   HISTORY.STEP contains:
#     No change to these:
#       forces, geom, E, followedUnitVector, oneDgradient, oneDhessian
#     Update these:
#       Dq - cut in half
#       projectedDE - recompute

def Dq_BACKSTEP(intcos, geom):
    print "\tRe-doing last optimization step - smaller this time."

    if len(History.steps) < 2:
        raise Exception("Backstep called, but no history is available.")

    History.consecutiveBacksteps += 1
    print "\tConsecutive backstep number %d." % (History.consecutiveBacksteps)

    # Erase last, partial step data for current step.
    del History.steps[-1]

    # Get data from previous step.
    # Put previous geometry into current working one.
    fq      = History.steps[-1].forces
    dq      = History.steps[-1].Dq
    geom[:] = History.steps[-1].geom
    oneDgradient  = History.steps[-1].oneDgradient
    oneDhessian  = History.steps[-1].oneDhessian

    # Compute new Dq and energy step projection.
    dq /= 2
    dqNorm = np.linalg.norm(dq)
    print "\tNorm of target step-size %10.5f\n" % dqNorm

    # Compute new Delta(E) projection.
    if Params.step_type == 'RFO':
        DEprojected = DE_projected('RFO', dqNorm, oneDgradient, oneDhessian)
    else:
        DEprojected = DE_projected('NR', dqNorm, oneDgradient, oneDhessian)
    print "\tProjected energy change : %20.10lf" % DEprojected

    fq_aJ = qShowForces(intcos, fq) # for printing
    displace(intcos, geom, dq, fq_aJ)
    dqNormActual = np.linalg.norm(dq)
    print "\tNorm of achieved step-size %15.10f" % dqNormActual
    # Symmetrize the geometry for next step
    # symmetrize_geom()

    # Update the history entries which changed.
    History.steps[-1].projectedDE = DEprojected
    History.steps[-1].Dq[:] = dq

    linearList = linearBendCheck(Molsys.intcos, Molsys.geom, dq)
    if linearList:
        raise INTCO_EXCEPT("New linear angles", linearList)

    return dq

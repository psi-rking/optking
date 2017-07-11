# Functions for step algorithms: Newton-Raphson, Rational Function Optimization,
# Steepest Descent. 
import numpy as np
from optParams import Params
from displace import displace
from intcosMisc import qShowForces
from addIntcos import linearBendCheck
from math import sqrt, fabs
from printTools import printArray, printMat, print_opt
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
    elif stepType == 'P_RFO':
        return Dq_P_RFO(Molsys, E, qForces, H)
    elif opt_type == 'IRC':
        return Dq_IRC(Molsys, E, qForces, H)  
    else:
        raise ValueError('Dq: step type not yet implemented')

# Apply crude maximum step limit by scaling.
def applyIntrafragStepScaling(dq):
    trust = Params.intrafrag_trust
    if sqrt(np.dot(dq,dq)) > trust:
        scale = trust / sqrt(np.dot(dq,dq))
        print_opt("\tStep length exceeds trust radius of %10.5f.\n" % trust)
        print_opt("\tScaling displacements by %10.5f\n" % scale)
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
def Dq_NR(Molsys, E, fq, H):
    print_opt("\tTaking NR optimization step.\n")

    # Hinv fq = dq
    Hinv = symmMatInv(H, redundant=True)
    dq = np.dot(Hinv, fq)

    # applies maximum internal coordinate change
    applyIntrafragStepScaling(dq);

    # get norm |q| and unit vector in the step direction
    nr_dqnorm = sqrt( np.dot(dq,dq) )
    nr_u = dq.copy() / nr_dqnorm
    print_opt("\tNorm of target step-size %15.10f\n" % nr_dqnorm)

    # get gradient and hessian in step direction
    nr_g = -1 * np.dot(fq, nr_u) # gradient, not force
    nr_h = np.dot( nr_u, np.dot(H, nr_u) )

    if Params.print_lvl > 1:
       print_opt('\t|NR target step|: %15.10f\n' % nr_dqnorm)
       print_opt('\tNR_gradient     : %15.10f\n' % nr_g)
       print_opt('\tNR_hessian      : %15.10f\n' % nr_h)
    DEprojected = DE_projected('NR', nr_dqnorm, nr_g, nr_h)
    print_opt("\tProjected energy change by quadratic approximation: %20.10lf\n" % DEprojected)

    # Scale fq into aJ for printing
    fq_aJ = qShowForces(Molsys.intcos, fq)
    displace(Molsys._fragments[0].intcos, Molsys._fragments[0].geom, dq, fq_aJ)

    dq_actual = sqrt( np.dot(dq,dq) )
    print_opt("\tNorm of achieved step-size %15.10f\n" % dq_actual)

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
    print_opt("\tTaking RFO optimization step.\n")
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
        print_opt("Original, unscaled RFO matrix:\n")
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
            print_opt("\tFailed to converge alpha. Doing simple step-scaling instead.\n")
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
            print_opt("\nScaled RFO matrix.\n")
            printMat(SRFOmat)

        # Find the eigenvectors and eigenvalues of RFO matrix.
        SRFOevals, SRFOevects = asymmMatEig(SRFOmat)

        if Params.print_lvl >= 4:
            print_opt("Eigenvectors of scaled RFO matrix.\n")
            printMat(SRFOevects)

        if Params.print_lvl >= 2:
            print_opt("Eigenvalues of scaled RFO matrix.\n")
            printArray(SRFOevals)
            print_opt("First eigenvector (unnormalized) of scaled RFO matrix.\n")
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
            print_opt("All scaled RFO eigenvectors (rows).\n")
            printMat(SRFOevects)

        # Use input rfo_root
        # If root-following is turned off, then take the eigenvector with the rfo_root'th lowest eigvenvalue.
        # If its the first iteration, then do the same.  In subsequent steps, overlaps will be checked.
        if not rfo_follow_root or len(History.steps) < 2:

            # Determine root only once at beginning ?
            if alphaIter == 0:
                print_opt("\tChecking RFO solution %d.\n" % (rfo_root+1))

                for i in range(rfo_root, dim+1):
                    # Check symmetry of root.
                    dq[:] = SRFOevects[i,0:dim]
                    if not Params.accept_symmetry_breaking:
                        symm_rfo_step = isDqSymmetric(Molsys.intcos, Molsys.geom, dq)
        
                        if not symm_rfo_step:  # Root is assymmetric so reject it.
                            print_opt("\tRejecting RFO root %d because it breaks the molecular point group.\n"\
                                        % (rfo_root+1))
                            continue
               
                    # Check normalizability of root.
                    if fabs(SRFOevects[i][dim]) < 1.0e-10:  # don't even try to divide
                        print_opt("\tRejecting RFO root %d because normalization gives large value.\n" % (rfo_root+1))
                        continue
                    tval = absMax(SRFOevects[i]/SRFOevects[i][dim])
                    if tval > Params.rfo_normalization_max: # matching test in code above
                        print_opt("\tRejecting RFO root %d because normalization gives large value.\n" % (rfo_root+1))
                        continue
                    rfo_root = i   # This root is acceptable.
                    break 
                else:
                    rfo_root = Params.rfo_root; # no good one found, use the default

                # Save initial root. 'Follow' during the RS-RFO iterations.
                rfo_follow_root = True

        else: # Do root following.
            # Find maximum overlap. Dot only within H block.
            dots = np.array ( [v3d.dot(SRFOevects[i], last_iter_evect, dim) for i in range(dim)], float)
            bestfit = np.argmax(dots)
            if bestfit != rfo_root:
                print_opt("Root-following has changed rfo_root value to %d." % (bestfit+1))
                rfo_root = bestfit

        if alphaIter == 0:
            print_opt("\tUsing RFO solution %d.\n" % (rfo_root+1))
        last_iter_evect[:] = SRFOevects[rfo_root][0:dim]  # omit last column on right

        # Print only the lowest eigenvalues/eigenvectors
        if Params.print_lvl >= 2:
            print_opt("\trfo_root is %d\n" % (rfo_root+1))
            for i in range(dim+1):
                if SRFOevals[i] < -1e-6 or i < rfo_root:
                    print_opt("Scaled RFO eigenvalue %d: %15.10lf (or 2*%-15.10lf)\n" % (i+1, SRFOevals[i], SRFOevals[i]/2))
                    print_opt("eigenvector:\n")
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
            print_opt("\n\tDetermining step-restricting scale parameter for RS-RFO.\n")

        if alphaIter == 0:
            print_opt("\tMaximum step size allowed %10.5lf\n" % trust)
            print_opt("\t Iter      |step|        alpha        rfo_root  \n")
            print_opt("\t------------------------------------------------\n")
            print_opt("\t%5d%12.5lf%14.5lf%12d\n" % (alphaIter+1, sqrt(dqtdq), alpha, rfo_root+1))

        elif alphaIter > 0 and not Params.simple_step_scaling:
            print_opt("\t%5d%12.5lf%14.5lf%12d\n" % (alphaIter+1, sqrt(dqtdq), alpha, rfo_root+1))

        # Find the analytical derivative, d(norm step squared) / d(alpha)
        Lambda = -1 * v3d.dot(fq, dq, dim)
        if Params.print_lvl >= 2:
            print_opt("dq:\n")
            printArray(dq, dim)
            print_opt("fq:\n")
            printArray(fq, dim)
            print_opt("\tLambda calculated by (dq^t).(-f)     = %20.10lf\n" % Lambda)

        # Calculate derivative of step size wrt alpha.
        # Equation 20, Besalu and Bofill, Theor. Chem. Acc., 1999, 100:265-274
        tval = 0
        for i in range(dim):
            tval += ( pow(v3d.dot(Hevects[i], fq, dim),2) ) / ( pow(( Hevals[i]-Lambda*alpha ),3) )

        analyticDerivative = 2*Lambda / (1 + alpha * dqtdq ) * tval
        if Params.print_lvl >= 2:
            print_opt("\tAnalytic derivative d(norm)/d(alpha) = %20.10lf\n" % analyticDerivative)

        # Calculate new scaling alpha value.
        # Equation 20, Besalu and Bofill, Theor. Chem. Acc., 1999, 100:265-274
        alpha += 2*(trust * sqrt(dqtdq) - dqtdq) / analyticDerivative

    # end alpha RS-RFO iterations

    print_opt("\t------------------------------------------------\n")

    # Crude/old way to limit step size if RS-RFO iterations
    #if not converged or Params.simple_step_scaling:
    applyIntrafragStepScaling(dq)

    if Params.print_lvl >= 3:
        print_opt("\tFinal scaled step dq:\n")
        printArray(dq)

    # Get norm |dq|, unit vector, gradient and hessian in step direction
    # TODO double check Hevects[i] here instead of H ? as for NR
    rfo_dqnorm = sqrt( np.dot(dq,dq) )
    print_opt("\tNorm of target step-size %15.10f\n" % rfo_dqnorm)
    rfo_u = dq.copy() / rfo_dqnorm
    rfo_g = -1 * np.dot(fq, rfo_u)
    rfo_h = np.dot( rfo_u, np.dot(H, rfo_u) )
    DEprojected = DE_projected('RFO', rfo_dqnorm, rfo_g, rfo_h)
    if Params.print_lvl > 1:
       print_opt('\t|RFO target step|  : %15.10f\n' % rfo_dqnorm)
       print_opt('\tRFO gradient       : %15.10f\n' % rfo_g)
       print_opt('\tRFO hessian        : %15.10f\n' % rfo_h)
    print_opt("\tProjected energy change by RFO approximation: %20.10lf\n" % DEprojected)

    # Scale fq into aJ for printing
    fq_aJ = qShowForces(Molsys.intcos, fq)

    # this won't work for multiple fragments yet until dq and fq get cut up.
    for F in Molsys._fragments:
        displace(F.intcos, F.geom, dq, fq_aJ)

    # For now, saving RFO unit vector and using it in projection to match C++ code,
    # could use actual Dq instead.
    dqnorm_actual = sqrt( np.dot(dq,dq) )
    print_opt("\tNorm of achieved step-size %15.10f\n" % dqnorm_actual)

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
    #print_opt("Step-size in mass-weighted cartesian coordinates [bohr (amu)^1/2] : %20.10lf\n" % change)

    #print_opt("\tSymmetrizing new geometry\n")
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


def Dq_P_RFO(Molsys, E, fq, H):
    Hdim = len(fq) # size of Hessian
    trust = Params.intrafrag_trust   # maximum step size
    rfo_follow_root = Params.rfo_follow_root # whether to follow root
    print_lvl = Params.print_lvl

    if print_lvl > 2:
        print_opt("Hessian matrix\n")
        printMat(H)
    
    # Diagonalize H (technically only have to semi-diagonalize)
    hEigValues, hEigVectors = symmMatEig(H)

    if print_lvl > 2:
        print_opt("Eigenvalues of Hessian\n")
        printArray(hEigValues)
        print_opt("Eigenvectors of Hessian (rows)\n")
        printMat(hEigVectors)

    # Construct diagonalized Hessian with evals on diagonal
    HDiag = np.zeros((Hdim,Hdim), float)
    for i in range(Hdim):
        HDiag[i,i] = hEigValues[i]

    if print_lvl > 2:
        print_opt("H diagonal\n")
        printMat(HDiag)

    print_opt("\tFor P-RFO, assuming rfo_root=1, maximizing along lowest eigenvalue of Hessian.\n")
    print_opt("\tLarger values of rfo_root are not yet supported.\n")
    rfo_root = 0

    """  TODO: use rfo_root to decide which eigenvectors are moved into the max/mu space.
    if not rfo_follow_root or len(History.steps) < 2:
        rfo_root = Params.rfo_root
        print_opt("\tMaximizing along %d lowest eigenvalue of Hessian.\n" % (rfo_root+1) )
    else:
        last_iter_evect = history[-1].Dq
        dots = np.array([v3d.dot(hEigVectors[i],last_iter_evect,Hdim) for i in range(Hdim)], float)
        rfo_root = np.argmax(dots)
        print_opt("\tOverlaps with previous step checked for root-following.\n")
        print_opt("\tMaximizing along %d lowest eigenvalue of Hessian.\n" % (rfo_root+1) )
    """

    # number of degrees along which to maximize; assume 1 for now
    mu = 1

    print_opt("\tInternal forces in au:\n")
    printArray(fq)

    fqTransformed = np.dot(hEigVectors, fq) #gradient transformation    
    print_opt("\tInternal forces in au, in Hevect basis:\n")
    printArray(fqTransformed)

    # Build RFO max
    maximizeRFO = np.zeros((mu + 1, mu + 1), float)
    for i in range(mu):
        maximizeRFO[i,i] = hEigValues[i]
        maximizeRFO[i, -1] = -fqTransformed[i]
        maximizeRFO[-1, i] = -fqTransformed[i]
    if print_lvl > 2:
        print_opt("RFO max\n")
        printMat(maximizeRFO)      

    # Build RFO min
    minimizeRFO = np.zeros((Hdim - mu + 1, Hdim - mu + 1), float)
    for i in range (0, Hdim - mu):
        minimizeRFO[i,i] = HDiag[i + mu,i + mu]
        minimizeRFO[i,-1] = -fqTransformed[i + mu]
        minimizeRFO[-1,i] = -fqTransformed[i + mu]
    if print_lvl > 2:
        print_opt("RFO min\n")
        printMat(minimizeRFO) 

    RFOMaxEValues, RFOMaxEVectors = symmMatEig(maximizeRFO)
    RFOMinEValues, RFOMinEVectors = symmMatEig(minimizeRFO)

    print_opt("RFO min eigenvalues:\n")
    printArray(RFOMinEValues)
    print_opt("RFO max eigenvalues:\n")
    printArray(RFOMaxEValues)

    if print_lvl > 2:
        print_opt("RFO min eigenvectors (rows) before normalization:\n")
        printMat(RFOMinEVectors)
        print_opt("RFO max eigenvectors (rows) before normalization:\n")
        printMat(RFOMaxEVectors)

    # Normalize max and min eigenvectors
    for i in range(mu+1):
        if abs(RFOMaxEVectors[i,mu]) > 1.0e-10:
            tval = abs( absMax(RFOMaxEVectors[i,0:mu]) / RFOMaxEVectors[i,mu])
            if fabs(tval) < Params.rfo_normalization_max:
                RFOMaxEVectors[i] /= RFOMaxEVectors[i,mu]
    if print_lvl > 2:
        print_opt("RFO max eigenvectors (rows):\n")
        printMat(RFOMaxEVectors)

    for i in range(Hdim-mu+1):
        if abs(RFOMinEVectors[i][Hdim-mu]) > 1.0e-10:
            tval = abs( absMax(RFOMinEVectors[i,0:Hdim-mu]) / RFOMinEVectors[i,Hdim-mu])
            if fabs(tval) < Params.rfo_normalization_max:
                RFOMinEVectors[i] /= RFOMinEVectors[i,Hdim-mu]
    if print_lvl > 2:
        print_opt("RFO min eigenvectors (rows):\n")
        printMat(RFOMinEVectors)

    VectorP = RFOMaxEVectors[mu,0:mu]
    VectorN = RFOMinEVectors[rfo_root,0:Hdim-mu]
    print_opt("Vector P\n")
    print_opt( str(VectorP)+'\n' )
    print_opt("Vector N\n")
    print_opt( str(VectorN)+'\n' )

    # Combines the eignvectors from RFO max and min
    PRFOEVector = np.zeros(Hdim, float)
    PRFOEVector[0:len(VectorP)] = VectorP
    PRFOEVector[len(VectorP):] = VectorN

    PRFOStep = np.dot(hEigVectors.transpose(), PRFOEVector)

    if print_lvl > 1:
        print_opt("RFO step in Hessian Eigenvector Basis\n")
        printArray(PRFOEVector)
        print_opt("RFO step in original Basis\n")
        printArray(PRFOStep)

    dq = PRFOStep
        
    #if not converged or Params.simple_step_scaling:
    applyIntrafragStepScaling(dq)

    # Get norm |dq|, unit vector, gradient and hessian in step direction
    # TODO double check Hevects[i] here instead of H ? as for NR
    rfo_dqnorm = sqrt( np.dot(dq,dq) )
    print_opt("\tNorm of target step-size %15.10f\n" % rfo_dqnorm)
    rfo_u = dq.copy() / rfo_dqnorm
    rfo_g = -1 * np.dot(fq, rfo_u)
    rfo_h = np.dot( rfo_u, np.dot(H, rfo_u) )
    DEprojected = DE_projected('RFO', rfo_dqnorm, rfo_g, rfo_h)
    if Params.print_lvl > 1:
       print_opt('\t|RFO target step|  : %15.10f\n' % rfo_dqnorm)
       print_opt('\tRFO gradient       : %15.10f\n' % rfo_g)
       print_opt('\tRFO hessian        : %15.10f\n' % rfo_h)
    print_opt("\tProjected Delta(E) : %15.10f\n\n" % DEprojected)

    # Scale fq into aJ for printing
    fq_aJ = qShowForces(Molsys.intcos, fq)

    # this won't work for multiple fragments yet until dq and fq get cut up.
    for F in Molsys._fragments:
        displace(F.intcos, F.geom, dq, fq_aJ)

    # For now, saving RFO unit vector and using it in projection to match C++ code,
    # could use actual Dq instead. 
    dqnorm_actual = sqrt( np.dot(dq,dq) )
    print_opt("\tNorm of achieved step-size %15.10f\n" % dqnorm_actual)
        
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
def Dq_SD(Molsys, E, fq):
    print_opt("\tTaking SD optimization step.\n")
    dim = len(fq)
    sd_h = Params.sd_hessian # default value

    if len(History.steps) > 1:
        previous_forces = History.steps[-2].forces
        previous_dq     = History.steps[-2].Dq

        # Compute overlap of previous forces with current forces.
        previous_forces_u = previous_forces.copy()/np.linalg.norm(previous_forces)
        forces_u = fq.copy() / np.linalg.norm(fq)
        overlap = np.dot(previous_forces_u, forces_u)
        print_opt("\tOverlap of current forces with previous forces %8.4lf\n" % overlap)
        previous_dq_norm = np.linalg.norm(previous_dq)

        if overlap > 0.50:
            # Magnitude of current force
            fq_norm = np.linalg.norm(fq)
            # Magnitude of previous force in step direction
            previous_forces_norm = v3d.dot(previous_forces, fq, dim)/fq_norm
            sd_h = (previous_forces_norm - fq_norm) / previous_dq_norm

    print_opt("\tEstimate of Hessian along step: %10.5e\n" % sd_h)
    dq = fq / sd_h

    applyIntrafragStepScaling(dq)

    sd_dqnorm = np.linalg.norm(dq)
    print_opt("\tNorm of target step-size %10.5f\n" % sd_dqnorm)

    # unit vector in step direction
    sd_u = dq.copy() / np.linalg.norm( dq )
    sd_g = -1.0 * sd_dqnorm

    DEprojected = DE_projected('NR', sd_dqnorm, sd_g, sd_h);
    print_opt("\tProjected energy change by quadratic approximation: %20.10lf\n" % DEprojected)

    fq_aJ = qShowForces(Molsys.intcos, fq) # for printing
    displace(Molsys._fragments[0].intcos, Molsys._fragments[0].geom, dq, fq_aJ)

    dqnorm_actual = np.linalg.norm(dq)
    print_opt("\tNorm of achieved step-size %15.10f\n" % dqnorm_actual)

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

def Dq_BACKSTEP(Molsys):
    print_opt("\tRe-doing last optimization step - smaller this time.\n")

    if len(History.steps) < 2:
        raise Exception("Backstep called, but no history is available.")

    History.consecutiveBacksteps += 1
    print_opt("\tConsecutive backstep number %d.\n" % (History.consecutiveBacksteps))

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
    print_opt("\tNorm of target step-size %10.5f\n" % dqNorm)

    # Compute new Delta(E) projection.
    if Params.step_type == 'RFO':
        DEprojected = DE_projected('RFO', dqNorm, oneDgradient, oneDhessian)
    else:
        DEprojected = DE_projected('NR', dqNorm, oneDgradient, oneDhessian)
    print_opt("\tProjected energy change : %20.10lf\n" % DEprojected)

    fq_aJ = qShowForces(Molsys.intcos, fq) # for printing
    displace(Molsys._fragments[0].intcos, Molsys._fragments[0].geom, dq, fq_aJ)
    dqNormActual = np.linalg.norm(dq)
    print_opt("\tNorm of achieved step-size %15.10f\n" % dqNormActual)
    # Symmetrize the geometry for next step
    # symmetrize_geom()

    # Update the history entries which changed.
    History.steps[-1].projectedDE = DEprojected
    History.steps[-1].Dq[:] = dq

    linearList = linearBendCheck(Molsys.intcos, Molsys.geom, dq)
    if linearList:
        raise INTCO_EXCEPT("New linear angles", linearList)

    return dq

def Dq_IRC (Molsys, intcos, geom, E, qForces, H, B): #if this method requires additinal parameters, calls to Dq will also need additional info
    #Call Hessian convert to internals, or pass in H already in internals when dq called. Hessian will be calculated from psi4
    Hq = intcosMisc.convertHessianToInternals(B, H, qForces, Molsys)
    #Compute G in mass weighted internals, can be performed using intcosMisc.Gmat(intcos, geom, True) returns matrix
    Gm = intcosMisc.Gmat(intcos, geom, True)
    #Compute gradient from psi4 
    #q omputed in mass weighted coordinates
    # Hessian transformed into mass weighted by H_m = (G^1/2)H(G^1/2) where G is calculated from dq in massweighted internals
    #Solve for N using G N = (g^t * Gg)^-1/2
    #q*k+1 = qk - s/2 * NGg
    #length of all vectors = 1/2 s
    #need to scan radius of 1/2 step size from pivot point to find qk+1 
    #for this do I simply need to scan all points at 1/2 s away from the pivot point?
    #Need to solve Ggk+1 = Gg'k+1 + GHq (qk+1 - q'k+1) #question remains how to find q'k+1 I think I can just find
    #some point at a set angle from the pivot point and calc the dq from this random poin  

#!/usr/bin/python
# Calling program provides user-specified options.
def optimize( Molsys, options_in, fSetGeometry, fGradient, fHessian, fEnergy ):
    
    import caseInsensitiveDict
    userOptions = caseInsensitiveDict.CaseInsensitiveDict( options_in )
    # Save copy of original user options.
    origOptions = userOptions.copy()
    
    # Create full list of parameters from defaults plus user options.
    import optParams as op
    op.Params = op.OPT_PARAMS(userOptions)
    print '\tParameters from optking.optimize'
    print op.Params

    # while op.Params.dynamic_level

    from printTools import printGeomGrad, printMat, printArray
    from addIntcos import connectivityFromDistances, markAsFrozen, parseFrozenString
    import optExceptions
    import history 
    import stepAlgorithms
    import intcosMisc
    import hessian
    import convCheck
    import testB
    energies = []

    print Molsys

    # *** Construct coordinates to use.
    # See what is bonded.
    C = connectivityFromDistances(Molsys.geom, Molsys.Z)

    if op.Params.frag_mode == 'SINGLE':
        #Add to connectivity to make sure all fragments connected.
        Molsys.augmentConnectivityToSingleFragment(C)
        Molsys.consolidateFragments();
    elif op.Params.frag_mode == 'MULTI':
        # should do nothing if fragments are already split by calling program/constructor.
        Molsys.splitFragmentsByConnectivity()

    if op.Params.opt_coordinates in ['REDUNDANT','BOTH']:
        Molsys.addIntcosFromConnectivity(C)
    if op.Params.opt_coordinates in ['CARTESIAN','BOTH']:
        Molsys.addCartesianIntcos()
# Testing Implementaiton of frozen coordinates
    if op.Params.frozen_distance != None:
        addIntcos.markAsFrozen(frozen_distance, Molsys.intcos)
    if op.Params.frozen_bend != None:
        addIntcos.markAsFrozen(frozen_bend, Molsys.intcos)
    if op.Params.frozen_dihedral != None:
        addIntcos.markAsFrozen(frozen_dihedral, Molsys.intcos)   	    	
    
    Molsys.printIntcos();

    stepNumber = 0
    while stepNumber < op.Params.geom_maxiter: 
        E, g_x = fGradient()
        printGeomGrad(Molsys.geom, g_x)
        energies.append( E )
    
    
        if op.Params.test_B:
            testB.testB(Molsys.intcos, Molsys.geom)
        if op.Params.test_derivative_B:
            testB.testDerivativeB(Molsys.intcos, Molsys.geom)
    
        if op.Params.print_lvl > 3:
            B = intcosMisc.Bmat(Molsys.intcos, Molsys.geom)
            print "B matrix:"
            printMat(B)

        fq = intcosMisc.qForces(Molsys.intcos, Molsys.geom, g_x)
        if (op.Params.print_lvl > 1):
            print "Internal forces in au"
            printArray(fq)
    
        history.History.append(Molsys.geom, E, fq); # Save initial step info.
    
        history.History.currentStepReport()
    
        if stepNumber == 0:
            C = connectivityFromDistances(Molsys.geom, Molsys.Z)
            H = hessian.guess(Molsys.intcos, Molsys.geom, Molsys.Z, C, op.Params.intrafrag_hess)
        else:
            history.History.hessianUpdate(H, Molsys.intcos)
    
        print 'Hessian (in au) is:'
        printMat(H)
        print 'Hessian in aJ/Ang^2 or aJ/deg^2'
        hessian.show(H, Molsys.intcos)

        intcosMisc.project_redundancies(Molsys.intcos, Molsys.geom, fq, H)

        try:
           # displaces and adds step to history
           Dq = stepAlgorithms.Dq(Molsys, E, fq, H)
        except optExceptions.BAD_STEP_EXCEPT:
           if history.History.consecutiveBacksteps < op.Params.consecutive_backsteps_allowed:
               print 'Taking backward step'
               Dq = stepAlgorithms.Dq(Molsys.intcos, Molsys.geom, E, fq, H, stepType="BACKSTEP")
           else:
               print 'Maximum number of backsteps has been attempted.  Continuing.'

        check = convCheck.convCheck(stepNumber, Molsys.intcos, Dq, fq, energies)
    
        if check:
           print 'Converged in %d steps!' % (int(stepNumber)+1)
           fSetGeometry(Molsys.geom)
           break
        
        # Now need to return geometry in preparation for next step.
        print "\tStructure for next step (au):"
        Molsys.printGeom()
        #print "\tStructure for next step (Angstroms):"
        #Molsys.showGeom()
        fSetGeometry(Molsys.geom)
    
        stepNumber += 1
    
    #print history.History
    history.History.summary()

    return history.History[-1].E


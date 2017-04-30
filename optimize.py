#!/usr/bin/python
# Calling program provides user-specified options.
def optimize( molsys, options_in, fsetGeometry, fgradient, fhessian, fenergy ):
    
    import caseInsensitiveDict
    userOptions = caseInsensitiveDict.CaseInsensitiveDict( options_from_calling_program )
    # Save copy of original user options.
    origOptions = userOptions.copy()
    
    # Create full list of parameters from defaults plus user options.
    import optParams as op
    op.Params = op.OPT_PARAMS(userOptions)
    print '\tParameters from optking.optimize'
    print op.Params

    from printTools import printGeomGrad, printMat, printArray
    import optExceptions
    import history 
    import stepAlgorithms
    import intcosMisc
    import hessian
    import convCheck
    import testB
    energies = []

    print molsys

    hessian_func()
    
    stepNumber = 0
    while stepNumber < op.Params.geom_maxiter: 
        E, g_x = gradient_func()
        misc.printGeomGrad(oneFrag.geom, g_x)
        energies.append( E )
    
        # Construct coordinates to use.
    
        if stepNumber == 0:
            if op.Params.opt_coordinates in ['REDUNDANT','BOTH']:
                oneFrag.addIntcosFromConnectivity()
            if op.Params.opt_coordinates in ['CARTESIAN','BOTH']:
                oneFrag.addCartesianIntcos()
    
        oneFrag.printIntcos();
    
        if op.Params.test_B:
            testB.testB(oneFrag.intcos, oneFrag.geom)
        if op.Params.test_derivative_B:
            testB.testDerivativeB(oneFrag.intcos, oneFrag.geom)
    
        if op.Params.print_lvl > 3:
            B = intcosMisc.Bmat(oneFrag.intcos, oneFrag.geom)
            print "B matrix:"
            misc.printMat(B)
    
        fq = intcosMisc.qForces(oneFrag.intcos, oneFrag.geom, g_x)
        if (op.Params.print_lvl > 1):
            print "Internal forces in au"
            misc.printArray(fq)
    
        history.History.append(oneFrag.geom, E, fq); # Save initial step info.
    
        history.History.currentStepReport()
    
        if stepNumber == 0:
            H = hessian.guess(oneFrag.intcos, oneFrag.geom, oneFrag.Z, op.Params.intrafrag_hess)
    
    
        else:
            history.History.hessianUpdate(H, oneFrag.intcos)
    
        print 'Hessian (in au) is:'
        misc.printMat(H)
        #print 'Hessian in aJ/Ang^2 or aJ/deg^2'
        #hessian.show(H, intcos)
    
        intcosMisc.project_redundancies(oneFrag.intcos, oneFrag.geom, fq, H)
    
        try:
           # displaces and adds step to history
           Dq = stepAlgorithms.Dq(oneFrag.intcos, oneFrag.geom, E, fq, H)
        except optExceptions.BAD_STEP_EXCEPT:
           if history.History.consecutiveBacksteps < op.Params.consecutive_backsteps_allowed:
               print 'Taking backward step'
               Dq = stepAlgorithms.Dq(oneFrag.intcos, oneFrag.geom, E, fq, H, stepType="BACKSTEP")
           else:
               print 'Maximum number of backsteps has been attempted.  Continuing.'
        
        #print '\tIntco Values (Angstroms and degrees)'
        #for intco in oneFrag.intcos:
        #   print '\t\t%-10s%10.5f' % (intco, intco.qShow(oneFrag.geom))
        # print 'Dq'
        # misc.printArray(Dq)
        # print 'fq'
        # misc.printArray(fq)
    
        check = False
        check = convCheck.convCheck(stepNumber, oneFrag.intcos, Dq, fq, energies)
    
        if check:
           print 'Converged in %d steps!' % (int(stepNumber)+1)
           break
        
        # Now need to return geometry in preparation for next step.
        print "\tStructure for next step (au):"
        oneFrag.printGeom()
        print "\tStructure for next step (Angstroms):"
        oneFrag.showGeom()
        setGeometry_func(oneFrag.geom)
    
        stepNumber += 1
    
    print history.History
    history.History.summary()


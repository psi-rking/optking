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

    from printTools import printGeomGrad, printMat, printArray
    import addIntcos
    import optExceptions
    import history 
    import stepAlgorithms
    import intcosMisc
    import hessian
    import convCheck
    import testB
    converged = False

    # Loop over a variety of algorithms
    while (converged == False) and (op.Params.dynamic_level < op.Params.dynamic_level_max):

        try:
            print Molsys
            energies = []
        
            # Construct connectivity.
            C = addIntcos.connectivityFromDistances(Molsys.geom, Molsys.Z)
        
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
            if op.Params.frozen_distance is not None:
                print (op.Params.frozen_distance)
                FrozenIntcosListDis = addIntcos.parseFrozenString(op.Params.frozen_distance)
                addIntcos.markDisAsFrozen(FrozenIntcosListDis, Molsys.intcos)
            if op.Params.frozen_bend is not None:
                FrozenIntcosListBend = addIntcos.parseFrozenString(op.Params.frozen_bend)
                addIntcos.markBendAsFrozen(FrozenIntcosListBend, Molsys.intcos)
            if op.Params.frozen_dihedral is not None:
                FrozenIntcosListTors = addIntcos.parseFrozenString(op.Params.frozen_dihedral)
                addIntcos.markTorsAsFrozen(FrozenIntcosListTors, Molsys.intcos)
            
            Molsys.printIntcos();
        
            for stepNumber in range(op.Params.geom_maxiter): 
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
                    C = addIntcos.connectivityFromDistances(Molsys.geom, Molsys.Z)
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
                    if history.History.consecutiveBacksteps < op.Params.consecutiveBackstepsAllowed:
                        print 'Taking backward step.'
                        Dq = stepAlgorithms.Dq(Molsys.intcos, Molsys.geom, E, fq, H, stepType="BACKSTEP")
                    else:
                        print 'Maximum number of backsteps has been attempted.'
                        print 'Re-raising BAD_STEP exception'
                        raise optExceptions.BAD_STEP_EXCEPT()
        
                converged = convCheck.convCheck(stepNumber, Molsys.intcos, Dq, fq, energies)

                print "\tStructure for next step (au):"
                Molsys.printGeom()
                fSetGeometry(Molsys.geom)
            
                if converged:
                    print 'Converged in %d steps!' % (stepNumber+1)
                    fSetGeometry(Molsys.geom)
                    break

            else: # executes if too many steps
                print "Number of steps (%d) has reached value of GEOM_MAXITER." % (stepNumber+1)
                raise optExceptions.BAD_STEP_EXCEPT()
        
        except optExceptions.BAD_STEP_EXCEPT:
            print "optimize.py: Caught bad step exception."   
            op.Params.dynamic_level += 1
            if op.Params.dynamic_level == op.Params.dynamic_level_max:
                print 'dynamic_level (%d) may not be further increased.' % op.Params.dynamic_level
            else:   # keep going
                print "increasing dynamic_level."
                print "Erasing old history, hessian, intcos."
                del H 
                for f in Molsys._fragments:
                    del f._intcos
                del history.History
                op.Params.updateDynamicLevelParameters(op.Params.dynamic_level)
    
    history.History.summary()
    return history.History[-1].E


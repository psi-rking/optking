
def optimize( Molsys, options_in, fSetGeometry, fGradient, fHessian, fEnergy):
    
    import caseInsensitiveDict
    userOptions = caseInsensitiveDict.CaseInsensitiveDict( options_in )
    # Save copy of original user options.
    origOptions = userOptions.copy()

    from printTools import printGeomGrad,printMat,printArray,print_opt

    # Create full list of parameters from defaults plus user options.
    import optParams as op
    op.welcome()  # print header
    print_opt("\tProcessing user input options...\n")
    op.Params = op.OPT_PARAMS(userOptions)
    print_opt("\tParameters from optking.optimize\n")
    print_opt( str(op.Params) )

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
            print_opt( str(Molsys) )
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

            addIntcos.addFrozenAndFixedIntcos(Molsys)

            Molsys.printIntcos()

            # Test Hessian transformations.  cartesians -> internals -> cartesians -> internals
            # Cartesians do not satisy constraints such as frozen COM (undetermined problem)
            """
            xyz = Molsys.geom.copy()
            E, grad = fGradient(xyz, printResults=True)
            Molsys.geom = xyz # use setter function to save data in fragments
            Hcart = fHessian(xyz, printResults=False)  # assuming geometry doesn't change
            Hint = intcosMisc.convertHessianToInternals(Hcart, Molsys.intcos, Molsys.geom, masses=None, g_x=grad)
            print_opt("Internal Hessian\n")
            printMat(Hint)
            G = intcosMisc.Gmat(Molsys.intcos, Molsys.geom, masses=None)
            import linearAlgebra
            Ginv = linearAlgebra.symmMatInv(G)
            B = intcosMisc.Bmat(Molsys.intcos, Molsys.geom, masses=None)
            import numpy as np
            Atranspose = np.dot(Ginv, B)
            q_grad = np.dot(Atranspose, grad)
            Hcart2 = intcosMisc.convertHessianToCartesians(Hint, Molsys.intcos, Molsys.geom, masses=None, g_q=q_grad)
            print_opt("Cartesian Hessian from internals\n")
            printMat(Hcart2)
            Hint2 = intcosMisc.convertHessianToInternals(Hcart2, Molsys.intcos, Molsys.geom, masses=None, g_x=grad)
            print_opt("Internal Hessian from Cartesian\n")
            printMat(Hint2)
            quit()
            """
        
            for stepNumber in range(op.Params.geom_maxiter): 
                xyz = Molsys.geom.copy()
                E, g_x = fGradient(xyz, printResults=False)
                Molsys.geom = xyz # use setter function to save data in fragments
                printGeomGrad(Molsys.geom, g_x)
                energies.append( E )
            
            
                if op.Params.test_B:
                    testB.testB(Molsys.intcos, Molsys.geom)
                if op.Params.test_derivative_B:
                    testB.testDerivativeB(Molsys.intcos, Molsys.geom)
            
                if op.Params.print_lvl > 3:
                    B = intcosMisc.Bmat(Molsys.intcos, Molsys.geom)
                    print_opt("B matrix:\n")
                    printMat(B)
        
                fq = intcosMisc.qForces(Molsys.intcos, Molsys.geom, g_x)
                if (op.Params.print_lvl > 1):
                    print_opt("Internal forces in au\n")
                    printArray(fq)
            
                history.History.append(Molsys.geom, E, fq); # Save initial step info.
            
                history.History.currentStepReport()

                if stepNumber == 0:
                    C = addIntcos.connectivityFromDistances(Molsys.geom, Molsys.Z)
                            
                if stepNumber == 0:
                    if op.Params.full_hess_every > -1:
                        xyz = Molsys.geom.copy()
                        Hcart = fHessian(xyz, printResults=False) # it's possible function moves geometry
                        H = intcosMisc.convertHessianToInternals(Hcart, Molsys.intcos, xyz, masses=None)
                        print "compute hessian"
                    else:
                        H = hessian.guess(Molsys.intcos, Molsys.geom, Molsys.Z, C, op.Params.intrafrag_hess)
                        print "guess hessian"
                else:  # not first step
                    if op.Params.full_hess_every < 1: # compute hessian never or only once
                        history.History.hessianUpdate(H, Molsys.intcos)
                        print "update hessian"
                    elif stepNumber % op.Params.full_hess_every == 0:
                        xyz = Molsys.geom.copy()
                        Hcart = fHessian(xyz, printResults=False) # it's possible function moves geometry
                        H = intcosMisc.convertHessianToInternals(Hcart, Molsys.intcos, xyz, masses=None)
                        print "compute hessian"
                    else:
                        history.History.hessianUpdate(H, Molsys.intcos)
                        print "update hessian"
            
                print_opt("Hessian (in au) is:\n")
                printMat(H)
                print_opt("Hessian in aJ/Ang^2 or aJ/deg^2\n")
                hessian.show(H, Molsys.intcos)
        
                intcosMisc.applyFixedForces(Molsys, fq, H, stepNumber)
                intcosMisc.projectRedundanciesAndConstraints(Molsys.intcos, Molsys.geom, fq, H)
        
                try:
                    if (op.Params.opt_type == 'IRC'):
                        Dq = IRCFollowing.Dq_IRC(Molsys, intcos, geom E, g, H, B, s, op.Params.irc_direction, stepNumber, fgradient)
                    else:
                    # displaces and adds step to history
                    Dq = stepAlgorithms.Dq(Molsys, E, fq, H, op.Params.step_type)
                except optExceptions.BAD_STEP_EXCEPT:
                    if history.History.consecutiveBacksteps < op.Params.consecutiveBackstepsAllowed:
                        print_opt("Taking backward step.\n")
                        Dq = stepAlgorithms.Dq(Molsys.intcos, Molsys.geom, E, fq, H, stepType="BACKSTEP")
                    else:
                        print_opt("Maximum number of backsteps has been attempted.\n")
                        print_opt("Re-raising BAD_STEP exception.\n")
                        raise optExceptions.BAD_STEP_EXCEPT()
        
                converged = convCheck.convCheck(stepNumber, Molsys.intcos, Dq, fq, energies)

                if converged:
                    print_opt("\tConverged in %d steps!\n" % (stepNumber+1))
                    print_opt("\tFinal energy is %20.13f\n" % E)
                    print_opt("\tFinal structure (Angstroms):\n")
                    Molsys.showGeom()
                    break

                print_opt("\tStructure for next step (au):\n")
                Molsys.printGeom()

            else: # executes if too many steps
                print_opt("Number of steps (%d) has reached value of GEOM_MAXITER.\n" % (stepNumber+1))
                raise optExceptions.BAD_STEP_EXCEPT()
        
        except optExceptions.BAD_STEP_EXCEPT:
            print_opt("optimize.py: Caught bad step exception.\n")   
            op.Params.dynamic_level += 1
            if op.Params.dynamic_level == op.Params.dynamic_level_max:
                print_opt("dynamic_level (%d) may not be further increased.\n" % (op.Params.dynamic_level-1))
            else:   # keep going
                print_opt("increasing dynamic_level.\n")
                print_opt("Erasing old history, hessian, intcos.\n")
                del H 
                for f in Molsys._fragments:
                    del f._intcos[:]
                del history.History[:] # delete steps in history
                history.History.stepsSinceLastHessian = 0
                history.History.consecutiveBacksteps = 0
                op.Params.updateDynamicLevelParameters(op.Params.dynamic_level)
    
    # print summary
    history.History.summary()
    energy = history.History[-1].E

    # clean up
    del H
    for f in Molsys._fragments:
        del f._intcos[:]
        del f
    del history.History[:]
    del op.Params

    return energy


    
def welcome():
    print_out("\n\t\t\t-----------------------------------------\n")
    print_out(  "\t\t\t OPTKING 3.0: for geometry optimizations \n")
    print_out(  "\t\t\t     By R.A. King, Bethel University     \n")
    print_out(  "\t\t\t        with contributions from          \n")
    print_out(  "\t\t\t    A.V. Copan, J. Cayton, A. Heide      \n")
    print_out(  "\t\t\t-----------------------------------------\n")


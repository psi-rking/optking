def optimize(Molsys, options_in, fSetGeometry, fGradient, fHessian, fEnergy):

    from . import caseInsensitiveDict
    userOptions = caseInsensitiveDict.CaseInsensitiveDict(options_in)
    origOptions = userOptions.copy()  # Save copy of original user options.

    from .printTools import printGeomGrad, printMat, printArray, print_opt

    # Create full list of parameters from user options plus defaults.
    from . import optParams as op
    op.welcome()  # print header
    print_opt("\tProcessing user input options...\n")
    op.Params = op.OPT_PARAMS(userOptions)
    print_opt("\tParameters from optking.optimize\n")
    print_opt(str(op.Params))

    from . import addIntcos
    from . import optExceptions
    from . import history
    from . import stepAlgorithms
    from . import intcosMisc
    from . import hessian
    from . import convCheck
    from . import testB
    from . import IRCFollowing
    converged = False

    # For IRC computations:
    ircNumber = 0
    qPivot = None  # Dummy argument for non-IRC

    while not converged:  # may contain multiple algorithms

        try:
            print_opt("Starting optimization algorithm.\n")
            print_opt(str(Molsys))

            # Set internal or cartesian coordinates.
            if not Molsys.intcos:
                C = addIntcos.connectivityFromDistances(Molsys.geom, Molsys.Z)

                if op.Params.frag_mode == 'SINGLE':
                    # Splits existing fragments if they are not connected.
                    Molsys.splitFragmentsByConnectivity()
                    # Add to connectivity to make sure all fragments connected.
                    Molsys.augmentConnectivityToSingleFragment(C)
                    #print_opt("Connectivity\n")
                    #printMat(C)
                    # Bring fragments together into one.
                    Molsys.consolidateFragments()
                elif op.Params.frag_mode == 'MULTI':
                    # should do nothing if fragments are already split by calling program/constructor.
                    Molsys.splitFragmentsByConnectivity()

                if op.Params.opt_coordinates in ['REDUNDANT', 'BOTH']:
                    Molsys.addIntcosFromConnectivity(C)

                if op.Params.opt_coordinates in ['CARTESIAN', 'BOTH']:
                    Molsys.addCartesianIntcos()

                addIntcos.addFrozenAndFixedIntcos(Molsys)
                Molsys.printIntcos()

            # Special code for first step of IRC.  Compute Hessian and take step along eigenvector.
            if op.Params.opt_type == 'IRC' and ircNumber == 0:
                ircStepList = []  #Holds data points for IRC steps
                xyz = Molsys.geom.copy()
                print_opt("Initial Cartesian Geom")
                printMat(xyz)
                qZero = intcosMisc.qValues(Molsys.intcos, Molsys.geom)
                print_opt("Initial internal coordinates\n")
                printArray(qZero)
                Hcart = fHessian(xyz, printResults=False)
                e, gX = fGradient(xyz)
                fq = intcosMisc.qForces(Molsys.intcos, Molsys.geom, gX)
                Hq = intcosMisc.convertHessianToInternals(Hcart, Molsys.intcos, xyz)
                B = intcosMisc.Bmat(Molsys.intcos, Molsys.geom, Molsys.masses)
                dqPrime, qPivot, qPrime = IRCFollowing.takeHessianHalfStep(
                    Molsys, Hq, B, fq, op.Params.irc_step_size)

            # Loop over geometry steps.
            energies = []  # should be moved into history
            for stepNumber in range(op.Params.geom_maxiter):
                # compute energy and gradient
                xyz = Molsys.geom.copy()
                E, g_x = fGradient(xyz, printResults=False)
                Molsys.geom = xyz  # use setter function to save data in fragments
                printGeomGrad(Molsys.geom, g_x)
                energies.append(E)

                if op.Params.test_B:
                    testB.testB(Molsys.intcos, Molsys.geom)
                if op.Params.test_derivative_B:
                    testB.testDerivativeB(Molsys.intcos, Molsys.geom)

                if op.Params.print_lvl > 3:
                    B = intcosMisc.Bmat(Molsys.intcos, Molsys.geom)
                    printMat(B, title="B matrix")

                fq = intcosMisc.qForces(Molsys.intcos, Molsys.geom, g_x)
                if op.Params.print_lvl > 1:
                    printArray(fq, title="Internal forces in au")

                history.History.append(Molsys.geom, E, fq)  # Save initial step info.

                # Analyze previous step performance; adjust trust radius accordingly.
                # Returns true on first step (no history)
                lastStepOK = history.History.currentStepReport()

                # If step was bad, take backstep here or raise exception.
                if lastStepOK:
                    history.HISTORY.consecutiveBacksteps = 0
                else:
                    # Don't go backwards until we've gone a few iterations.
                    if len(history.History.steps) < 5:
                        print_opt("\tNear start of optimization, so ignoring bad step.\n")
                    elif history.HISTORY.consecutiveBacksteps < op.Params.consecutiveBackstepsAllowed:
                        print_opt("\tTaking backward step.\n")
                        Dq = stepAlgorithms.Dq(
                            Molsys.intcos, Molsys.geom, E, fq, H, stepType="BACKSTEP")
                        history.HISTORY.consecutiveBacksteps += 1
                        continue
                    else:
                        raise optExceptions.ALG_FAIL(
                            "Bad step, and no more backsteps allowed.")

                # Produce guess Hessian or update existing Hessian..
                if stepNumber == 0:
                    if op.Params.full_hess_every > -1:
                        xyz = Molsys.geom.copy()
                        Hcart = fHessian(
                            xyz, printResults=False)  # don't let function move geometry
                        H = intcosMisc.convertHessianToInternals(
                            Hcart, Molsys.intcos, xyz, Molsys.masses)
                    else:
                        H = hessian.guess(Molsys.intcos, Molsys.geom, Molsys.Z, C,
                                          op.Params.intrafrag_hess)
                else:
                    if op.Params.full_hess_every < 1:  # that is, compute hessian never or only once.
                        history.History.hessianUpdate(H, Molsys.intcos)
                    elif stepNumber % op.Params.full_hess_every == 0:
                        xyz = Molsys.geom.copy()
                        Hcart = fHessian(
                            xyz,
                            printResults=False)  # it's possible function moves geometry
                        H = intcosMisc.convertHessianToInternals(
                            Hcart, Molsys.intcos, xyz, masses=None)
                    else:
                        history.History.hessianUpdate(H, Molsys.intcos)
                    #print_opt("Hessian (in au) is:\n")
                    #printMat(H)
                    #print_opt("Hessian in aJ/Ang^2 or aJ/deg^2\n")
                    #hessian.show(H, Molsys.intcos)

                    # handle user defined forces, redundances and constraints
                intcosMisc.applyFixedForces(Molsys, fq, H, stepNumber)
                intcosMisc.projectRedundanciesAndConstraints(Molsys.intcos, Molsys.geom,
                                                             fq, H)
                intcosMisc.qShowValues(Molsys.intcos, Molsys.geom)

                if op.Params.opt_type == 'IRC':
                    xyz = Molsys.geom.copy()
                    E, g = fGradient(xyz, False)
                    Dq = IRCFollowing.Dq(Molsys, g, E, Hq, B, op.Params.irc_step_size,
                                         qPrime, dqPrime)
                else:  # Displaces and adds step to history.
                    Dq = stepAlgorithms.Dq(Molsys, E, fq, H, op.Params.step_type, fEnergy)

                converged = convCheck.convCheck(stepNumber, Molsys, Dq, fq, energies,
                                                qPivot)

                if converged and (op.Params.opt_type == 'IRC'):
                    converged = False
                    #add check for minimum
                    if atMinimum:
                        converged = True
                        break
                elif converged:
                    print_opt("\tConverged in %d steps!\n" % (stepNumber + 1))
                    print_opt("\tFinal energy is %20.13f\n" % E)
                    print_opt("\tFinal structure (Angstroms):\n")
                    Molsys.showGeom()
                    break

                print_opt("\tStructure for next step (au):\n")
                Molsys.printGeom()

            else:  # executes if step limit is reached
                print_opt("Number of steps (%d) exceeds maximum allowed (%d).\n" %
                          (stepNumber + 1, op.Params.geom_maxiter))
                raise optExceptions.ALG_FAIL("Maximum number of steps exceeded.")

            #This should be called at the end of each iteration of the for loop,
            if (op.Params.opt_type == 'IRC') and (not atMinimum):
                ircNumber += 1
                xyz = Molsys.geom.copy()
                ircStepsList.append(ircStep.IRCStep(qPivot, xyz, ircNumber))
                history.History.hessianUpdate(H, Molsys.intcos)
                Hq = H
                E, gX = fGradient(xyz, printResults=False)
                B = intcosMisc.Bmat(Molsys.intcos, Molsys.geom, Molsys.masses)
                qPivot, qPrime, Dq = IRCFollowing.takeGradientHalfStep(
                    Molsys, E, Hq, B, op.Params.irc_step_size, gX)

        except optExceptions.ALG_FAIL as AF:
            print_opt("\tCaught ALG_FAIL exception\n")
            eraseHistory = False
            eraseIntcos = False

            if AF.linearBends:  # New linear bends detected; Add them, and continue at current level.
                from . import bend
                for l in AF.linearBends:
                    if l.bendType == "LINEAR":  # no need to repeat this code for "COMPLEMENT"
                        F = addIntcos.checkFragment(l.atoms, Molsys)
                        intcosMisc.removeOldNowLinearBend(l.atoms,
                                                          Molsys._fragments[F].intcos)
                Molsys.addIntcosFromConnectivity()
                eraseHistory = True
            elif op.Params.dynamic_level >= (op.Params.dynamic_level_max - 1):
                print_opt("\t Current approach/dynamic_level is %d.\n" %
                          op.Params.dynamic_level)
                print_opt("\t Alternative approaches are not available or turned on.\n")
                raise optExceptions.OPT_FAIL("Maximum dynamic_level exceeded.")
            else:
                op.Params.dynamic_level += 1
                print_opt("\t Increasing dynamic_level algorithm to %d.\n" %
                          op.Params.dynamic_level)
                print_opt("\t Erasing old history, hessian, intcos.\n")
                eraseIntcos = True
                eraseHistory = True
                op.Params.updateDynamicLevelParameters(op.Params.dynamic_level)

            if eraseIntcos:
                print_opt("\t Erasing coordinates.\n")
                for f in Molsys._fragments:
                    del f._intcos[:]

            if eraseHistory:
                print_opt("\t Erasing history.\n")
                stepNumber = 0
                del H
                del history.History[:]  # delete steps in history
                history.History.stepsSinceLastHessian = 0
                history.History.consecutiveBacksteps = 0

    # print summary
    history.History.summary()

    if op.Params.trajectory:
        # history doesn't contain atomic numbers so pass them in
        returnVal = history.History.trajectory(Molsys.Z)
    else:
        returnVal = history.History[-1].E

    # clean up
    del H
    for f in Molsys._fragments:
        del f._intcos[:]
        del f
    del history.History[:]
    del op.Params

    return returnVal


def welcome():
    print_out("\n\t\t\t-----------------------------------------\n")
    print_out("\t\t\t OPTKING 3.0: for geometry optimizations \n")
    print_out("\t\t\t     By R.A. King, Bethel University     \n")
    print_out("\t\t\t        with contributions from          \n")
    print_out("\t\t\t    A.V. Copan, J. Cayton, A. Heide      \n")
    print_out("\t\t\t-----------------------------------------\n")

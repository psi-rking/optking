import numpy as np
# import copy
import logging
from pprint import PrettyPrinter

from psi4.driver import json_wrapper  # COMMENT FOR INDEP DOCS BUILD

from . import hessian
from . import stepAlgorithms
from . import caseInsensitiveDict
from . import optparams as op
from . import optExceptions
from . import addIntcos
from . import history
from . import intcosMisc
from . import convCheck
from . import testB
# import IRCFollowing
from . import psi4methods
from .qcdbjson import jsonSchema
from .printTools import (printGeomGrad,
                         printMatString,
                         printArrayString,
                         welcome)

pp = PrettyPrinter(indent=4)


def optimize(oMolsys, options_in, json_in=None):
    """Logical 'driver' for optking's optimization procedure takes in optking's
    moolecular system a list of options for both the optimizer and the QM
    program with the options of passing a json_file in directly.
    """

    try:
        optimize_log = logging.getLogger(__name__)

        userOptions = caseInsensitiveDict.CaseInsensitiveDict(options_in)
        # Save copy of original user options. Commented out because it was never used
        # origOptions = copy.deepcopy(userOptions)

        # Create full list of parameters from user options plus defaults.
        optimize_log.info(welcome())  # print header
        optimize_log.info("\n\tProcessing user input options...\n")
        op.Params = op.optParams(userOptions)
        optimize_log.info(str(op.Params))

        # if op.Params.logging_lvl == 'DEBUG' or op.Params.print_lvl == 5:
        #    log_file.setLevel(logging.DEBUG)
        # elif op.Params.logging_lvl == 'INFO' or op.Params.print_lvl >= 3:

        op.Params.dynamic_level_max = 3

        converged = False

        # generates a json dictionary if optking is not being called directly by json.
        # other option would be to have the wrapper make a JSON object and make json
        # be a requirement of calling optking. If we're adding additional ways to call optking
        # this if json_in is None will be refactored into a method within optimize
        o_json = 0
        if json_in is None:
            QM_method, basis, keywords = psi4methods.collect_psi4_options(options_in)
            atom_list = oMolsys.get_atom_list()
            o_json = jsonSchema.make_qcschema("", atom_list, QM_method, basis, keywords)
        else:
            o_json = json_in
            op.Params.output_type = 'JSON'

        # For IRC computations:
        # ircNumber = 0
        qPivot = None  # Dummy argument for non-IRC

        while not converged:  # may contain multiple algorithms

            try:
                optimize_log.info("\tStarting optimization algorithm.\n")
                optimize_log.info(str(oMolsys))

                # Set internal or cartesian coordinates.
                if not oMolsys.intcos:
                    C = addIntcos.connectivityFromDistances(oMolsys.geom, oMolsys.Z)
                    optimize_log.debug("Connectivity Matrix\n" + printMatString(C))
                    if op.Params.frag_mode == 'SINGLE':
                        # Splits existing fragments if they are not connected.
                        oMolsys.splitFragmentsByConnectivity()
                        # Add to connectivity to make sure all fragments connected.
                        oMolsys.augmentConnectivityToSingleFragment(C)
                        # print_opt("Connectivity\n")
                        # printMat(C)
                        # Bring fragments together into one.
                        oMolsys.consolidateFragments()
                    elif op.Params.frag_mode == 'MULTI':
                        # should do nothing if fragments are already split by calling
                        # program / constructor.
                        oMolsys.splitFragmentsByConnectivity()

                    if op.Params.opt_coordinates in ['REDUNDANT', 'BOTH']:
                        oMolsys.addIntcosFromConnectivity(C)

                    if op.Params.opt_coordinates in ['CARTESIAN', 'BOTH']:
                        oMolsys.addCartesianIntcos()

                    addIntcos.addFrozenAndFixedIntcos(oMolsys)
                    oMolsys.printIntcos()

                # Special code for first step of IRC. Compute Hessian and take step along eigen vec
                # if op.Params.opt_type == 'IRC' and ircNumber == 0:
                #    ircStepList = []  #Holds data points for IRC steps
                #    xyz = oMolsys.geom.copy()
                #    print_opt("Initial Cartesian Geom")
                #    printMat(xyz)
                #    qZero = intcosMisc.qValues(oMolsys.intcos, oMolsys.geom)
                #    print_opt("Initial internal coordinates\n")
                #    printArray(qZero)
                #    Hcart = get_hessian(xyz, o_json, printResults=False)
                #    e, gX = get_gradient(xyz)
                #    fq = intcosMisc.qForces(oMolsys.intcos, oMolsys.geom, gX)
                #    Hq = intcosMisc.convertHessianToInternals(Hcart, oMolsys.intcos, xyz)
                #    B = intcosMisc.Bmat(oMolsys.intcos, oMolsys.geom, oMolsys.masses)
                #    dqPrime, qPivot, qPrime = IRCFollowing.takeHessianHalfStep(
                #        oMolsys, Hq, B, fq, op.Params.irc_step_size)

                # Loop over geometry steps.
                energies = []  # should be moved into history
                for stepNumber in range(op.Params.geom_maxiter):
                    # compute energy and gradient
                    xyz = oMolsys.geom.copy()
                    E, g_x, nuc = get_gradient(xyz, o_json, printResults=False, nuc=True)
                    oMolsys.geom = xyz  # use setter function to save data in fragments
                    printGeomGrad(oMolsys.geom, g_x)
                    energies.append(E)

                    if op.Params.test_B:
                        testB.testB(oMolsys.intcos, oMolsys.geom)
                    if op.Params.test_derivative_B:
                        testB.testDerivativeB(oMolsys.intcos, oMolsys.geom)

                    if op.Params.print_lvl > 3:
                        B = intcosMisc.Bmat(oMolsys.intcos, oMolsys.geom)
                        optimize_log.info(printMatString(B, title="B matrix"))

                    fq = intcosMisc.qForces(oMolsys.intcos, oMolsys.geom, g_x)
                    if op.Params.print_lvl > 1:
                        optimize_log.info(printArrayString(fq, title="Internal forces in au"))

                    history.oHistory.append(oMolsys.geom, E, fq)  # Save initial step info.
                    history.oHistory.nuclear_repulsion_energy = nuc
                    # Analyze previous step performance; adjust trust radius accordingly.
                    # Returns true on first step (no history)
                    lastStepOK = history.oHistory.currentStepReport()

                    # If step was bad, take backstep here or raise exception.
                    if lastStepOK:
                        history.History.consecutiveBacksteps = 0
                    else:
                        # Don't go backwards until we've gone a few iterations.
                        if len(history.oHistory.steps) < 5:
                            optimize_log.info(
                                    "\tNear start of optimization, so ignoring bad step.\n")
                        elif history.History.consecutiveBacksteps < op.Params.consecutiveBackstepsAllowed:
                            history.History.consecutiveBacksteps += 1
                            optimize_log.info("\tCalling for consecutive backstep number %d.\n"
                                              % history.History.consecutiveBacksteps)
                            # IDE complains about H not being delcared. Should be fine
                            Dq = stepAlgorithms.Dq(oMolsys, E, fq, H, stepType="BACKSTEP")
                            optimize_log.info("\tStructure for next step (au):\n")
                            oMolsys.showGeom()
                            continue
                        elif op.Params.dynamic_level == 0:  # not using dynamic level, so ignore.
                            optimize_log.info("\tNo more backsteps allowed."
                                              + "Dynamic level is off.\n")
                            pass
                        else:
                            raise optExceptions.AlgFail(
                                "Bad step, and no more backsteps allowed.")

                    # Produce guess Hessian or update existing Hessian..
                    if stepNumber == 0:
                        if op.Params.full_hess_every > -1:
                            xyz = oMolsys.geom.copy()
                            Hcart = get_hessian(
                                xyz, o_json, printResults=True)  # don't let function move geometry
                            H = intcosMisc.convertHessianToInternals(
                                Hcart, oMolsys.intcos, xyz, masses=None)
                        else:
                            H = hessian.guess(oMolsys.intcos, oMolsys.geom, oMolsys.Z, C,
                                              op.Params.intrafrag_hess)
                            if op.Params.print_lvl >= 4:
                                optimize_log.info(printMatString(H, title="Initial Hessian Guess"))
                    else:
                        # that is, compute hessian never or only once.
                        if op.Params.full_hess_every < 1:
                            history.oHistory.hessianUpdate(H, oMolsys.intcos)
                        elif stepNumber % op.Params.full_hess_every == 0:
                            xyz = oMolsys.geom.copy()
                            Hcart = get_hessian(
                                xyz, o_json,
                                printResults=False)  # it's possible function moves geometry
                            H = intcosMisc.convertHessianToInternals(
                                Hcart, oMolsys.intcos, xyz, masses=None)
                        else:
                            history.oHistory.hessianUpdate(H, oMolsys.intcos)
                        # print_opt("Hessian (in au) is:\n")
                        # printMat(H)
                        # print_opt("Hessian in aJ/Ang^2 or aJ/deg^2\n")
                        # hessian.show(H, oMolsys.intcos)

                        # handle user defined forces, redundances and constraints
                    intcosMisc.applyFixedForces(oMolsys, fq, H, stepNumber)
                    intcosMisc.projectRedundanciesAndConstraints(oMolsys.intcos, oMolsys.geom,
                                                                 fq, H)
                    intcosMisc.qShowValues(oMolsys.intcos, oMolsys.geom)

                    # if op.Params.opt_type == 'IRC':
                    #    xyz = Molsys.geom.copy()
                    #    E, g = get_gradient(xyz, False)
                    #    Dq = IRCFollowing.Dq(oMolsys, g, E, Hq, B, op.Params.irc_step_size,
                    #                         qPrime, dqPrime)
                    # else:  # Displaces and adds step to history.
                    Dq = stepAlgorithms.Dq(oMolsys, E, fq, H, op.Params.step_type,
                                           get_energy, o_json)
                    # else statement paried with above IRC if was removed

                    converged = convCheck.convCheck(stepNumber, oMolsys, Dq, fq, energies,
                                                    qPivot)

                    # if converged and (op.Params.opt_type == 'IRC'):
                    #    converged = False
                    #    #add check for minimum
                    #    if atMinimum:
                    #        converged = True
                    #        break
                    if converged:  # changed from elif when above if statement active
                        optimize_log.info("\tConverged in %d steps!" % (stepNumber + 1))
                        optimize_log.info("\tFinal energy is %20.13f" % E)
                        optimize_log.info("\tFinal structure (Angstroms): \n\n"
                                          + oMolsys.showGeom())
                        break

                    optimize_log.info("\tStructure for next step (au):\n" + oMolsys.showGeom())

                else:  # executes if step limit is reached
                    optimize_log.error("\tNumber of steps (%d) exceeds maximum allowed (%d).\n"
                                       % (stepNumber + 1, op.Params.geom_maxiter))
                    history.oHistory.summary()
                    raise optExceptions.AlgFail("Maximum number of steps exceeded.")

                # This should be called at the end of each iteration of the for loop,
                # if (op.Params.opt_type == 'IRC') and (not atMinimum):
                #    ircNumber += 1
                #    xyz = oMolsys.geom.copy()
                #    ircStepsList.append(ircStep.IRCStep(qPivot, xyz, ircNumber))
                #    history.oHistory.hessianUpdate(H, oMolsys.intcos)
                #    Hq = H
                #    E, gX = get_gradient(xyz, printResults=False)
                #    B = intcosMisc.Bmat(oMolsys.intcos, oMolsys.geom, oMolsys.masses)
                #    qPivot, qPrime, Dq = IRCFollowing.takeGradientHalfStep(
                #        oMolsys, E, Hq, B, op.Params.irc_step_size, gX)

            except optExceptions.AlgFail as AF:
                optimize_log.error("\n\tCaught AlgFail exception\n")
                eraseHistory = False
                eraseIntcos = False

                if AF.linearBends:
                    # New linear bends detected; Add them, and continue at current level.
                    # from . import bend # import not currently being used according to IDE
                    for l in AF.linearBends:
                        if l.bendType == "LINEAR":  # no need to repeat this code for "COMPLEMENT"
                            F = addIntcos.checkFragment(l.atoms, oMolsys)
                            intcosMisc.removeOldNowLinearBend(l.atoms,
                                                              oMolsys._fragments[F].intcos)
                    oMolsys.addIntcosFromConnectivity()
                    eraseHistory = True
                elif op.Params.dynamic_level >= (op.Params.dynamic_level_max - 1):
                    optimize_log.critical("\n\t Current approach/dynamic_level is %d.\n"
                                          % op.Params.dynamic_level)
                    optimize_log.critical("\n\t Alternative approaches are not available or"
                                          + "turned on.\n")
                    raise optExceptions.OptFail("Maximum dynamic_level exceeded.")
                else:
                    op.Params.dynamic_level += 1
                    optimize_log.warning("\n\t Increasing dynamic_level algorithm to %d.\n"
                                         % op.Params.dynamic_level)
                    optimize_log.warning("\n\t Erasing old history, hessian, intcos.\n")
                    eraseIntcos = True
                    eraseHistory = True
                    op.Params.updateDynamicLevelParameters(op.Params.dynamic_level)

                if eraseIntcos:
                    optimize_log.warning("\n\t Erasing coordinates.\n")
                    for f in oMolsys._fragments:
                        del f._intcos[:]

                if eraseHistory:
                    optimize_log.warning("\n\t Erasing history.\n")
                    stepNumber = 0
                    del H
                    del history.oHistory[:]  # delete steps in history
                    history.oHistory.stepsSinceLastHessian = 0
                    history.oHistory.consecutiveBacksteps = 0

        output_dict = {}
        # print summary
        if op.Params.output_type == 'FILE':
            logging.info("\tOptimization Finished\n" + history.generate_file_output())
        else:
            output_dict = o_json.generate_json_output(history.oHistory[-1].geom)

        if op.Params.trajectory:
            # history doesn't contain atomic numbers so pass them in
            returnVal = history.oHistory.trajectory(oMolsys.Z)
        else:
            returnVal = history.oHistory[-1].E

        returnNuc = history.oHistory.nuclear_repulsion_energy

        # clean up
        del H
        for f in oMolsys._fragments:
            del f._intcos[:]
            del f
        del history.oHistory[:]
        del oMolsys

        if op.Params.output_type == 'FILE':
            del op.Params
            del o_json
            return returnVal, returnNuc
        else:
            del op.Params
            del o_json
            return output_dict

    except Exception as error:
        # TODO needs some improvements
        optimize_log.critical("\tA non-optimization error has occured\n")
        optimize_log.critical(("\tResetting all optimzation options to prevent queued" +
                               "optimizations from failing\n"))
        optimize_log.exception("Error Type:" + str(type(error)))
        optimize_log.exception("Error caught:" + str(error))
        del history.oHistory[:]
        del o_json
        # had been deleting oMolsys and Fragments here but IDE complained they
        # were not declared
        del op.Params


def get_gradient(new_geom, o_json, printResults=False, nuc=True, QM='psi4'):
    """Use JSON interface to have QM program perform gradient calculation
    Only Psi4 is currently implemented
    """

    json_output = psi4methods.psi4_calculation(new_geom, o_json)
    return o_json.get_JSON_result(json_output, 'gradient', nuc)


def get_hessian(new_geom, o_json, printResults=False, QM='psi4'):
    """Use JSON interface to have QM program perform hessian calculation
    Only Psi4 is currently implemented
    """
    json_output = psi4methods.psi4_calculation(new_geom, o_json, driver="hessian")
    H = np.array(o_json.get_JSON_result(json_output, 'hessian'))
    return H


def get_energy(new_geom, o_json, printResults=False, nuc=True, QM='psi4'):
    """ Use JSON interface to have QM program perform energy calculation
    Only psi4 is current implemented
    """

    json_output = psi4methods.psi4_calculation(new_geom, o_json, driver='energy')
    return o_json.get_JSON_result(json_output, 'energy', nuc)

import numpy as np
import copy
import logging

from psi4.driver import json_wrapper  # COMMENT FOR INDEP DOCS BUILD

from . import hessian
from . import stepAlgorithms
from . import caseInsensitiveDict
from . import optparams as op
from .exceptions import OptError, AlgError, IRCendReached
from . import addIntcos
from . import history
from . import intcosMisc
from . import convCheck
from . import testB
from . import IRCfollowing
from . import psi4methods
from . import IRCdata
from .linearAlgebra import lowestEigenvectorSymmMat, symmMatRoot, symmMatInv
from .qcdbjson import jsonSchema
from .printTools import (printGeomGrad,
                         printMatString,
                         printArrayString,
                         welcome)

def optimize(oMolsys, options_in, json_in=None):
    """Driver for OptKing's optimization procedure

    Parameters
    ----------
    oMolsys : cls
        optking molecular system
    options_in : dict
        options for QM program and optking
    json_in : dict, optional
        MolSSI qc schema

    Returns
    -------
    float, float or dict
        energy and nuclear repulsion energy or MolSSI qc_schema_output as dict

    """

    try:  # Try to optimize one structure OR set of IRC points. OptError and all Exceptions caught below.
        optimize_log = logging.getLogger(__name__)

        userOptions = caseInsensitiveDict.CaseInsensitiveDict(options_in)
        # Save copy of original user options. Commented out until it is used
        # origOptions = copy.deepcopy(userOptions)

        # Create full list of parameters from user options plus defaults.
        optimize_log.info(welcome())
        optimize_log.debug("\n\tProcessing user input options...\n")
        op.Params = op.OptParams(userOptions)
        optimize_log.debug(str(op.Params))

        # Construct a json dictionary if optking was not provided one.
        o_json = 0
        if json_in is None:
            QM_method, basis, keywords = psi4methods.collect_psi4_options(options_in)
            o_json = jsonSchema.make_qcschema(oMolsys.geom, oMolsys.atom_symbols,
                         QM_method, basis, keywords, oMolsys.multiplicity)
        else:
            o_json = json_in

        # Prepare for multiple IRC computation
        if op.Params.opt_type == 'IRC':
            IRCstepNumber = 0
            IRCdata.history = IRCdata.IRCdata()
            IRCdata.history.set_atom_symbols(oMolsys.atom_symbols)
            IRCdata.history.set_step_size_and_direction(op.Params.irc_step_size, op.Params.irc_direction)
            optimize_log.info("\tIRC data object created\n")

        converged = False
        totalStepsTaken = -1
        # following loop may repeat over multiple algorithms OR over IRC points
        while not converged:
            try:
                optimize_log.info("Starting optimization algorithm.\n")
                optimize_log.info(str(oMolsys))

                # if optimization coordinates are absent, choose them.
                if not oMolsys.intcos:
                    connectivity = addIntcos.connectivityFromDistances(oMolsys.geom, oMolsys.Z)
                    optimize_log.debug("Connectivity Matrix\n" + printMatString(connectivity))

                    if op.Params.frag_mode == 'SINGLE':
                        oMolsys.splitFragmentsByConnectivity()
                        oMolsys.augmentConnectivityToSingleFragment(connectivity)
                        oMolsys.consolidateFragments()
                    elif op.Params.frag_mode == 'MULTI':
                        oMolsys.splitFragmentsByConnectivity() # does nothing if already split

                    if op.Params.opt_coordinates in ['REDUNDANT', 'BOTH']:
                        oMolsys.addIntcosFromConnectivity(connectivity)

                    if op.Params.opt_coordinates in ['CARTESIAN', 'BOTH']:
                        oMolsys.addCartesianIntcos()

                    addIntcos.addFrozenAndFixedIntcos(oMolsys) # make sure these are in the set
                    oMolsys.printIntcos()

                # Do special initial step-0 for each IRC point.
                # For IRC point, we form/get the Hessian now.
                if op.Params.opt_type == 'IRC':
                    if IRCstepNumber == 0: # Step along lowest eigenvector of mass-weighted Hessian.
                        optimize_log.info("Beginning IRC from the transition state.\n")
                        optimize_log.info("Stepping along lowest Hessian eigenvector.\n")

                        #C = addIntcos.connectivityFromDistances(oMolsys.geom, oMolsys.Z)
                        #H = hessian.guess(oMolsys.intcos, oMolsys.geom, oMolsys.Z, C, op.Params.intrafrag_hess)

                        # TODO: Use computed Hessian.
                        Hcart = get_hessian(oMolsys.geom, o_json, printResults=False)
                        E = get_energy(oMolsys.geom, o_json, nuc=False)
                        (E, gX), qcjson  = get_gradient(oMolsys.geom, o_json, wantNuc=False)
                        H = intcosMisc.convertHessianToInternals(Hcart, oMolsys.intcos, oMolsys.geom)
                        optimize_log.debug(printMatString(H, title="Transformed Hessian in internal coordinates."))

                        # Add the transition state as the first IRC point
                        x_0 = oMolsys.geom
                        q_0 = intcosMisc.qValues(oMolsys.intcos, x_0)
                        f_x = np.zeros(len(oMolsys.geom))
                        f_q = np.zeros(len(oMolsys.intcos))
                        #f_q = np.array( for debugging with C++ code
                        #    [0.000003625246638, -0.000060308327958, 0.000003625246638,
                        #    -0.000019735265755, -0.000019735265755, 0.000000009192292] )

                        IRCdata.history.add_irc_point(0, q_0, x_0, f_q, f_x, E)
                        IRCstepNumber += 1

                        # Lowest eigenvector of mass-weighted Hessian.
                        G = intcosMisc.Gmat(oMolsys.intcos, oMolsys.geom, oMolsys.masses)
                        G_root = symmMatRoot(G)
                        H_q_m = np.dot(np.dot(G_root, H), G_root.T)
                        vM = lowestEigenvectorSymmMat(H_q_m)
                        optimize_log.info(printArrayString(vM, title="Lowest evect of H_q_M"))

                        # Un mass-weight vector. Remember that we could have redundant coordinates.
                        G_root_inv = symmMatInv(G_root, redundant=True)
                        v = np.dot(G_root_inv, vM)

                        if op.Params.irc_direction == 'BACKWARD':
                            v *= -1
                        ## hardwired to match bofill update from C++ code done with TS gradient
                        #H[:] = [
                        #[  0.585155,  0.000074, -0.000016,  0.000062,  0.000062, -0.000619],
                        #[  0.000074,  0.459684,  0.000074, -0.000286, -0.000286,  0.002836],
                        #[ -0.000016,  0.000074,  0.585155,  0.000062,  0.000062, -0.000619],
                        #[  0.000062, -0.000286,  0.000062,  0.159759, -0.000241,  0.002388],
                        #[  0.000062, -0.000286,  0.000062, -0.000241,  0.159759,  0.002388],
                        #[ -0.000619,  0.002836, -0.000619,  0.002388,  0.002388, -0.020493]]

                    else: # Step along gradient.
                        optimize_log.info("\tBeginning search for next IRC point.\n")
                        optimize_log.info("\tStepping along gradient.\n")
                        v = IRCdata.history.f_q()

                    IRCfollowing.computePivotAndGuessPoints(oMolsys, v, op.Params.irc_step_size)

                energies = []  # should be moved into history TODO get rid of this!
                for stepNumber in range(op.Params.alg_geom_maxiter):
                    optimize_log.info("Beginning algorithm loop, step number %d" % stepNumber) 
                    totalStepsTaken += 1
                    # compute energy and gradient
                    xyz = oMolsys.geom.copy()
                    (E, gX, nuc), qcjson = get_gradient(xyz, o_json, printResults=False, wantNuc=True)
                    oMolsys.geom = xyz  # use setter function to save data in fragments
                    printGeomGrad(oMolsys.geom, gX)
                    energies.append(E)

                    if op.Params.test_B:
                        testB.testB(oMolsys.intcos, oMolsys.geom)
                    if op.Params.test_derivative_B:
                        testB.testDerivativeB(oMolsys.intcos, oMolsys.geom)

                    B = intcosMisc.Bmat(oMolsys.intcos, oMolsys.geom)
                    optimize_log.debug(printMatString(B, title="B matrix"))

                    f_q = intcosMisc.qForces(oMolsys.intcos, oMolsys.geom, gX)
                    # Check if forces indicate we are approaching minimum.
                    if op.Params.opt_type == "IRC" and IRCstepNumber > 2:
                        if ( IRCdata.history.testForIRCminimum(f_q) ):
                            optimize_log.info("A mininum has been reached on the IRC.  Stopping here.\n")
                            raise IRCendReached()
                    #f_q = np.array( [ 0.000019538372495, 0.000213081515583,  0.000019538372495, 0.001090978572604,
                    #0.001090978572604,  -0.003640029745080], float)
                    optimize_log.info(printArrayString(f_q, title="Internal forces in au"))

                    history.oHistory.append(oMolsys.geom, E, f_q, qcjson)  # Save initial step info.
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
                            # IDE complains about H not being declared. Should be fine
                            Dq = stepAlgorithms.Dq(oMolsys, E, f_q, H, stepType="BACKSTEP")
                            optimize_log.info("\tStructure for next step (au):\n")
                            oMolsys.showGeom()
                            continue
                        elif op.Params.dynamic_level == 0:  # not using dynamic level, so ignore.
                            optimize_log.info("\tNo more backsteps allowed."
                                              + "Dynamic level is off.\n")
                            pass
                        else:
                            raise AlgError("Bad step, and no more backsteps allowed.")

                    # Produce Hessian via guess, update, or transformation.
                    if op.Params.opt_type != "IRC":
                        if stepNumber == 0:
                            if op.Params.full_hess_every > -1: # compute hessian at least once. 
                                xyz = oMolsys.geom.copy()
                                Hcart = get_hessian( xyz, o_json, printResults=True)
                                H = intcosMisc.convertHessianToInternals(Hcart, oMolsys.intcos, xyz)
                            else:
                                C = addIntcos.connectivityFromDistances(oMolsys.geom, oMolsys.Z)
                                H = hessian.guess(oMolsys.intcos, oMolsys.geom, oMolsys.Z, C,
                                                  op.Params.intrafrag_hess)
                        else: # not IRC, not first step
                            if op.Params.full_hess_every < 1:
                                history.oHistory.hessianUpdate(H, oMolsys.intcos)
                            elif stepNumber % op.Params.full_hess_every == 0:
                                xyz = copy.deepcopy(oMolsys.geom)
                                Hcart = get_hessian( xyz, o_json, printResults=False)
                                H = intcosMisc.convertHessianToInternals(
                                    Hcart, oMolsys.intcos, xyz)
                            else:
                                history.oHistory.hessianUpdate(H, oMolsys.intcos)
                    else: # IRC
                        if stepNumber == 0:
                            if IRCstepNumber == 0:
                                pass  # Initial H chosen in pre-optimization.
                            #else:     # update with one preserved point from previous rxnpath pt..
                                #history.oHistory.hessianUpdate(H, oMolsys.intcos)
                        else: # IRC, not first step
                            if op.Params.full_hess_every < 1:
                                history.oHistory.hessianUpdate(H, oMolsys.intcos)
                            elif stepNumber % op.Params.full_hess_every == 0:
                                xyz = copy.deepcopy(oMolsys.geom)
                                Hcart = get_hessian( xyz, o_json, printResults=False)
                                H = intcosMisc.convertHessianToInternals(
                                    Hcart, oMolsys.intcos, xyz)
                            else:
                                history.oHistory.hessianUpdate(H, oMolsys.intcos)


                    if op.Params.print_lvl >= 4:
                        hessian.show(H, oMolsys.intcos)

                    intcosMisc.applyFixedForces(oMolsys, f_q, H, stepNumber)
                    intcosMisc.projectRedundanciesAndConstraints(oMolsys.intcos, oMolsys.geom,
                                                                 f_q, H)
                    intcosMisc.qShowValues(oMolsys.intcos, oMolsys.geom)

                    if op.Params.opt_type == 'IRC':
                        DqGuess = IRCdata.history.q_pivot() - IRCdata.history.q()
                        Dq = IRCfollowing.Dq_IRC(oMolsys, E, f_q, H, op.Params.irc_step_size, DqGuess)
                    else:  # Displaces and adds step to history.
                        Dq = stepAlgorithms.Dq(oMolsys, E, f_q, H, op.Params.step_type, o_json)

                    if op.Params.opt_type == "IRC":
                        converged = convCheck.convCheck(stepNumber, oMolsys, Dq, f_q, energies, IRCdata.history.q_pivot())
                        optimize_log.info("\tConvergence check returned %s." % converged)

                        if converged:

                            lineDistStep = IRCfollowing.calcLineDistStep(oMolsys)
                            arcDistStep  = IRCfollowing.calcArcDistStep(oMolsys)

                            IRCdata.history.add_irc_point(IRCstepNumber,
                                intcosMisc.qValues(oMolsys.intcos, oMolsys.geom),
                                oMolsys.geom,
                                intcosMisc.qForces(oMolsys.intcos, oMolsys.geom, gX),
                                np.multiply(-1, gX),
                                energies[-1], lineDistStep, arcDistStep)
                            IRCdata.history.progress_report()

                    else:  #not IRC.
                        converged = convCheck.convCheck(stepNumber, oMolsys, Dq, f_q, energies)
                        optimize_log.info("\tConvergence check returned %s" % converged)
                    
                    if converged:  # changed from elif when above if statement active
                        optimize_log.info("\tConverged in %d steps!" % (stepNumber + 1))
                        optimize_log.info("\tFinal energy is %20.13f" % E)
                        optimize_log.info("\tFinal structure (Angstroms): \n\n"
                                          + oMolsys.showGeom())
                        break # break out of stepNumber loop

                    optimize_log.info("\tStructure for next step (au):\n" + oMolsys.showGeom())

                    # Hard quit if too many total steps taken (inc. all IRC points and algorithms).
                    if (totalStepsTaken == op.Params.geom_maxiter):
                        optimize_log.error("\tTotal number of steps (%d) exceeds maximum allowed (%d).\n"
                                           % (totalStepsTaken, op.Params.geom_maxiter))
                        raise OptError("Maximum number of steps exceeded: {}.".format(op.Params.geom_maxiter))

                else:  # executes if stepNumber reaches alg_geom_maxiter
                    optimize_log.error("\tNumber of steps (%d) exceeds maximum for algorithm (%d).\n"
                                       % (stepNumber + 1, op.Params.alg_geom_maxiter))
                    raise AlgError("Maximum number of steps exceeded for algorithm")

                # For IRC, save and queue up for the optimization of the next point. 
                if op.Params.opt_type == 'IRC':
                    IRCstepNumber += 1
                    if IRCstepNumber == op.Params.irc_points:
                        optimize_log.info("\tThe requested (%d) IRC points have been obtained."
                                           % op.Params.irc_points)
                        raise IRCendReached()
                    else:
                        optimize_log.info("\tStarting search for next IRC point.")
                        optimize_log.info("\tClearing old constrained optimization history.")
                        history.oHistory.resetToMostRecent() # delete old steps
                        converged = False

            # Catch non-fatal algorithm errors and try modifying internals,
            # changing run-levels, optimization parameters, etc. and start over again.
            except AlgError as AF:
                optimize_log.error("\n\tCaught AlgError exception\n")
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
                elif op.Params.dynamic_level == op.Params.dynamic_level_max:
                    optimize_log.critical("\n\t Current algorithm/dynamic_level is %d.\n"
                                          % op.Params.dynamic_level)
                    optimize_log.critical("\n\t Alternative approaches are not available or"
                                          + "turned on.\n")
                    raise OptError("Maximum dynamic_level reached.")
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

        # print summary
        logging.info("\tOptimization Finished\n" + history.summaryString())
        output_dict = o_json.generate_json_output(history.oHistory[-1].geom, gX)
        json_original = o_json._get_original(oMolsys.geom)
        json_original["success"] = True

        if op.Params.opt_type == 'linesearch':
            (E, gX), qcjson = get_gradient(oMolsys.geom, o_json, wantNuc=False)

        if op.Params.trajectory:
            # history doesn't contain atomic numbers so pass them in
            output_dict['properties']['trajectory'] = history.oHistory.trajectory(oMolsys.Z)
        else:
            returnVal = history.oHistory[-1].E

        del H
        del history.oHistory[:]
        oMolsys.clear()
        del op.Params
        json_original.update(output_dict)
        return json_original

    except IRCendReached:
        optimize_log.info(IRCdata.history.final_geom_coords(oMolsys.intcos)) 
        optimize_log.info("Tabulating rxnpath results.")
        IRCdata.history.progress_report()
        output_dict = o_json.generate_json_output(IRCdata.history.x(-1),
         np.multiply(-1,IRCdata.history.f_x(-1)))
        json_original = o_json._get_original(IRCdata.history.x(-1))
        json_original.update(output_dict)
        rxnpath = IRCdata.history.rxnpathDict()
        optimize_log.debug(rxnpath)
        json_original['properties']['IRC'] = rxnpath
        json_original["success"] = True

        # delete some stuff
        del H
        del history.oHistory[:]
        oMolsys.clear()
        del op.Params
        return json_original

    except OptError as error:  # We are quitting for an optimization problem reason.
        optimize_log.critical("\tA critical optimization-specific error has occured.\n")
        optimize_log.critical("\tResetting all optimization options for potential queued jobs.\n")
        optimize_log.exception("Error Type:  " + str(type(error)))
        optimize_log.exception("Error caught:" + str(error))
        # Dump histories if possible
        try:
            logging.debug("\tDumping history: Warning last point not converged.\n" + history.summaryString())
            if op.Params.opt_type == 'IRC':
                logging.info("\tDumping IRC points completed")
                IRCdata.history.progress_report()
        except:
            pass

        output_dict = o_json.generate_json_output(history.oHistory[-1].geom, gX)
        json_original = o_json._get_original(oMolsys.geom)
        json_original.update(output_dict)  # may not be wise or feasable in all cases
        json_original["error"] = repr(error)
        json_original["success"] = False
        if op.Params.opt_type == 'IRC':
            rxnpath = IRCdata.history.rxnpathDict()
            optimize_log.debug(rxnpath)
            json_original['properties']['IRC'] = rxnpath

        del history.oHistory[:]
        oMolsys.clear()
        del op.Params
        del o_json

        return json_original

    except Exception as error:
        optimize_log.critical("\tA non-optimization-specific error has occured.\n")
        optimize_log.critical("\tResetting all optimization options for potential queued jobs.\n")
        optimize_log.exception("Error Type:  " + str(type(error)))
        optimize_log.exception("Error caught:" + str(error))

        json_original = o_json._get_original(oMolsys.geom)
        if history.oHistory:
            output_dict = o_json.generate_json_output(history.oHistory[-1].geom, gX)
            json_original.update(output_dict)
        json_original["error"] = repr(error)
        json_original["success"] = False

        del history.oHistory[:]
        oMolsys.clear()
        del op.Params
        del o_json

        return json_original

# TODO move these elsewhere
# TODO need to activate printResults for get_x methods
def get_gradient(new_geom, o_json, printResults=False, wantNuc=True, QM='psi4'):
    """Use JSON interface to have QM program perform gradient calculation
    Only Psi4 is currently implemented

    Parameters
    ----------
    new_geom : ndarray
        (nat, 3) current geometry of molecule
    o_json : object
        instance of optking's jsonSchema class
    printResults : bool, optional
        flag to print the gradient
    nuc : bool
        flag to return the nuclear repulsion energy as well
    QM : str
        NYI will have options for programs other than psi4 eventually

    Returns
    -------
    return_energy: float
        calculated energy
    return_result: ndarray
        (nat, 3) cartesian gradient
    return_nuc : float
        returned depending on function parameters
    """

    json_output = psi4methods.psi4_calculation(new_geom, o_json)
    return o_json.get_JSON_result(json_output, 'gradient', wantNuc), json_output


def get_hessian(new_geom, o_json, printResults=False, QM='psi4'):
    """Use JSON interface to have QM program perform hessian calculation
    Only Psi4 is currently implemented

    Parameters
    ----------
    new_geom : ndarray
        (nat, 3) current geometry of molecule
    o_json : object
        instance of optking's jsonSchema class
    printResults : Boolean, optional
        flag to print the gradient
    QM : str
        NYI will have options for programs other than psi4 eventually

    Returns
    -------
    return_result : ndarray
        (nat, nat) hessian in cartesians
    """
    json_output = psi4methods.psi4_calculation(new_geom, o_json, driver="hessian")
    return np.array(o_json.get_JSON_result(json_output, 'hessian'))


def get_energy(new_geom, o_json, printResults=False, nuc=True, QM='psi4'):
    """ Use JSON interface to have QM program perform energy calculation
    Only psi4 is current implemented

    Parameters
    ----------
    new_geom : ndarray
        (nat, 3) current geometry of molecule
    o_json : object
        instance of optking's jsonSchema class
    printResults : Boolean, optional
        flag to print the gradient
    nuc : Boolean
        flag to return the nuclear repulsion energy as well
    QM : str
        NYI will have options for programs other than psi4 eventually

    Returns
    -------
    return_energy: float
        calculated energy
    return_nuc : float
        returned depending on function parameters
    """

    json_output = psi4methods.psi4_calculation(new_geom, o_json, driver='energy')
    return o_json.get_JSON_result(json_output, 'energy', nuc)



import numpy as np
import copy
import logging
from pprint import PrettyPrinter

from psi4.driver import json_wrapper  # COMMENT FOR INDEP DOCS BUILD

from . import hessian
from . import stepAlgorithms
from . import caseInsensitiveDict
from . import optparams as op
from .exceptions import OptError, AlgError
from . import addIntcos
from . import history
from . import intcosMisc
from . import convCheck
from . import testB
from . import IRCFollowing
from . import psi4methods
from .qcdbjson import jsonSchema
from .printTools import (printGeomGrad,
                         printMatString,
                         printArrayString,
                         welcome)

pp = PrettyPrinter(indent=4)


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
    try:
        optimize_log = logging.getLogger(__name__)

        userOptions = caseInsensitiveDict.CaseInsensitiveDict(options_in)
        # Save copy of original user options. Commented out because it was never used
        # origOptions = copy.deepcopy(userOptions)

        # Create full list of parameters from user options plus defaults.
        optimize_log.info(welcome())  # print header
        optimize_log.debug("\n\tProcessing user input options...\n")
        op.Params = op.optParams(userOptions)
        optimize_log.debug(str(op.Params))

        # if op.Params.logging_lvl == 'DEBUG' or op.Params.print_lvl == 5:
        #    log_file.setLevel(logging.DEBUG)
        # elif op.Params.logging_lvl == 'INFO' or op.Params.print_lvl >= 3:

        converged = False

        # generates a json dictionary if optking is not being called directly by json.
        o_json = 0
        if json_in is None:
            QM_method, basis, keywords = psi4methods.collect_psi4_options(options_in)
            atom_list = oMolsys.get_atom_list()
            o_json = jsonSchema.make_qcschema(oMolsys.geom, atom_list, QM_method, basis, keywords)
        else:
            o_json = json_in

        # For IRC computations:
        ircNumber = 0
        dqPivot = None  # Dummy argument for non-IRC
        # TODO if i'm remembering correctly, we need a way to determine if we have converged a
        # TODO constrained optimization or have successfully found the global minimum

        atMinimum = False
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
                if op.Params.opt_type == 'IRC' and ircNumber == 0:
                    irc_step_list = []  # Holds data points for IRC steps
                    qZero = intcosMisc.qValues(oMolsys.intcos, oMolsys.geom)

                    optimize_log.debug("Initial internal coordinates\n" + printArrayString(qZero))

                    #Prepare for IRC Step
                    # Hcart = get_hessian(oMolsys.geom, o_json, printResults=False)
                    (E, gX), qcjson  = get_gradient(oMolsys.geom, o_json, nuc=False)
                    B = intcosMisc.Bmat(oMolsys.intcos, oMolsys.geom)

                    fq = intcosMisc.qForces(oMolsys.intcos, oMolsys.geom, gX)
                    H = hessian.guess(oMolsys.intcos, oMolsys.geom, oMolsys.Z, C,
                                      op.Params.intrafrag_hess)
                    # Hq = intcosMisc.convertHessianToInternals(Hcart, oMolsys.intcos, 
                    #        oMolsys.geom)
                    intcosMisc.projectRedundanciesAndConstraints(oMolsys.intcos, oMolsys.geom, fq, H)
                    HM = intcosMisc.mass_weight_hessian_internals(H, B, oMolsys.intcos, oMolsys.masses)

                    #need to save inital information, for step Alg to add append to.
                    history.oHistory.append(oMolsys.geom, E, fq, qcjson)
                
                    irc_dqs_geoms = IRCFollowing.take_half_step(oMolsys, H, fq, 
                                                                op.Params.irc_step_size, gX,
                                                                initial=True)

                    optimize_log.debug("irc_dqs_geoms: " + str(irc_dqs_geoms))
                    optimize_log.debug("irc_dqs_geoms: " + str(type(irc_dqs_geoms)))
                    irc_step_list.append(irc_dqs_geoms)
                    #irc_dqs_geoms contains dqPivot, qPivot, dqGuess, qGuess
                    #Do we actually want the B matrix mass weighed her?
                    # Need to take a look at what we actually want coming back here

                
                # At this point, we should have already updated the geometry to the guess point. 
                # Need to (at this point) update the hessian, get the new gradient


                # Loop over geometry steps.
                energies = []  # should be moved into history
                for stepNumber in range(op.Params.geom_maxiter):
                    # compute energy and gradient
                    xyz = oMolsys.geom.copy()
                    (E, gX, nuc), qcjson = get_gradient(xyz, o_json, printResults=False, nuc=True)
                    oMolsys.geom = xyz  # use setter function to save data in fragments
                    printGeomGrad(oMolsys.geom, gX)
                    energies.append(E)

                    if op.Params.test_B:
                        testB.testB(oMolsys.intcos, oMolsys.geom)
                    if op.Params.test_derivative_B:
                        testB.testDerivativeB(oMolsys.intcos, oMolsys.geom)

                    B = intcosMisc.Bmat(oMolsys.intcos, oMolsys.geom)
                    optimize_log.debug(printMatString(B, title="B matrix"))

                    fq = intcosMisc.qForces(oMolsys.intcos, oMolsys.geom, gX)
                    if op.Params.print_lvl > 1:
                        optimize_log.info(printArrayString(fq, title="Internal forces in au"))

                    history.oHistory.append(oMolsys.geom, E, fq, qcjson)  # Save initial step info.
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
                            Dq = stepAlgorithms.Dq(oMolsys, E, fq, H, stepType="BACKSTEP")
                            optimize_log.info("\tStructure for next step (au):\n")
                            oMolsys.showGeom()
                            continue
                        elif op.Params.dynamic_level == 0:  # not using dynamic level, so ignore.
                            optimize_log.info("\tNo more backsteps allowed."
                                              + "Dynamic level is off.\n")
                            pass
                        else:
                            raise AlgError(
                                "Bad step, and no more backsteps allowed.")

                    # Produce guess Hessian or update existing Hessian..
                    if stepNumber == 0 and op.Params.opt_type != "IRC":
                        if op.Params.full_hess_every > -1:
                            xyz = oMolsys.geom.copy()
                            Hcart = get_hessian(
                                xyz, o_json, printResults=True)  # don't let function move geometry
                            H = intcosMisc.convertHessianToInternals(Hcart, oMolsys.intcos, 
                                                                     xyz)
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
                            xyz = copy.deepcopy(oMolsys.geom)
                            Hcart = get_hessian( xyz, o_json, printResults=False)
                            # it's possible function moves geometry
                            H = intcosMisc.convertHessianToInternals(
                                Hcart, oMolsys.intcos, xyz)
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

                    if op.Params.opt_type == 'IRC':

                        #New geom and new gradient at point on hypersphere
                        #E, gX = get_gradient(oMolsys.geom, o_json, nuc=False)
                        #B = intcosMisc.Bmat(oMolsys.intcos, oMolsys.geom)
                        Dq = IRCFollowing.Dq(oMolsys, gX, E, H, B, op.Params.irc_step_size, irc_step_list[-1][2])
                    
                    else:  # Displaces and adds step to history.
                        Dq = stepAlgorithms.Dq(oMolsys, E, fq, H, op.Params.step_type, o_json)
                    # else statement paried with above IRC if was removed


                    if op.Params.opt_type == "IRC":

                        converged = convCheck.convCheck(stepNumber, oMolsys, Dq, fq, energies,
                                                        irc_step_list[-1][1])
                        if converged:
                            converged = False
                            # TODO add check for minimum
                            if atMinimum:
                                converged = True
                    else:    
                        converged = convCheck.convCheck(stepNumber, oMolsys, Dq, fq, energies)
                    
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
                    raise OptError("Maximum number of steps exceeded: {}.".format(op.Params.geom_maxiter))

                # This should be called at the end of each iteration of the while loop
                if op.Params.opt_type == 'IRC' and not atMinimum:
                    ircNumber += 1
                    ircStepList.append(history.IRC_step(qPivot, oMolsys.geom, ircNumber))

                    H = history.oHistory.hessianUpdate(H, oMolsys.intcos)
                    (E, gX), qcjson = get_gradient(xyz, o_json, printResults=False, nuc=False)

                    B = intcosMisc.Bmat(oMolsys.intcos, oMolsys.geom, oMolsys.masses)
                    irc_dqs_geoms = IRCFollowing.take_half_step(oMolsys, E, H, B, 
                                                                op.Params.irc_step_size, gX)
                    irc_step_list.append(irc_dqs_geoms)

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
        logging.info("\tOptimization Finished\n" + history.generate_file_output())
        output_dict = o_json.generate_json_output(history.oHistory[-1].geom, gX)
        json_original = o_json._get_original(oMolsys.geom)

        if op.Params.opt_type == 'linesearch':
            (gX), qcjson = get_gradient(oMolsys.geom, o_json, nuc=False)

        if op.Params.trajectory:
            # history doesn't contain atomic numbers so pass them in
            output_dict['properties']['trajectory'] = history.oHistory.trajectory(oMolsys.Z)
        else:
            returnVal = history.oHistory[-1].E

        # clean up
        del H
        for f in oMolsys._fragments:
            del f._intcos[:]
            del f
        del history.oHistory[:]
        del oMolsys
        del op.Params
        json_original.update(output_dict)
        return json_original

    except Exception as error:
        # TODO needs some improvements
        optimize_log.critical("\tA non-optimization error has occured\n")
        optimize_log.critical(("\tResetting all optimzation options to prevent queued" +
                               "optimizations from failing\n"))
        optimize_log.exception("Error Type:" + str(type(error)))
        optimize_log.exception("Error caught:" + str(error))

        output_dict = o_json.generate_json_output(history.oHistory[-1].geom, gX)
        json_original = o_json._get_original(oMolsys.geom)
        json_original.update(output_dict)
        json_original["error"] = repr(error)
        json_original["success"] = False

        del history.oHistory[:]
        del o_json
        # had been deleting oMolsys and Fragments here but IDE complained they
        # were not declared
        del op.Params

        return json_original

# TODO need to activate printResults for get_x methods
def get_gradient(new_geom, o_json, printResults=False, nuc=True, QM='psi4'):
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
    return o_json.get_JSON_result(json_output, 'gradient', nuc), json_output


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

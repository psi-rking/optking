import numpy as np
import copy
import logging
import json

from qcelemental.models import Molecule

from . import hessian
from . import stepAlgorithms
from . import optparams as op
from . import addIntcos
from . import history
from . import intcosMisc
from . import convCheck
from . import testB
from . import IRCfollowing
from . import IRCdata
from .exceptions import OptError, AlgError, IRCendReached
from .linearAlgebra import lowestEigenvectorSymmMat, symmMatRoot, symmMatInv
from .printTools import printGeomGrad, printMatString, printArrayString

from .molsys import Molsys


def optimize(oMolsys, computer):
    """ Driver for OptKing's optimization procedure. Suggested that users use optimize_psi4 or optimize_qcengine
        to perform a normal (full) optimization

    Parameters
    ----------
    oMolsys : cls
        optking molecular system
    opt_keys : dict
        options for QM program and optking
        format: {'OPTKING': {'program': 'psi4'}, 'QM': '{'method': 'HF', 'basis': 'STO-3G', keys: values}
    qc_result_input: dict
        optional. Provide python dictionary of the MolSSI QCInputSpecification model
        Do not need to provide 'QM' sub-dictionary above if provided. Model requirements may be seen at
        https://github.com/MolSSI/QCElemental/blob/master/qcelemental/models/procedures.py

    Returns
    -------
    float, float or dict
        energy and nuclear repulsion energy or MolSSI qc_schema_output as dict

    """
    print("optimize(oMolsys,computer)")
    print(oMolsys)
    logger = logging.getLogger(__name__)
    
    # Take care of some initial variable declarations
    stepNumber = 0 # number of steps taken. Partial. IRC alg uses two step counters
    H = 0 # hessian in internals

    try:  # Try to optimize one structure OR set of IRC points. OptError and all Exceptions caught below.
        

        # Prepare for multiple IRC computation
        if op.Params.opt_type == 'IRC':
            IRCstepNumber = 0
            IRCdata.history = IRCdata.IRCdata()
            IRCdata.history.set_atom_symbols(oMolsys.atom_symbols)
            # Why do we need to have IRCdata.history store its own copy?
            IRCdata.history.set_step_size_and_direction(op.Params.irc_step_size, op.Params.irc_direction)
            logger.info("\tIRC data object created\n")

        converged = False
        total_steps_taken = -1
        #oMolsys = make_internal_coords(oMolsys)
        if not oMolsys.intcos_present:
            make_internal_coords(oMolsys)

        # following loop may repeat over multiple algorithms OR over IRC points
        while not converged:
            try:
                # if optimization coordinates are absent, choose them. Could be erased after AlgError
                if not oMolsys.intcos_present:
                    make_internal_coords(oMolsys)
                    #oMolsys = make_internal_coords(oMolsys)

                logger.info("\tStarting optimization algorithm.\n")
                print('Starting optimization algorithm')
                logger.info(str(oMolsys))

                # Do special initial step-0 for each IRC point.
                # For IRC point, we form/get the Hessian now.

                if op.Params.opt_type == 'IRC':
                    if IRCstepNumber == 0:  # Step along lowest eigenvector of mass-weighted Hessian.
                        logger.info("\tBeginning IRC from the transition state.\n")
                        logger.info("\tStepping along lowest Hessian eigenvector.\n")

                        hess = computer.compute(oMolsys.geom, driver='hessian', return_full=False,
                                              print_result=True)
                        Hcart = np.asarray(hess).reshape(oMolsys.geom.size,oMolsys.geom.size)  # 3N x 3N

                        # TODO see if gradient comes back with the hessian
                        grad = computer.compute(oMolsys.geom, driver='gradient', return_full=False)
                        gX = np.asarray(grad)
                        E = computer.energies[-1]
                        H = intcosMisc.convertHessianToInternals(Hcart, oMolsys.intcos, oMolsys.geom)
                        logger.debug(printMatString(H, title="Transformed Hessian in internal coordinates."))

                        # Add the transition state as the first IRC point
                        x_0 = oMolsys.geom
                        q_0 = intcosMisc.qValues(oMolsys.intcos, x_0)

                        f_x = np.multiply(-1, gX)
                        B = intcosMisc.Bmat(oMolsys.intcos, x_0)
                        f_q = intcosMisc.qForces(q_0, x_0, gX, B)

                        IRCdata.history.add_irc_point(0, q_0, x_0, f_q, f_x, E)
                        IRCstepNumber += 1

                        # Lowest eigenvector of mass-weighted Hessian.
                        G = intcosMisc.Gmat(oMolsys.intcos, oMolsys.geom, oMolsys.masses)
                        G_root = symmMatRoot(G)
                        H_q_m = np.dot(np.dot(G_root, H), G_root.T)
                        vM = lowestEigenvectorSymmMat(H_q_m)
                        logger.info(printArrayString(vM, title="Lowest evect of H_q_M"))

                        # Un mass-weight vector.
                        G_root_inv = symmMatInv(G_root)
                        v = np.dot(G_root_inv, vM)

                        if op.Params.irc_direction == 'BACKWARD':
                            v *= -1
                    # end if IRCStepNumber == 0

                    else:  # Step along gradient.
                        logger.info("\tBeginning search for next IRC point.\n")
                        logger.info("\tStepping along gradient.\n")
                        v = IRCdata.history.f_q()

                    IRCfollowing.computePivotAndGuessPoints(oMolsys, v, op.Params.irc_step_size)
                #end if 'IRC'

                for stepNumber in range(op.Params.alg_geom_maxiter):
                    logger.info("\tBeginning algorithm loop, step number %d" % stepNumber)
                    total_steps_taken += 1
                    # compute energy and gradient
                    xyz = oMolsys.geom.copy()
                    grad = computer.compute(xyz, driver='gradient', return_full=False)
                    gX = np.asarray(grad)
                    E = computer.energies[-1]
                    oMolsys.geom = xyz  # use setter function to save data in fragments
                    printGeomGrad(oMolsys.geom, gX)

                    if op.Params.test_B:
                        testB.testB(oMolsys.intcos, oMolsys.geom)
                    if op.Params.test_derivative_B:
                        testB.testDerivativeB(oMolsys.intcos, oMolsys.geom)

                    B = intcosMisc.Bmat(oMolsys)
                    logger.debug(printMatString(B, title="B matrix"))

                    f_q = intcosMisc.qForces(oMolsys, gX)
                    # Check if forces indicate we are approaching minimum.
                    if op.Params.opt_type == "IRC" and IRCstepNumber > 2:
                        if IRCdata.history.testForIRCminimum(f_q):
                            logger.info("A minimum has been reached on the IRC.  Stopping here.\n")
                            raise IRCendReached()

                    logger.info(printArrayString(f_q, title="Internal forces in au"))

                    history.oHistory.append(oMolsys.geom, E, f_q)  # Save initial step info.
                    history.oHistory.nuclear_repulsion_energy = \
                        (computer.trajectory[-1]['properties']['nuclear_repulsion_energy'])

                    # Analyze previous step performance; adjust trust radius accordingly.
                    # Returns true on first step (no history)
                    lastStepOK = history.oHistory.currentStepReport()

                    # If step was bad, take backstep here or raise exception.
                    if lastStepOK:
                        history.oHistory.consecutiveBacksteps = 0
                    else:
                        # Don't go backwards until we've gone a few iterations.
                        if len(history.oHistory.steps) < 5:
                            logger.info(
                                    "\tNear start of optimization, so ignoring bad step.\n")
                        elif history.History.consecutiveBacksteps < op.Params.consecutiveBackstepsAllowed:
                            history.History.consecutiveBacksteps += 1
                            logger.info("\tCalling for consecutive backstep number %d.\n"
                                              % history.History.consecutiveBacksteps)
                            stepAlgorithms.take_step(oMolsys, E, f_q, H, stepType="BACKSTEP")
                            logger.info("\tStructure for next step (au):\n")
                            oMolsys.showGeom()
                            continue
                        elif op.Params.dynamic_level == 0:  # not using dynamic level, so ignore.
                            logger.info("\tNo more backsteps allowed."
                                              + "Dynamic level is off.\n")
                            pass
                        else:
                            raise AlgError("Bad step, and no more backsteps allowed.")

                    # Produce Hessian via guess, update, or transformation.
                    if op.Params.opt_type != "IRC":
                        if stepNumber == 0:
                            if op.Params.full_hess_every > -1:  # compute hessian at least once.
                                xyz = oMolsys.geom.copy()
                                Hcart = computer.compute(xyz, driver='hessian', return_full=False,
                                                       print_result=False)
                                Hcart = np.asarray(Hcart).reshape(oMolsys.geom.size, oMolsys.geom.size)
                                H = intcosMisc.convertHessianToInternals(Hcart, oMolsys.intcos, xyz)
                            else:
                                H = hessian.guess(oMolsys, op.Params.intrafrag_hess)
                        else:  # not IRC, not first step
                            if op.Params.full_hess_every < 1:
                                history.oHistory.hessianUpdate(H, oMolsys)
                            elif stepNumber % op.Params.full_hess_every == 0:
                                xyz = copy.deepcopy(oMolsys.geom)
                                Hcart = computer.compute(xyz, driver='hessian', return_full=False,
                                                       print_result=False)
                                Hcart = np.asarray(Hcart).reshape(oMolsys.geom.size, oMolsys.geom.size)
                                H = intcosMisc.convertHessianToInternals(Hcart, oMolsys.intcos, xyz)
                            else:
                                history.oHistory.hessianUpdate(H, oMolsys)
                    else:  # IRC
                        if stepNumber == 0:
                            if IRCstepNumber == 0:
                                pass  # Initial H chosen in pre-optimization.
                            # else:   # update with one preserved point from previous rxnpath pt..
                                # history.oHistory.hessianUpdate(H, oMolsys)
                        else:  # IRC, not first step
                            if op.Params.full_hess_every < 1:
                                history.oHistory.hessianUpdate(H, oMolsys)
                            elif stepNumber % op.Params.full_hess_every == 0:
                                xyz = copy.deepcopy(oMolsys.geom)
                                Hcart = computer.compute(xyz, driver='hessian', return_full=False,
                                                       print_result=False)
                                Hcart = np.asarray(Hcart).reshape(oMolsys.geom.size, oMolsys.geom.size)
                                H = intcosMisc.convertHessianToInternals(Hcart, oMolsys.intcos, xyz)
                            else:
                                history.oHistory.hessianUpdate(H, oMolsys)

                    if op.Params.print_lvl >= 4:
                        hessian.show(H, oMolsys.intcos)

                    intcosMisc.applyFixedForces(oMolsys, f_q, H, stepNumber)
                    intcosMisc.projectRedundanciesAndConstraints(oMolsys, f_q, H)
                    oMolsys.qShow()

                    if op.Params.opt_type == 'IRC':
                        DqGuess = IRCdata.history.q_pivot() - IRCdata.history.q()
                        Dq = IRCfollowing.Dq_IRC(oMolsys, E, f_q, H, op.Params.irc_step_size, DqGuess)
                    else:  # Displaces and adds step to history.
                        Dq = stepAlgorithms.take_step(oMolsys, E, f_q, H, op.Params.step_type, computer)

                    if op.Params.opt_type == "IRC":
                        converged = convCheck.convCheck(stepNumber, oMolsys, Dq, f_q, computer.energies,
                                                        IRCdata.history.q_pivot())
                        logger.info("\tConvergence check returned %s." % converged)

                        if converged:
                            q_irc_point = intcosMisc.qValues(oMolsys.intcos, oMolsys.geom)
                            forces_irc_point = intcosMisc.qForces(oMolsys.intcos, oMolsys.geom, gX)
                            lineDistStep = IRCfollowing.calcLineDistStep(oMolsys)
                            arcDistStep = IRCfollowing.calcArcDistStep(oMolsys)

                            IRCdata.history.add_irc_point(IRCstepNumber, q_irc_point, oMolsys.geom, forces_irc_point,
                                                          np.multiply(-1, gX), computer.energies[-1],
                                                          lineDistStep, arcDistStep)
                            IRCdata.history.progress_report()

                    else:  # not IRC.
                        converged = convCheck.convCheck(stepNumber, oMolsys, Dq, f_q, computer.energies)
                        logger.info("\tConvergence check returned %s" % converged)
                    
                    if converged:  # changed from elif when above if statement active
                        logger.info("\tConverged in %d steps!" % (stepNumber + 1))
                        logger.info("\tFinal energy is %20.13f" % E)
                        logger.info("\tFinal structure (Angstroms): \n\n"
                                    + oMolsys.showGeom())
                        break  # break out of stepNumber loop

                    logger.info("\tStructure for next step (au):\n" + oMolsys.showGeom())

                    # Hard quit if too many total steps taken (inc. all IRC points and algorithms).
                    if total_steps_taken == op.Params.geom_maxiter:
                        logger.error("\tTotal number of steps (%d) exceeds maximum allowed (%d).\n"
                                           % (total_steps_taken, op.Params.geom_maxiter))
                        raise OptError("Maximum number of steps exceeded: {}.".format(op.Params.geom_maxiter),
                                       'OptError')

                else:  # Associated with above for loop, executes if break is not reached
                    logger.error("\tNumber of steps (%d) exceeds maximum for algorithm (%d).\n"
                                       % (stepNumber + 1, op.Params.alg_geom_maxiter))
                    raise AlgError("Maximum number of steps exceeded for algorithm")

                # For IRC, save and queue up for the optimization of the next point. 
                if op.Params.opt_type == 'IRC':
                    IRCstepNumber += 1
                    if IRCstepNumber == op.Params.irc_points:
                        logger.info(f"\tThe requested {op.Params.irc_points} IRC points have been obtained.")
                        raise IRCendReached()
                    else:
                        logger.info("\tStarting search for next IRC point.")
                        logger.info("\tClearing old constrained optimization history.")
                        history.oHistory.resetToMostRecent()  # delete old steps
                        converged = False

            # Catch non-fatal algorithm errors and try modifying internals,
            # changing run-levels, optimization parameters, etc. and start over again.
            except AlgError as AF:
                logger.error("\n\tCaught AlgError exception\n")
                eraseIntcos = False

                if AF.linearBends:
                    # New linear bends detected; Add them, and continue at current level.
                    # from . import bend # import not currently being used according to IDE
                    for l in AF.linearBends:
                        if l.bendType == "LINEAR":  # no need to repeat this code for "COMPLEMENT"
                            F = addIntcos.checkFragment(l.atoms, oMolsys)
                            intcosMisc.removeOldNowLinearBend(l.atoms, oMolsys._fragments[F].intcos)
                    oMolsys.addIntcosFromConnectivity()
                    eraseHistory = True
                elif op.Params.dynamic_level == op.Params.dynamic_level_max:
                    logger.critical("\n\t Current algorithm/dynamic_level is %d.\n"
                                          % op.Params.dynamic_level)
                    logger.critical("\n\t Alternative approaches are not available or"
                                          + "turned on.\n")
                    raise OptError("Maximum dynamic_level reached.")
                else:
                    op.Params.dynamic_level += 1
                    logger.warning("\n\t Increasing dynamic_level algorithm to %d.\n"
                                         % op.Params.dynamic_level)
                    logger.warning("\n\t Erasing old history, hessian, intcos.\n")
                    eraseIntcos = True
                    eraseHistory = True
                    op.Params.updateDynamicLevelParameters(op.Params.dynamic_level)

                if eraseIntcos:
                    logger.warning("\n\t Erasing coordinates.\n")
                    for f in oMolsys._fragments:
                        del f._intcos[:]

                if eraseHistory:
                    logger.warning("\n\t Erasing history.\n")
                    stepNumber = 0
                    del H
                    del history.oHistory[:]  # delete steps in history
                    history.oHistory.stepsSinceLastHessian = 0
                    history.oHistory.consecutiveBacksteps = 0

        # print summary
        logger.info("\tOptimization Finished\n" + history.summaryString())

        if op.Params.opt_type == 'linesearch':
            logger.info("\tObtaining gradient at the final geometry for line-search optimization\n")
            # Calculate gradient to show user
            gX = computer.compute(oMolsys.geom, driver='gradient', return_full=False)
            del gX
        qc_output = prepare_opt_output(oMolsys, computer, error=None)

        del H
        del history.oHistory[:]
        oMolsys.clear()
        del op.Params
        return qc_output
    
    # Expect to hit this error. not an issue
    except IRCendReached:
        logger.info(IRCdata.history.final_geom_coords(oMolsys.intcos))
        logger.info("Tabulating rxnpath results.")
        IRCdata.history.progress_report()
        np.multiply(-1, IRCdata.history.f_x(-1))
        rxnpath = IRCdata.history.rxnpathDict()
        logger.debug(rxnpath)

        qc_output = prepare_opt_output(oMolsys, computer, rxnpath=rxnpath, error=None)

        # delete some stuff
        del H
        del history.oHistory[:]
        oMolsys.clear()
        del op.Params
        return qc_output
    
    # Fatal error. Cannot proceed.
    except OptError as error:
        logger.critical("\tA critical optimization-specific error has occured.\n")
        logger.critical("\tResetting all optimization options for potential queued jobs.\n")
        logger.exception("Error Type:  " + str(type(error)))
        logger.exception("Error caught:" + str(error))
        # Dump histories if possible
        try:
            logging.debug("\tDumping history: Warning last point not converged.\n" + history.summaryString())
            if op.Params.opt_type == 'IRC':
                logging.info("\tDumping IRC points completed")
                IRCdata.history.progress_report()
            del history.oHistory[:]
        except NameError:
            pass

        rxnpath = None
        if op.Params.opt_type == 'IRC':
            rxnpath = IRCdata.history.rxnpathDict()
            logger.debug(rxnpath)

        qc_output = prepare_opt_output(oMolsys, computer, rxnpath=rxnpath, error=error)

        del history.oHistory[:]
        oMolsys.clear()
        del op.Params
        del computer

        return qc_output

    except Exception as error:
        logger.critical("\tA non-optimization-specific error has occurred.\n")
        logger.critical("\tResetting all optimization options for potential queued jobs.\n")
        logger.exception("Error Type:  " + str(type(error)))
        logger.exception("Error caught:" + str(error))

        rxnpath = None
        if len(history.oHistory.steps) >= 1:
            rxnpath = None
            if op.Params.opt_type == 'IRC':
                rxnpath = IRCdata.history.rxnpathDict()
                logger.debug(rxnpath)

        qc_output = prepare_opt_output(oMolsys, computer, rxnpath=rxnpath, error=error)

        del history.oHistory[:]
        oMolsys.clear()
        del op.Params
        del computer

        return qc_output

def initialize_options(opt_keys):
    try:
        userOptions = caseInsensitiveDict.CaseInsensitiveDict(opt_keys['OPTKING'])
        # Save copy of original user options. Commented out until it is used
        # origOptions = copy.deepcopy(userOptions)

        # Create full list of parameters from user options plus defaults.
        logger = logging.getLogger(__name__)
        logger.info(welcome())
        logger.debug("\n\tProcessing user input options...\n")
        op.Params = op.OptParams(userOptions) 
        # TODO we should make this just be a normal object and we should return it to the optimize method
        logger.debug(str(op.Params))
    except OptError as e:
        raise e

def make_internal_coords(oMolsys):
    """
    Add optimization coordinates to molecule system.
    May be called if coordinates have not been added yet, or have been removed due to an
    algorithm error (bend going linear, or energy increasing, etc.).

    Parameters
    ----------
    oMolsys: Molsys
        current molecular system.

    Returns
    -------
    oMolsys: Molsys
        The molecular system updated with internal coordinates.
        TODO: why not just add them to existing one?
    """
    optimize_log = logging.getLogger(__name__)
    optimize_log.debug("\t Adding internal coordinates to molecular system")

    # Use covalent radii to determine bond connectivity.
    connectivity = addIntcos.connectivityFromDistances(oMolsys.geom, oMolsys.Z)
    optimize_log.debug("Connectivity Matrix\n" + printMatString(connectivity))

    if op.Params.frag_mode == 'SINGLE':
        # Make a single, supermolecule.
        oMolsys.consolidateFragments()          # collapse into one frag (if > 1)
        oMolsys.splitFragmentsByConnectivity()  # separate by connectivity
        # increase connectivity until all atoms are connected
        oMolsys.augmentConnectivityToSingleFragment(connectivity)
        oMolsys.consolidateFragments()          # collapse into one frag
    elif op.Params.frag_mode == 'MULTI':
        # if provided multiple frags, then we use these.
        # if not, then split them (if not connected).
        if oMolsys.Nfragments == 1:
            oMolsys.splitFragmentsByConnectivity()
        if oMolsys.Nfragments > 1:
            print('Nfragments > 1')
            addIntcos.addDimerFragIntcos(oMolsys)
        if oMolsys.Nfragments > 1:
            oMolsys.purgeInterfragmentConnectivity(connectivity)

    if op.Params.opt_coordinates in ['REDUNDANT', 'BOTH']:
        oMolsys.addIntcosFromConnectivity(connectivity)

    if op.Params.opt_coordinates in ['CARTESIAN', 'BOTH']:
        oMolsys.addCartesianIntcos()
    addIntcos.addFrozenAndFixedIntcos(oMolsys)  # make sure these are in the set
    print('end of make_internal_coords:')
    print(oMolsys)
    return


def prepare_opt_output(oMolsys, computer, rxnpath=False, error=None):
    logger = logging.getLogger(__name__)
    logger.debug("Preparing OptimizationResult")
    # Get molecule from most recent step. Add provenance and fill in non-required fills. Turn back to dict
    final_molecule = oMolsys.molsys_to_qc_molecule()

    qc_output = {"schema_name": 'qcschema_optimization_output', "trajectory": computer.trajectory,
                 "energies": computer.energies, "final_molecule": final_molecule,
                 "extras": {}, "success": True,}

    if error:
        qc_output.update({"success": False, "error": {"error_type": error.err_type, "error_message": error.mesg}})
    
    if rxnpath:
        qc_output['extras'].update({"irc_rxn_path": rxnpath})

    return qc_output


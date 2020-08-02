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
from .linearAlgebra import lowest_eigenvector_symm_mat, symm_mat_root, symm_mat_inv
from .printTools import print_geom_grad, print_mat_string, print_array_string

from .molsys import Molsys


def optimize(oMolsys, computer):
    """ Driver for OptKing's optimization procedure. Suggested that users use optimize_psi4 or optimize_qcengine
        to perform a normal (full) optimization

    Parameters
    ----------
    oMolsys : cls
        optking molecular system
    computer : compute_wrappers.ComputeWrapper

    Returns
    -------
    float, float or dict
        energy and nuclear repulsion energy or MolSSI qc_schema_output as dict

    """
    logger = logging.getLogger(__name__)
    logger.info("Running optimize(oMolsys,computer)")
    
    # Take care of some initial variable declarations
    step_number = 0 # number of steps taken. Partial. IRC alg uses two step counters
    irc_step_number = None
    total_steps_taken = 0
    H = 0 # hessian in internals

    try:  # Try to optimize one structure OR set of IRC points. OptError and all Exceptions caught below.

        # Prepare for multiple IRC computation
        if op.Params.opt_type == 'IRC':
            irc_step_number = 0
            IRCdata.history = IRCdata.IRCdata()
            IRCdata.history.set_atom_symbols(oMolsys.atom_symbols)
            # Why do we need to have IRCdata.history store its own copy?
            IRCdata.history.set_step_size_and_direction(op.Params.irc_step_size, op.Params.irc_direction)
            logger.info("\tIRC data object created\n")

        converged = False
        # oMolsys = make_internal_coords(oMolsys)
        if not oMolsys.intcos_present:
            make_internal_coords(oMolsys)
            logger.debug("Molecular systems after make_internal_coords:")
            logger.debug(str(oMolsys))

        # following loop may repeat over multiple algorithms OR over IRC points
        while not converged:
            try:
                # if optimization coordinates are absent, choose them. Could be erased after AlgError
                if not oMolsys.intcos_present:
                    make_internal_coords(oMolsys)
                    logger.debug("Molecular systems after make_internal_coords:")
                    logger.debug(str(oMolsys))

                logger.info("\tStarting optimization algorithm.\n")
                logger.info(str(oMolsys))

                # Do special initial step-0 for each IRC point.
                # For IRC point, we form/get the Hessian now.

                if op.Params.opt_type == 'IRC':
                    if irc_step_number == 0:
                        # Step along lowest eigenvector of mass-weighted Hessian.
                        logger.info("\tBeginning IRC from the transition state.\n")
                        logger.info("\tStepping along lowest Hessian eigenvector.\n")

                        H, gX = get_pes_info(H, computer, oMolsys, step_number, irc_step_number)
                        logger.debug(print_mat_string(H, title="Transformed Hessian in internals."))

                        # Add the transition state as the first IRC point
                        # x_0 = oMolsys.geom
                        # q_0 = intcosMisc.q_values(oMolsys.intcos, x_0)
                        # f_x = np.multiply(-1, gX)
                        # B = intcosMisc.compute_b_mat(oMolsys.intcos, x_0)
                        # f_q = intcosMisc.q_forces(q_0, x_0, gX, B)

                        q_0 = oMolsys.q_array()
                        x_0 = oMolsys.geom
                        f_q = oMolsys.q_forces(gX)
                        f_x = np.multiply(-1, gX)
                        E = computer.energies[-1]

                        IRCdata.history.add_irc_point(0, q_0, x_0, f_q, f_x, E)
                        irc_step_number += 1

                        # Lowest eigenvector of mass-weighted Hessian.
                        G = oMolsys.compute_g_mat(oMolsys.masses)
                        G_root = symm_mat_root(G)
                        H_q_m = np.dot(np.dot(G_root, H), G_root.T)
                        vM = lowest_eigenvector_symm_mat(H_q_m)
                        logger.info(print_array_string(vM, title="Lowest evect of H_q_M"))

                        # Un mass-weight vector.
                        G_root_inv = symm_mat_inv(G_root, redundant=True)
                        v = np.dot(G_root_inv, vM)

                        if op.Params.irc_direction == 'BACKWARD':
                            v *= -1
                    # end if IRCStepNumber == 0

                    else:  # Step along gradient.
                        logger.info("\tBeginning search for next IRC point.\n")
                        logger.info("\tStepping along gradient.\n")
                        v = IRCdata.history.f_q()
                        irc_step_number += 1

                    IRCfollowing.compute_pivot_and_guess_points(oMolsys, v, op.Params.irc_step_size)
                # end if 'IRC'

                for step_number in range(op.Params.alg_geom_maxiter):
                    logger.info("\tBeginning algorithm loop, step number %d" % step_number)
                    total_steps_taken += 1
                    # compute energy and gradient
                    # xyz = oMolsys.geom.copy() unused warning in IDE

                    H, gX = get_pes_info(H, computer, oMolsys, step_number, irc_step_number)
                    E = computer.energies[-1]

                    # oMolsys.geom = xyz  # use setter function to save data in fragments
                    print_geom_grad(oMolsys.geom, gX)

                    if op.Params.print_lvl >= 4:
                        hessian.show(H, oMolsys)

                    f_q = oMolsys.q_forces(gX)
                    intcosMisc.apply_fixed_forces(oMolsys, f_q, H, step_number)
                    intcosMisc.project_redundancies_and_constraints(oMolsys, f_q, H)
                    oMolsys.q_show()

                    if op.Params.test_B:
                        testB.test_b(oMolsys)
                    if op.Params.test_derivative_B:
                        testB.test_derivative_b(oMolsys)

                    # B = intcosMisc.compute_b_mat(oMolsys)
                    # logger.debug(print_mat_string(B, title="B matrix"))

                    # Check if forces indicate we are approaching minimum.
                    if op.Params.opt_type == "IRC" and irc_step_number > 2:
                        if IRCdata.history.test_for_irc_minimum(f_q):
                            logger.info("A minimum has been reached on the IRC.  Stopping here.\n")
                            raise IRCendReached()

                    logger.info(print_array_string(f_q, title="Internal forces in au"))

                    history.oHistory.append(oMolsys.geom, E, f_q)  # Save initial step info.
                    history.oHistory.nuclear_repulsion_energy = \
                        (computer.trajectory[-1]['properties']['nuclear_repulsion_energy'])

                    # Analyze previous step performance; adjust trust radius accordingly.
                    # Returns true on first step (no history)
                    lastStepOK = history.oHistory.current_step_report()

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
                            oMolsys.show_geom()
                            continue
                        elif op.Params.dynamic_level == 0:  # not using dynamic level, so ignore.
                            logger.info("\tNo more backsteps allowed."
                                              + "Dynamic level is off.\n")
                            pass
                        else:
                            raise AlgError("Bad step, and no more backsteps allowed.")

                    if op.Params.opt_type == 'IRC':
                        DqGuess = IRCdata.history.q_pivot() - IRCdata.history.q()
                        Dq = IRCfollowing.dq_irc(oMolsys, E, f_q, H, op.Params.irc_step_size, DqGuess)
                    else:  # Displaces and adds step to history.
                        Dq = stepAlgorithms.take_step(oMolsys, E, f_q, H, op.Params.step_type, computer)

                    if op.Params.opt_type == "IRC":
                        converged = convCheck.conv_check(step_number, oMolsys, Dq, f_q, computer.energies,
                                                         IRCdata.history.q_pivot())
                        logger.info("\tConvergence check returned %s." % converged)

                        if converged:
                            q_irc_point = oMolsys.q_array()
                            forces_irc_point = oMolsys.q_forces(gX)
                            lineDistStep = IRCfollowing.calc_line_dist_step(oMolsys)
                            arcDistStep = IRCfollowing.calc_arc_dist_step(oMolsys)

                            IRCdata.history.add_irc_point(irc_step_number, q_irc_point, oMolsys.geom, forces_irc_point,
                                                          np.multiply(-1, gX), computer.energies[-1],
                                                          lineDistStep, arcDistStep)
                            IRCdata.history.progress_report()

                    else:  # not IRC.
                        converged = convCheck.conv_check(step_number, oMolsys, Dq, f_q, computer.energies)
                        logger.info("\tConvergence check returned %s" % converged)
                    
                    if converged:  # changed from elif when above if statement active
                        logger.info("\tConverged in %d steps!" % (step_number + 1))
                        logger.info("\tFinal energy is %20.13f" % E)
                        logger.info("\tFinal structure (Angstroms): \n\n"
                                    + oMolsys.show_geom())
                        break  # break out of step_number loop

                    logger.info("\tStructure for next step (au):\n" + oMolsys.show_geom())

                    # Hard quit if too many total steps taken (inc. all IRC points and algorithms).

                    if total_steps_taken == op.Params.geom_maxiter:
                        logger.error("\tTotal number of steps (%d) exceeds maximum allowed (%d).\n"
                                           % (total_steps_taken, op.Params.geom_maxiter))
                        raise OptError("Maximum number of steps exceeded: {}.".format(op.Params.geom_maxiter),
                                       'OptError')

                else:  # Associated with above for loop, executes if break is not reached
                    logger.error("\tNumber of steps (%d) exceeds maximum for algorithm (%d).\n"
                                       % (step_number + 1, op.Params.alg_geom_maxiter))
                    raise AlgError("Maximum number of steps exceeded for algorithm")

                # For IRC, save and queue up for the optimization of the next point. 
                if op.Params.opt_type == 'IRC':
                    if irc_step_number == op.Params.irc_points:
                        logger.info(f"\tThe requested {op.Params.irc_points} IRC points have been obtained.")
                        raise IRCendReached()
                    else:
                        logger.info("\tStarting search for next IRC point.")
                        logger.info("\tClearing old constrained optimization history.")
                        history.oHistory.reset_to_most_recent()  # delete old steps
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
                        if l.bend_type == "LINEAR":  # no need to repeat this code for "COMPLEMENT"
                            iF = addIntcos.check_fragment(l.atoms, oMolsys)
                            F = oMolsys.fragments[iF]
                            intcosMisc.remove_old_now_linear_bend(l.atoms, F.intcos)
                            F.add_intcos_from_connectivity()
                    eraseHistory = True
                elif op.Params.dynamic_level == op.Params.dynamic_level_max:
                    logger.critical("\n\t Current algorithm/dynamic_level is %d.\n"
                                          % op.Params.dynamic_level)
                    logger.critical("\n\t Alternative approaches are not available or turned on.\n")
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
                    for f in oMolsys.fragments:
                        del f.intcos[:]

                if eraseHistory:
                    logger.warning("\n\t Erasing history.\n")
                    step_number = 0
                    del H
                    H = 0
                    del history.oHistory[:]  # delete steps in history
                    history.oHistory.stepsSinceLastHessian = 0
                    history.oHistory.consecutiveBacksteps = 0

        # print summary
        logger.info("\tOptimization Finished\n" + history.summary_string())

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
        rxnpath = IRCdata.history.rxnpath_dict()
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
            logging.debug("\tDumping history: Warning last point not converged.\n" + history.summary_string())
            if op.Params.opt_type == 'IRC':
                logging.info("\tDumping IRC points completed")
                IRCdata.history.progress_report()
            del history.oHistory[:]
        except NameError:
            pass

        rxnpath = None
        if op.Params.opt_type == 'IRC':
            rxnpath = IRCdata.history.rxnpath_dict()
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
                rxnpath = IRCdata.history.rxnpath_dict()
                logger.debug(rxnpath)

        qc_output = prepare_opt_output(oMolsys, computer, rxnpath=rxnpath, error=error)

        del history.oHistory[:]
        oMolsys.clear()
        del op.Params
        del computer

        return qc_output


def get_pes_info(H, computer, oMolsys, step_number, irc_step_number):
    """ Calculate, update, or guess hessian as appropriate. Calculate gradient, pulling
    gradient from hessian output if possible.
    Parameters
    ----------
    H: np.ndarray
        current Hessian
    computer: compute_wrappers.ComputeWrapper
    oMolsys : molsys.Molsys
    step_number: int
    irc_step_number: int
    Returns
    -------
    np.ndarray,
    """

    logger = logging.getLogger(__name__)

    if step_number == 0:
        if op.OptParams.opt_type != "IRC":
            if op.Params.full_hess_every > -1:  # compute hessian at least once.
                H, g_X = get_hess_grad(computer, oMolsys)
            else:
                logger.debug(f"Guessing Hessian with {str(op.Params.intrafrag_hess)}")
                H = hessian.guess(oMolsys, guessType=op.Params.intrafrag_hess)
                grad = computer.compute(oMolsys.geom, driver='gradient', return_full=False)
                g_X = np.asarray(grad)
        else:  # IRC
            if irc_step_number == 0:
                # OLD COMMENT: Initial H chosen in pre-optimization.
                """hessian was calculated explicitly in IRC section of optimize at the time of this
                comment. Moving here"""
                # TODO read in hessian so only 1 needs to be calculated for IRC forward/backward
                H, g_X = get_hess_grad(computer, oMolsys)
            else:
                logger.critical(f"""It should be impossible to hit this. Ever""")
                raise OptError("irc_step_number is {irc_step_number} but step_number is \
                                {step_number}. Values not allowed.")
    else:
        if op.Params.full_hess_every < 1:
            logger.debug(f"Updating Hessian with {str(op.Params.hess_update)}")
            history.oHistory.hessian_update(H, oMolsys)
            grad = computer.compute(oMolsys.geom, driver='gradient', return_full=False)
            g_X = np.asarray(grad)
        elif step_number % op.Params.full_hess_every == 0:
            H, g_X = get_hess_grad(computer, oMolsys)
        else:
            logger.debug(f"Updating Hessian with {str(op.Params.hess_update)}")
            history.oHistory.hessian_update(H, oMolsys)
            grad = computer.compute(oMolsys.geom, driver='gradient', return_full=False)
            g_X = np.asarray(grad)

    logger.debug(print_mat_string(H, title="Hessian matrix"))
    return H, g_X


def get_hess_grad(computer, oMolsys):
    """ Compute hessian and get gradient from output if possible
        Perform separate gradient calculation if needed
    Parameters
    ----------
    computer: compute_wrappers.ComputeWrapper
    oMolsys: molsys.Molsys
    Returns
    -------
    tuple(np.ndarray, np.ndarray)
    Notes
    -----
    Hessian is in internals gradient is in cartesian
    """
    # Not sure why we need a copy here
    logger = logging.getLogger(__name__)
    logger.debug("Computing an analytical hessian")
    xyz = oMolsys.geom.copy()
    # Always return_true so we don't have to compute the gradient as well
    ret = computer.compute(xyz, driver='hessian', return_full=True, print_result=False)
    h_cart = np.asarray(ret['return_result']).reshape(oMolsys.geom.size, oMolsys.geom.size)
    try:
        logger.debug("Looking for gradient in hessian output")
        g_cart = ret['extras']['qcvars']['CURRENT GRADIENT']
    except KeyError:
        logger.error("Could not find the gradient in qcschema")
        grad = computer.compute(oMolsys.geom, driver='gradient', return_full=False)
        g_cart = np.asarray(grad)
    # Likely not at stationary point. Include forces
    # ADDENDUM currently neglects forces term for all points - including non-stationary
    H = intcosMisc.hessian_to_internals(h_cart, oMolsys)

    return H, g_cart


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
    """
    optimize_log = logging.getLogger(__name__)
    optimize_log.debug("\t Adding internal coordinates to molecular system")

    # Use covalent radii to determine bond connectivity.
    connectivity = addIntcos.connectivity_from_distances(oMolsys.geom, oMolsys.Z)
    optimize_log.debug("Connectivity Matrix\n" + print_mat_string(connectivity))

    if op.Params.frag_mode == 'SINGLE':
        # Make a single, supermolecule.
        oMolsys.consolidate_fragments()          # collapse into one frag (if > 1)
        oMolsys.split_fragments_by_connectivity()  # separate by connectivity
        # increase connectivity until all atoms are connected
        oMolsys.augment_connectivity_to_single_fragment(connectivity)
        oMolsys.consolidate_fragments()         # collapse into one frag

        if op.Params.opt_coordinates in ['REDUNDANT', 'BOTH']:
            oMolsys.fragments[0].add_intcos_from_connectivity(connectivity)

        if op.Params.opt_coordinates in ['CARTESIAN', 'BOTH']:
            oMolsys.fragments[0].add_cartesian_intcos()

    elif op.Params.frag_mode == 'MULTI':
        # if provided multiple frags, then we use these.
        # if not, then split them (if not connected).
        if oMolsys.nfragments == 1:
            oMolsys.split_fragments_by_connectivity()

        if oMolsys.nfragments > 1:
            addIntcos.add_dimer_frag_intcos(oMolsys)
            # remove connectivity so that we don't add redundant coordinates
            # between fragments
            oMolsys.purge_interfragment_connectivity(connectivity)

        if op.Params.opt_coordinates in ['REDUNDANT', 'BOTH']:
            for iF, F in enumerate(oMolsys.fragments):
                C = np.ndarray((F.natom, F.natom))
                C[:] = connectivity[oMolsys.frag_atom_slice(iF), oMolsys.frag_atom_slice(iF)]
                F.add_intcos_from_connectivity(C)

        if op.Params.opt_coordinates in ['CARTESIAN', 'BOTH']:
            for F in oMolsys.fragments:
                F.add_cartesian_intcos()

    addIntcos.add_frozen_and_fixed_intcos(oMolsys)  # make sure these are in the set
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
        qc_output.update({"success": False, "error": {"error_type": error.err_type,
                                                      "error_message": error.mesg}})
    
    if rxnpath:
        qc_output['extras']["irc_rxn_path"] = rxnpath

    return qc_output


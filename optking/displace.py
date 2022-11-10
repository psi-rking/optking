import logging

import numpy as np

from . import intcosMisc
from . import optparams as op
from . addIntcos import linear_bend_check
from .exceptions import AlgError, OptError
from .linearAlgebra import abs_max, rms, symm_mat_inv
from .printTools import print_mat_string, print_array_string
from . import log_name

logger = logging.getLogger(f"{log_name}{__name__}")
# Functions in this file displace.py
#
#  displace_molsys: Displace each fragment.  Displace dimer coordinates.
#  displace_frag  : Displace a fragment by dq.  Double check frozen coordinates
#                 are satisfied.  Reduce stepsize as needed if
#                 ensure_convergence is true.  Also double check to ensure
#                 ranged coordinates are now outside prescribed range.
#  back_transformation:  Call dq_to_dx iteratively to try to converge to desired
#                  Dq as much as possible.
#  dq_to_dx       : Given Delta(q), compute and invert B, take Delta(x) step.


# Displace molecular system
def displace_molsys(molsys, dq_in, fq=None, ensure_convergence=False, return_str=False):
    """Manage internal coordinate step for a molecular system
    Parameters
    ----------
    oMolsys : Molsys
              input molecular system
    dq      : ndarray
              input coordinate step
    fq      : forces in internal coordinates (used for printing).
                passed in au. converted to aJ

    Returns
    -------
    np.ndarray
    """

    # Modify dq_in to account for frozen coordinates and ranged coordinates
    # These do not represent desired Delta(q)
    q_in = molsys.q_array()
    for iF, F in enumerate(molsys.fragments):
        if F.frozen:
            # For accounting only, since displace_frag is not called.
            dq_in[molsys.frag_intco_range(iF)] = 0
            logger.info("\tFragment %d is frozen, so not displacing" % (iF + 1))
        start = molsys.frag_1st_intco(iF)
        for i, I in enumerate(F.intcos):
            if I.frozen:
                dq_in[start + i] = 0.0
            elif I.ranged:
                tentative = q_in[start + i] + dq_in[start + i]
                if tentative > I.range_max:
                    dq_in[start + i] = I.range_max - q_in[start + i]
                    logger.info("setting to max: {:10.5f}".format(dq_in[start + i]))
                elif tentative < I.range_min:
                    dq_in[start + i] = I.range_min - q_in[start + i]
                    logger.info("setting to min: {:10.5f}".format(dq_in[start + i]))
                else:
                    pass  # value within range

    geom_in = molsys.geom
    q_in = molsys.q_array()  # recompute with limitations above
    q_target = q_in + dq_in

    for iF, F in enumerate(molsys.fragments):
        if F.frozen or F.num_intcos == 0:
            continue
        logger.info("\tDetermining Cartesian step for fragment %d." % (iF + 1))
        dq_frag, conv = displace_frag(F, dq_in[molsys.frag_intco_slice(iF)], ensure_convergence)

    for i, DI in enumerate(molsys.dimer_intcos):
        logger.info("\tTaking step for dimer coordinates of fragments %d and %d."
                    % (DI.A_idx + 1, DI.B_idx + 1))

        Axyz = molsys.frag_geom(DI.A_idx)
        Bxyz = molsys.frag_geom(DI.B_idx)
        Bxyz[:] = DI.orient_fragment(Axyz, Bxyz, q_target[molsys.dimerfrag_intco_slice(i)])

    geom_final = molsys.geom
    # Analyze relative to original input geometry
    molsys.geom = geom_in
    molsys.update_dihedral_orientations()
    molsys.fix_bend_axes()
    q_orig = molsys.q_array()
    qShow_orig = molsys.q_show_array()

    molsys.geom = geom_final
    q_final = molsys.q_array()
    qShow_final = molsys.q_show_array()

    dx = geom_final - geom_in

    dqShow = qShow_final - qShow_orig
    molsys.unfix_bend_axes()

    intco_lbls = molsys.intco_lbls

    coordinate_change_report = "\n\n\t        --- Internal Coordinate Step in ANG or DEG, aJ/ANG or AJ/DEG ---\n"
    coordinate_change_report += "\t-------------------------------------------------------------------------------\n"

    if fq is None:
        coordinate_change_report += "\t           Coordinate      Previous         Change          New \n"
        coordinate_change_report += "\t           ----------      --------        ------        ------\n"
        for i in range(len(dq_in)):
            coordinate_change_report += "\t%21s%14.5f%14.5f%14.5f\n" % (
                intco_lbls[i],
                qShow_orig[i],
                dqShow[i],
                qShow_final[i],
            )
    else:
        fq_aJ = molsys.q_show_forces(fq)  # print forces for step
        coordinate_change_report += (
            "\t           Coordinate      Previous         Force          Change          New \n"
        )
        coordinate_change_report += (
            "\t           ----------      --------        ------          ------        ------\n"
        )
        for i in range(len(dq_in)):
            coordinate_change_report += "\t%21s%14.5f%15.5f%15.5f%14.5f\n" % (
                intco_lbls[i],
                qShow_orig[i],
                fq_aJ[i],
                dqShow[i],
                qShow_final[i],
            )
    coordinate_change_report += "\t-------------------------------------------------------------------------------\n"
    logger.info(coordinate_change_report)

    # Return final, total displacement ACHIEVED
    dq = q_final - q_orig

    linear_list = linear_bend_check(molsys)
    if linear_list:
        raise AlgError("New linear angles", newLinearBends=linear_list)

    # RAK TODO : remember why I want to return dx and what to do with it.
    if return_str:
        return dq, dx, coordinate_change_report
    else:
        return dq, dx


def displace_frag(F, dq_in, ensure_convergence=False):
    """Converts internal coordinate step into the new cartesian geometry
    Parameters
    ----------
    F  : Fragment (geometry is changed)
    dq : ndarray
        step (displacement) in internal coordiantes
    ensure_convergence : bool
        reduce the magntitude of the step size as necessary until the
        iterative back-transformation actually converges.

    Returns
    -------
    tuple(np.ndarray, bool) : dq achieved, conv and frozen_conv
    """
    geom = F.geom
    dq = dq_in.copy()
    if not F.num_intcos or not len(geom) or not len(dq_in):
        return dq and True

    geom_orig = np.copy(geom)
    q_orig = F.q_array()

    best_geom = np.zeros(geom_orig.shape)
    conv = False  # is back-transformation converged?

    if ensure_convergence:
        cnt = -1

        while not conv:
            cnt += 1
            if cnt > 0:
                logger.info("\tReducing step-size by a factor of %d." % (2 * cnt))
                dq[:] = dq_in / (2.0 * cnt)

            F.fix_bend_axes()
            F.update_dihedral_orientations()
            conv = back_transformation(F.intcos, geom, dq, op.Params.print_lvl)
            F.unfix_bend_axes()

            if not conv:
                if cnt == 5:
                    logger.warning(
                        "\tUnable to back-transform even 1/10th of the desired step rigorously."
                        + "\tContinuing with best (small) step"
                    )
                    break
                else:
                    geom[:] = geom_orig  # put original geometry back for next try at smaller step.

        if conv and cnt > 0:  # We were able to take a modest step.  Try to complete it.
            logger.info("\tAble to take a small step; trying another partial back-transformations.\n")

            for j in range(1, 2 * cnt):
                logger.info("\tMini-step %d of %d.\n", j + 1, 2 * cnt)
                dq[:] = dq_in / (2 * cnt)

                best_geom[:] = geom

                F.fix_bend_axes()
                conv = back_transformation(F.intcos, geom, dq, op.Params.print_lvl)
                F.unfix_bend_axes()

                if not conv:
                    logger.warning("\tCouldn't converge this mini-step; quitting with previous geometry.\n")
                    geom[:] = best_geom
                    break

    else:  # try to back-transform, but continue even if desired dq is not achieved
        F.fix_bend_axes()
        F.update_dihedral_orientations()
        conv = back_transformation(F.intcos, geom, dq, op.Params.print_lvl)
        F.unfix_bend_axes()

        if op.Params.opt_type == "IRC" and not conv:
            raise OptError("Could not take constrained step in an IRC computation.")

    # Fix drift/error in any frozen coordinates
    frozen_conv = True
    if any(intco.frozen for intco in F.intcos) or any(intco.ranged for intco in F.intcos):

        F.update_dihedral_orientations()
        F.fix_bend_axes()
        qnow = intcosMisc.q_values(F.intcos, geom)
        dq_adjust_frozen = np.zeros(len(F.intcos))

        for i, intco in enumerate(F.intcos):
            if intco.frozen:  # cleanup step = -Dq
                dq_adjust_frozen[i] = q_orig[i] - qnow[i]
            elif intco.ranged:  # put within range
                if qnow[i] > intco.range_max:
                    dq_adjust_frozen[i] = intco.range_max - qnow[i]
                elif qnow[i] < intco.range_min:
                    dq_adjust_frozen[i] = intco.range_min - qnow[i]

        # For stability try scaling the adjustment if its quite long.
        # Slow progress towards the constraint is better than none
        if np.linalg.norm(dq_adjust_frozen) > 0.5:
            scale = 0.5 / np.linalg.norm(dq_adjust_frozen)
            dq_adjust_frozen *= scale

        frozen_msg = "\tAdditional back-transformation to adjust frozen/ranged coordinates: "

        frozen_conv = back_transformation(
            F.intcos,
            geom,
            dq_adjust_frozen,
            op.Params.print_lvl - 1,  # suppress printing
            bt_dx_conv=1.0e-12,
            bt_dx_rms_change_conv=1.0e-12,
            bt_max_iter=100,
        )

        F.unfix_bend_axes()

        if frozen_conv:
            frozen_msg += "successful.\n"
            logger.info(frozen_msg)
        else:
            frozen_msg += "unsuccessful, but continuing.\n"
            logger.info(frozen_msg)
            logger.warning(frozen_msg)

    # Make sure final Dq is actual change
    q_final = intcosMisc.q_values(F.intcos, geom)
    dq[:] = q_final - q_orig

    if op.Params.print_lvl >= 1:
        frag_report = "\tReport of back-transformation: (au)\n"
        frag_report += "\n\t  int       q_final         q_target          Error\n"
        frag_report += "\t---------------------------------------------------\n"
        q_target = q_orig + dq_in
        for i in range(F.num_intcos):
            frag_report += "\t%5d%15.10lf%15.10f%15.10lf\n" % (i + 1, q_final[i], q_target[i], (q_final - q_target)[i],)
        frag_report += "\t--------------------------------------------------\n"
        logger.debug(frag_report)

    return dq, conv and frozen_conv


def back_transformation(
    intcos, geom, dq, print_lvl, bt_dx_conv=None, bt_dx_rms_change_conv=None, bt_max_iter=None,
):

    dx_rms_last = -1
    if bt_dx_conv is None:
        bt_dx_conv = op.Params.bt_dx_conv
    if bt_dx_rms_change_conv is None:
        bt_dx_rms_change_conv = op.Params.bt_dx_rms_change_conv
    if bt_max_iter is None:
        bt_max_iter = op.Params.bt_max_iter

    q_orig = intcosMisc.q_values(intcos, geom)
    q_target = q_orig + dq

    if print_lvl > 1:
        target_step_str = "Back-transformation in back_transformation():\n"
        target_step_str += "          Original         Target           Dq\n"
        for i in range(len(dq)):
            target_step_str += "%15.10f%15.10f%15.10f\n" % (q_orig[i], q_target[i], dq[i],)
        logger.debug(target_step_str)

    if print_lvl > 0:
        step_iter_str = "\t             Back Transformation Report            "
        step_iter_str += "\n\t---------------------------------------------------\n"
        step_iter_str += "\t Iter        RMS(dx)        Max(dx)        RMS(dq) \n"
        step_iter_str += "\t---------------------------------------------------\n"

    new_geom = np.copy(geom)  # cart geometry to start each iter
    best_geom = np.zeros(new_geom.shape)

    bt_iter_continue = True
    bt_converged = False
    bt_iter_cnt = 0

    while bt_iter_continue:

        # dq_rms = rms(dq)
        dx_rms, dx_max = dq_to_dx(intcos, geom, dq, print_lvl > 2)

        # Met convergence thresholds
        if dx_rms < bt_dx_conv and dx_max < bt_dx_conv:
            bt_converged = True
            bt_iter_continue = False
        # No further progress toward convergence.
        elif np.absolute(dx_rms - dx_rms_last) < bt_dx_rms_change_conv or bt_iter_cnt >= bt_max_iter or dx_rms > 100.0:
            bt_converged = False
            bt_iter_continue = False

        dx_rms_last = dx_rms

        new_q = intcosMisc.q_values(intcos, geom)
        dq[:] = q_target - new_q
        del new_q

        dq_rms = rms(dq)
        if bt_iter_cnt == 0 or dq_rms < best_dq_rms:  # short circuit evaluation
            best_geom[:] = geom
            best_dq_rms = dq_rms

        if print_lvl > 0:
            step_iter_str += "\t%5d %14.1e %14.1e %14.1e\n" % (bt_iter_cnt + 1, dx_rms, dx_max, dq_rms,)
        bt_iter_cnt += 1

    if print_lvl > 0:
        step_iter_str += "\t---------------------------------------------------\n"
        logger.debug(step_iter_str)

    bt_final_step = f"\tRMS(dx): {dx_rms: .3e} \tMax(dx): {dx_max: .3e} \tRMS(dq): {dq_rms: .3e}"
    if bt_converged:
        logger.info("\tSuccessfully converged to displaced geometry.")
        logger.info(bt_final_step)
    else:
        logger.warning("\tUnable to completely converge to displaced geometry.")
        logger.warning(bt_final_step)

    if dq_rms > best_dq_rms:
        logger.warning("\tPrevious geometry is closer to target in internal coordinates, so using that one.\n")
        logger.warning("\tBest geometry has RMS(Delta(q)) = %8.2e\n" % best_dq_rms)
        geom[:] = best_geom

    return bt_converged


# Convert dq to dx.  Geometry is updated.
# B dx = dq
# B dx = (B Bt)(B Bt)^-1 dq
# B (dx) = B * [Bt (B Bt)^-1 dq]
#   dx = Bt (B Bt)^-1 dq
#   dx = Bt G^-1 dq, where G = B B^t.
def dq_to_dx(intcos, geom, dq, printDetails=False):
    """Convert dq to dx.  Geometry is updated

    Parameters
    ----------
    intcos : list of Stre, Bend, Tors, or Oofp
    geom : ndarray
        cartesian geometry updated to new geometry
    dq : displacement in internal coordinates

    Returns
    -------
    float :
        rms of cartesian displacement
    float :
        absolute maximum of cartesian displacement
    """
    B = intcosMisc.Bmat(intcos, geom)
    G = B @ B.T
    Ginv = np.linalg.pinv(G)
    dx = B.T @ Ginv @ dq

    if printDetails:
        qOld = intcosMisc.q_values(intcos, geom)

    geom += dx.reshape(geom.shape)

    if printDetails:
        dq_achieved = intcosMisc.q_values(intcos, geom) - qOld
        displacement_str = "\t      Report of Single-step\n"
        displacement_str += "\t  int       dq_achieved     deviation from target\n"
        for i in range(len(intcos)):
            displacement_str += "\t%5d%15.10f%15.10f\n" % (i + 1, dq_achieved[i], dq_achieved[i] - dq[i],)
        logger.debug(displacement_str)

    dx_rms = rms(dx)
    dx_max = abs_max(dx)
    del B, G, Ginv, dx
    return dx_rms, dx_max

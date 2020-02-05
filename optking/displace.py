import numpy as np
import logging

from . import intcosMisc
from .exceptions import AlgError, OptError
from . import optparams as op
from .linearAlgebra import absMax, rms, symmMatInv

# Functions in this file displace.py
#
#  displaceMolsys: Displace each fragment.  Displace dimer coordinates.
#  displaceFrag  : Displace a fragment by dq.  Double check frozen coordinates
#                 are satisfied.  Reduce stepsize as needed if
#                 ensure_convergence is true.
#  DqToDxIterate:  Call DxStep iteratively to try to converge to desired
#                  Dq as much as possible.
#  DxStep       : Given Delta(q), compute and invert B, take Delta(x) step.

# Displace molecular system
def displaceMolsys(oMolsys, dq, fq=None):
    """ Manage internal coordinate step for a molecular system
    Parameters
    ----------
    oMolsys : Molsys
              input molecular system
    dq      : ndarray
              input coordinate step
    fq      : forces in internal coordinates (used for printing).
    """
    logger = logging.getLogger(__name__)

    q = oMolsys.qArray()
    print('Initial q')
    print(q)
    q_target = q + dq
    # q_target is used for dimer coordinates; does it need corrected for
    # dihedrals through pi here?  don't think so.
    print('Target q')
    print(q_target)
    geom_orig = oMolsys.geom

    forces = None
    for iF,F in enumerate(oMolsys._fragments):
        logger.info("\tDetermining Cartesian step for fragment %d." % (iF+1))
        conv = displaceFrag(F, dq[oMolsys.frag_intco_slice(iF)])
        if conv:
            logger.info("\tStep for fragment succeeded.")
        else:
            logger.info("\tStep for fragment falied.")
            logger.warning("\tStep for fragment succeeded.")

    for i, DI in enumerate(oMolsys._dimer_intcos):
        Axyz = oMolsys.frag_geom( DI.A_idx )
        Bxyz = oMolsys.frag_geom( DI.B_idx )
        Bxyz[:] = DI.orient_fragment(Axyz, Bxyz,
                     q_target[oMolsys.dimerfrag_intco_slice(i)])

    geom_final = oMolsys.geom
    # Analyze relative to original input geometry
    oMolsys.geom = geom_orig
    oMolsys.updateDihedralOrientations()
    oMolsys.fixBendAxes()
    q_orig       = oMolsys.qArray()
    qShow_orig   = oMolsys.qShowArray()

    oMolsys.geom = geom_final
    q_final      = oMolsys.qArray()
    qShow_final  = oMolsys.qShowArray()

    # Set dq to final, total displacement ACHIEVED
    dq[:]        = q_final - q_orig
    dqShow       = qShow_final - qShow_orig
    oMolsys.unfixBendAxes()

    coordinate_change_report = (
        "\n\n\t       --- Internal Coordinate Step in ANG or DEG, aJ/ANG or AJ/DEG ---\n")
    coordinate_change_report += (
        "\t-----------------------------------------------------------------------------\n")

    if type(fq) == type(None):
        coordinate_change_report += (
            "\t         Coordinate      Previous         Change          New \n")
        coordinate_change_report += (
            "\t         ----------      --------        ------        ------\n")
        for i in range(len(dq)):
            coordinate_change_report += ("\t%19s%14.5f%14.5f%14.5f\n"
                                         % (i, qShow_orig[i], dqShow[i], qShow_final[i]))
    else:
        coordinate_change_report += (
            "\t         Coordinate      Previous         Force          Change          New \n")
        coordinate_change_report += (
            "\t         ----------      --------        ------          ------        ------\n")
        for i in range(len(dq)):
            coordinate_change_report += ("\t%19s%14.5f%14.5f%14.5f%14.5f\n"
                                         % (i, qShow_orig[i], fq[i], dqShow[i], qShow_final[i]))
    coordinate_change_report += (
        "\t-----------------------------------------------------------------------------\n")
    logger.info(coordinate_change_report)

    return


def displaceFrag(F, dq, ensure_convergence=False):
    """ Converts internal coordinate step into the new cartesian geometry
    Parameters
    ----------
    F  : Fragment
    dq : ndarray
        step (displacement) in internal coordiantes
        overriden to actual displacements performed
    ensure_convergence : bool
        reduce the magntitude of the step size as necessary until the
        iterative back-transformation actually converges.
    """
    logger = logging.getLogger(__name__)
    geom   = F.geom
    if not F.Nintcos or not len(geom) or not len(dq):
        dq[:] = 0
        return

    geom_orig = np.copy(geom)
    dq_orig   = np.copy(dq)
    q_orig    = F.qArray()

    best_geom = np.zeros(geom_orig.shape)
    conv = False # is back-transformation converged?

    if ensure_convergence:
        cnt = -1

        while not conv:
            cnt += 1
            if cnt > 0:
                logger.info("\tReducing step-size by a factor of %d." % (2 * cnt))
                dq[:] = dq_orig / (2.0 * cnt)

            F.fixBendAxes(geom)
            F.updateDihedralOrientations(geom)
            conv = DqToDxIterate(F.intcos, geom, dq, op.Params.print_lvl)
            F.unfixBendAxes()

            if not conv:
                if cnt == 5:
                    logger.warning(
                        "\tUnable to back-transform even 1/10th of the desired step rigorously."
                        + "\tContinuing with best (small) step")
                    break
                else:
                    geom[:] = geom_orig  # put original geometry back for next try at smaller step.

        if conv and cnt > 0:  # We were able to take a modest step.  Try to complete it.
            logger.info(
                "\tAble to take a small step; trying another partial back-transformations.\n")

            for j in range(1, 2 * cnt):
                logger.info("\tMini-step %d of %d.\n", (j + 1, 2 * cnt))
                dq[:] = dq_orig / (2 * cnt)

                best_geom[:] = geom

                F.fixBendAxes(geom)
                conv = DqToDxIterate(intcos, geom, dq, op.Params.print_lvl)
                F.unfixBendAxes()

                if not conv:
                    logger.warning(
                        "\tCouldn't converge this mini-step; quitting with previous geometry.\n")
                    geom[:] = best_geom
                    break

    else:  # try to back-transform, but continue even if desired dq is not achieved
        F.fixBendAxes(geom)
        F.updateDihedralOrientations(geom)
        conv = DqToDxIterate(intcos, geom, dq, op.Params.print_lvl)
        F.unfixBendAxes()

        if op.Params.opt_type == "IRC" and not conv:
            raise OptError("Could not take constrained step in an IRC computation.")

    # Fix drift/error in any frozen coordinates
    frozen_conv = True
    if any(intco.frozen for intco in intcos):

        # Set dq for unfrozen intcos to zero.
        F.updateDihedralOrientations(geom)
        F.fixBendAxes(geom)
        dq_adjust_frozen = q_orig - intcosMisc.qValues(intcos, geom)

        for i, intco in enumerate(intcos):
            if not intco.frozen:
                dq_adjust_frozen[i] = 0

        frozen_msg = (
                "\tAdditional back-transformation to adjust frozen coordinates:\n")

        frozen_conv = DqToDxIterate( intcos, geom,
            dq_adjust_frozen,
            op.Params.print_lvl-1, # suppress printing
            bt_dx_conv=1.0e-12,
            bt_dx_rms_change_conv=1.0e-12,
            bt_max_iter=100)

        F.unfixBendAxes()

        if check:
            frozen_msg += ("\tsuccessful.\n")
            logger.info(frozen_msg)
        else:
            frozen_msg += ("\tunsuccessful, but continuing.\n")
            logger.info(frozen_msg)
            logger.warning(frozen_msg)

    # Make sure final Dq is actual change
    q_final = intcosMisc.qValues(intcos, geom)
    dq[:] = q_final - q_orig

    if op.Params.print_lvl >= 1:
        frag_report = ("\tReport of back-transformation: (au)\n")
        frag_report += ("\n\t  int       q_final         q_target          Error\n")
        frag_report += (  "\t---------------------------------------------------\n")
        q_target = q_orig + dq_orig
        for i in range(F.Nintcos):
            frag_report += ("\t%5d%15.10lf%15.10f%15.10lf\n"
                                  % (i + 1, q_final[i], q_target[i], (q_final - q_target)[i]))
        frag_report += ("\t--------------------------------------------------\n")
        logger.debug(frag_report)

    return conv and frozen_conv


def DqToDxIterate(intcos, geom, dq, print_lvl,
             bt_dx_conv=None, bt_dx_rms_change_conv=None, bt_max_iter=None):

    logger = logging.getLogger(__name__)
    dx_rms_last = -1
    if bt_dx_conv is None:
        bt_dx_conv = op.Params.bt_dx_conv
    if bt_dx_rms_change_conv is None:
        bt_dx_rms_change_conv = op.Params.bt_dx_rms_change_conv
    if bt_max_iter is None:
        bt_max_iter = op.Params.bt_max_iter

    q_orig = intcosMisc.qValues(intcos, geom)
    q_target = q_orig + dq

    if print_lvl > 1:
        target_step_str = "Back-transformation in DqToDxIterate():\n"
        target_step_str += "          Original         Target           Dq\n"
        for i in range(len(dq)):
            target_step_str += "%15.10f%15.10f%15.10f\n" % (q_orig[i], q_target[i], dq[i])
        logger.info(target_step_str)

    if print_lvl > 0:
        step_iter_str = ("\n\n\t---------------------------------------------------\n")
        step_iter_str += ("\t Iter        RMS(dx)        Max(dx)        RMS(dq) \n")
        step_iter_str += ("\t---------------------------------------------------\n")

    new_geom = np.copy(geom)  # cart geometry to start each iter
    best_geom = np.zeros(new_geom.shape)

    bt_iter_continue = True
    bt_converged = False
    bt_iter_cnt = 0

    while bt_iter_continue:

        #dq_rms = rms(dq)
        dx_rms, dx_max = DxStep(intcos, geom, dq, print_lvl > 2)

        # Met convergence thresholds
        if dx_rms < bt_dx_conv and dx_max < bt_dx_conv:
            bt_converged = True
            bt_iter_continue = False
        # No further progress toward convergence.
        elif (np.absolute(dx_rms - dx_rms_last) < bt_dx_rms_change_conv
              or bt_iter_cnt >= bt_max_iter or dx_rms > 100.0):
            bt_converged = False
            bt_iter_continue = False

        dx_rms_last = dx_rms

        new_q = intcosMisc.qValues(intcos, geom)
        dq[:] = q_target - new_q
        del new_q

        dq_rms = rms(dq)
        if bt_iter_cnt == 0 or dq_rms < best_dq_rms:  # short circuit evaluation
            best_geom[:] = geom
            best_dq_rms = dq_rms

        if print_lvl > 0:
            step_iter_str += ("\t%5d %14.1e %14.1e %14.1e\n"
                              % (bt_iter_cnt + 1, dx_rms, dx_max, dq_rms))
        bt_iter_cnt += 1

    if print_lvl > 0:
        step_iter_str += ("\t---------------------------------------------------\n")
        logger.info(step_iter_str)

    if bt_converged:
        logger.info("\tSuccessfully converged to displaced geometry.")
    else:
        logger.warning("\tUnable to completely converge to displaced geometry.")

    if dq_rms > best_dq_rms:
        logger.warning("\tPrevious geometry is closer to target in internal coordinates,"
                       + " so using that one.\n")
        logger.warning("\tBest geometry has RMS(Delta(q)) = %8.2e\n" % best_dq_rms)
        geom[:] = best_geom

    return bt_converged


# Convert dq to dx.  Geometry is updated.
# B dx = dq
# B dx = (B Bt)(B Bt)^-1 dq
# B (dx) = B * [Bt (B Bt)^-1 dq]
#   dx = Bt (B Bt)^-1 dq
#   dx = Bt G^-1 dq, where G = B B^t.
def DxStep(intcos, geom, dq, printDetails=False):
    """ Convert dq to dx.  Geometry is updated

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
    G = np.dot(B, B.T)
    Ginv = symmMatInv(G, redundant=True)
    tmp_v_Nint = np.dot(Ginv, dq)
    dx = np.dot(B.T, tmp_v_Nint)

    if printDetails:
        qOld = intcosMisc.qValues(intcos, geom)

    geom += dx.reshape(geom.shape)

    if printDetails:
        dq_achieved = intcosMisc.qValues(intcos, geom) - qOld
        displacement_str =  "\t      Report of Single-step\n"
        displacement_str += "\t  int       dq_achieved     deviation from target\n"
        for i in range(len(intcos)):
            displacement_str += "\t%5d%15.10f%15.10f\n" % (i + 1,
                dq_achieved[i], dq_achieved[i] - dq[i])
        logger.info(displacement_str)

    dx_rms = rms(dx)
    dx_max = absMax(dx)
    del B, G, Ginv, tmp_v_Nint, dx
    return dx_rms, dx_max

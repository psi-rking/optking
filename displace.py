import numpy as np
import optParams as op
import intcosMisc
from linearAlgebra import absMax,rms,symmMatInv

# dq : displacements in internal coordinates to be performed.
#      On exit, overridden to actual displacements performed.
# fq : internal coordinate forces (used for printing).
# atom_offset : increment to atom #'s in the fragment (used for printing)
# ensure_convergence :
#   Reduce step size as necessary until back-transformation converges.

def displace(intcos, geom, dq, fq, atom_offset=0, ensure_convergence=False):
    if not len(intcos) or not len(geom) or not len(dq):
        dq[:] = 0
        return

    Nint = len(intcos)

    intcosMisc.updateDihedralOrientations(intcos, geom)
    geom_orig    = np.copy(geom)
    dq_orig      = np.copy(dq)
    intcosMisc.unfixBendAxes(intcos)
    q_orig       = intcosMisc.qValues(intcos, geom_orig);

    best_geom    = np.zeros( geom_orig.shape ,float)

    # Do your best to backtransform all internal coordinate displacments.
    print "\tBeginnning displacement in cartesian coordinates..."

    if ensure_convergence:
        conv = False
        cnt = -1

        while not conv:
            ++cnt;
            if cnt > 0:
                print "Reducing step-size by a factor of %d.\n" % (2*cnt)
                dq[:] = dq_orig / (2.0*cnt)

            intcosMisc.fixBendAxes(intcos, geom)
            conv = stepIter(intcos, geom, dq);
            intcosMisc.unfixBendAxes(intcos)

            if not conv:
                if cnt == 5:
                    print "\tUnable to back-transform even 1/10th of the desired step rigorously."
                    print "\tContinuing with best (small) step."
                    break;
                else:
                    geom[:] = geom_orig  # put original geometry back for next try at smaller step.

        if conv and cnt > 0:  # We were able to take a modest step.  Try to complete it.
            print "\tAble to take a small step; trying another partial back-transformations."

            for j in range(1,2*cnt):
                print "Mini-step %d of %d.\n", (j+1, 2*cnt)
                dq[:] = dq_orig / (2*cnt);

                best_geom[:] = geom

                intcosMisc.fixBendAxes(intcos, geom)
                conv = stepIter(intcos, geom, dq)
                intcosMisc.unfixBendAxes(intcos)

                if not conv:
                    print "\tCouldn't converge this mini-step, so quitting with previous geometry."
                    geom[:] = best_geom
                    break

    else: # try to back-transform, but continue even if desired dq is not achieved
        intcosMisc.fixBendAxes(intcos, geom)
        stepIter(intcos, geom, dq);
        intcosMisc.unfixBendAxes(intcos)

    # Fix drift/error in any frozen coordinates
    if any(intco.frozen for intco in intcos):

        # Set dq for unfrozen intcos to zero.
        dq_adjust_frozen = q_orig - intcosMisc.qValues(intcos,geom)

        for i, intco in enumerate(intcos):
            if not intco.frozen:
                dq_adjust_frozen[i] = 0

        print "\n\tBack-transformation to cartesian coordinates to adjust frozen coordinates..."

        intcosMisc.fixBendAxes(intcos, geom)
        check = stepIter(intcos, geom, dq_adjust_frozen, bt_dx_conv=1.0e-12, \
                             bt_dx_rms_change_conv=1.0e-12, bt_max_iter=100)
        intcosMisc.unfixBendAxes(intcos)

        if check: print "\tsuccessful."
        else:     print "\tunsuccessful, but continuing."

    # Make sure final Dq is actual change
    q_final = intcosMisc.qValues(intcos, geom)
    dq[:]   = q_final - q_orig

    if op.Params.print_lvl > 1:
        print "\n\tReport of back-transformation: (au)"
        print "\t  int       q_target          Error"
        print "\t-----------------------------------"
        q_target = q_orig + dq_orig
        for i in range(Nint):
            print "\t%5d%15.10lf%15.10lf" % (i+1, q_target[i], (q_final-q_target)[i])
        print "\t-----------------------------------"

    # Set dq to final, total displacement ACHIEVED
    qShow_final = intcosMisc.qShowValues(intcos, geom)
    qShow_orig  = intcosMisc.qShowValues(intcos, geom_orig)
    dqShow      = qShow_final - qShow_orig;

    print "\n\t      --- Internal Coordinate Step in ANG or DEG, aJ/ANG or AJ/DEG ---"
    print   "\t----------------------------------------------------------------------------"
    print   "\t     Coordinate       Previous          Force         Change           New "
    print   "\t     ----------       --------         ------         ------        ------"
    for i, intco in enumerate(intcos):
        print "\t%15s%15.6f%15.6f%15.6f%15.6f" % (intco, qShow_orig[i], fq[i], dqShow[i], qShow_final[i])
    print "\t----------------------------------------------------------------------------\n"


def stepIter(intcos, geom, dq, bt_dx_conv=None, bt_dx_rms_change_conv=None, bt_max_iter=None):
    dx_rms_last = -1
    if not bt_dx_conv:            bt_dx_conv  = op.Params.bt_dx_conv
    if not bt_dx_rms_change_conv: bt_dx_rms_change_conv = op.Params.bt_dx_rms_change_conv
    if not bt_max_iter:           bt_max_iter = op.Params.bt_max_iter
    print_lvl = op.Params.print_lvl

    q_orig = intcosMisc.qValues(intcos, geom)
    q_target = q_orig + dq

    if print_lvl > 1:
        print "\t---------------------------------------------------"
        print "\t Iter        RMS(dx)        Max(dx)        RMS(dq) "
        print "\t---------------------------------------------------"

    new_geom   = np.copy(geom)          # cart geometry to start each iter
    best_geom  = np.zeros( new_geom.shape, float)

    bt_iter_continue = True
    bt_converged = False
    bt_iter_cnt = 0

    while bt_iter_continue:

        dq_rms = rms(dq)
        dx_rms, dx_max = oneStep(intcos, geom, dq, print_lvl>2)

        # Met convergence thresholds
        if dx_rms < bt_dx_conv and dx_max < bt_dx_conv:
            bt_converged = True
            bt_iter_continue = False
        # No further progress toward convergence.
        elif np.absolute(dx_rms - dx_rms_last) < bt_dx_rms_change_conv \
          or bt_iter_cnt >= bt_max_iter \
          or dx_rms > 100.0:
            bt_converged = False
            bt_iter_continue = False

        dx_rms_last = dx_rms

        new_q = intcosMisc.qValues(intcos, geom)
        dq[:] = q_target - new_q
        del new_q

        if bt_iter_cnt == 0 or dq_rms < best_dq_rms:
            best_geom[:] = geom
            best_dq_rms = dq_rms

        if print_lvl > 1:
            print "\t%5d %14.1e %14.1e %14.1e" % (bt_iter_cnt+1, dx_rms, dx_max, dq_rms)

        bt_iter_cnt += 1

    if print_lvl > 1:
        print "\t---------------------------------------------------"

    if bt_converged:    print "\tSuccessfully converged to displaced geometry."
    else:               print "\tUnable to completely converge to displaced geometry."

    if dq_rms > best_dq_rms:
        print "\tPrevious geometry is closer to target in internal coordinates, so using that one."
        print "\tBest geometry has RMS(Delta(q)) = %8.2e\n" % best_dq_rms
        geom[:] = best_geom

    if op.Params.opt_type == "IRC" and not bt_converged:
        raise INTCO_EXCEPTION("Could not take constrained step in an IRC computation.")

    return bt_converged


# Convert dq to dx.  Geometry is updated.
# B dx = dq
# B dx = (B Bt)(B Bt)^-1 dq
# B (dx) = B * [Bt (B Bt)^-1 dq]
#   dx = Bt (B Bt)^-1 dq
#   dx = Bt G^-1 dq, where G = B B^t.
def oneStep(intcos, geom, dq, printDetails=False):
    B = intcosMisc.Bmat(intcos, geom)
    G = np.dot(B, B.T)
    Ginv          = symmMatInv(G,redundant=True)
    tmp_v_Nint    = np.dot(Ginv, dq)
    dx            = np.zeros( geom.shape[0]*geom.shape[1], float)  # dx is 1D here

    dx[:]         = np.dot(B.T, tmp_v_Nint)
    if printDetails: qOld = intcosMisc.qValues(intcos, geom)
    geom          += dx.reshape(geom.shape)

    if printDetails:
        dq_achieved = intcosMisc.qValues(intcos, geom) - qOld
        print "\t      Report of Single-step"
        print "\t  int       dq_achieved        dq_error"
        for i in range(len(intcos)):
            print "\t%5d%15.10lf%15.10lf" % (i+1, dq_achieved[i], dq_achieved[i]-dq[i])

    dx_rms = rms(dx)
    dx_max = absMax(dx)
    del B,G,Ginv,tmp_v_Nint,dx
    return dx_rms, dx_max


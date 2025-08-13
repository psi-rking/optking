# Test the analytic B matrix (dq/dx) via finite differences
# The 5-point formula should be good to DISP_SIZE^4 -
#  a few unfortunates will be slightly worse.
# Returns True or False, doesn't raise exceptions
import copy
import logging
from math import fabs

import numpy as np

from . import intcosMisc
from .printTools import print_mat_string
from . import log_name
from . import op

logger = logging.getLogger(f"{log_name}{__name__}")


def test_b(oMolsys):
    Natom = oMolsys.natom
    Nintco = oMolsys.num_intcos
    DISP_SIZE = 0.01
    MAX_ERROR = 50 * DISP_SIZE * DISP_SIZE * DISP_SIZE * DISP_SIZE

    logger.info("\tTesting B-matrix numerically...")

    B_analytic = oMolsys.Bmat()

    if op.Params.print_lvl >= 3:
        logger.debug("Analytic B matrix in au")
        logger.debug(print_mat_string(B_analytic))

    B_fd = np.zeros((Nintco, 3 * Natom))

    oMolsys.update_dihedral_orientations()
    oMolsys.fix_bend_axes()

    geom_orig = oMolsys.geom  # to restore below
    coord = oMolsys.geom  # returns a copy

    for atom in range(Natom):
        for xyz in range(3):
            coord[atom, xyz] -= DISP_SIZE
            oMolsys.geom = coord
            q_m = oMolsys.q()

            coord[atom, xyz] -= DISP_SIZE
            oMolsys.geom = coord
            q_m2 = oMolsys.q()

            coord[atom, xyz] += 3 * DISP_SIZE
            oMolsys.geom = coord
            q_p = oMolsys.q()

            coord[atom, xyz] += DISP_SIZE
            oMolsys.geom = coord
            q_p2 = oMolsys.q()

            coord[atom, xyz] -= 2 * DISP_SIZE  # restore to original
            for i in range(Nintco):
                B_fd[i, 3 * atom + xyz] = (q_m2[i] - 8 * q_m[i] + 8 * q_p[i] - q_p2[i]) / (
                    12.0 * DISP_SIZE
                )

    if op.Params.print_lvl >= 3:
        logger.debug(
            "Numerical B matrix in au, DISP_SIZE = %lf\n" % DISP_SIZE + print_mat_string(B_fd)
        )

    oMolsys.geom = geom_orig  # restore original
    oMolsys.unfix_bend_axes()

    max_error = -1.0
    max_error_intco = -1
    for i in range(Nintco):
        for j in range(3 * Natom):
            if fabs(B_analytic[i, j] - B_fd[i, j]) > max_error:
                max_error = fabs(B_analytic[i][j] - B_fd[i][j])
                max_error_intco = i

    logger.info(
        "\t\tMaximum difference is %.1e for internal coordinate %d."
        % (max_error, max_error_intco + 1)
    )
    # logger.info("\t\tThis coordinate is %s" % str(intcos[max_error_intco]))

    if max_error > MAX_ERROR:
        logger.warning(
            "\tB-matrix could be in error. However, numerical tests may fail for\n"
            + "\ttorsions at 180 degrees, and slightly for linear bond angles."
            + "This is OK.\n"
        )
        return False
    else:
        logger.info("\t...Passed.")
        return True


# Test the analytic derivative B matrix (d2q/dx2) via finite differences
# The 5-point formula should be good to DISP_SIZE^4 -
#  a few unfortunates will be slightly worse
def test_derivative_b(oMolsys):
    logger = logging.getLogger(__name__)
    DISP_SIZE = 0.01
    MAX_ERROR = 10 * DISP_SIZE * DISP_SIZE * DISP_SIZE * DISP_SIZE

    geom_orig = oMolsys.geom  # to restore below

    logger.info("\tTesting Derivative B-matrix numerically.")
    if oMolsys._dimer_intcos:
        logger.info("\tDerivative B-matrix for interfragment modes not yet implemented.")

    warn = False
    for iF, F in enumerate(oMolsys._fragments):
        logger.info("\t\tTesting fragment %d." % (iF + 1))

        Natom = F.natom
        Nintco = F.num_intcos
        coord = F.geom  # not a copy
        dq2dx2_fd = np.zeros((3 * Natom, 3 * Natom))
        dq2dx2_analytic = np.zeros((3 * Natom, 3 * Natom))

        for i, I in enumerate(F._intcos):
            logger.info("\t\tTesting internal coordinate %d :" % (i + 1))

            dq2dx2_analytic.fill(0)
            I.Dq2Dx2(coord, dq2dx2_analytic)

            if op.Params.print_lvl >= 3:
                logger.info(
                    "Analytic B' (Dq2Dx2) matrix in au\n" + print_mat_string(dq2dx2_analytic)
                )

            # compute B' matrix from B matrices
            for atom_a in range(Natom):
                for xyz_a in range(3):
                    coord[atom_a, xyz_a] += DISP_SIZE
                    B_p = intcosMisc.Bmat(F.intcos, coord)

                    coord[atom_a, xyz_a] += DISP_SIZE
                    B_p2 = intcosMisc.Bmat(F.intcos, coord)

                    coord[atom_a, xyz_a] -= 3.0 * DISP_SIZE
                    B_m = intcosMisc.Bmat(F.intcos, coord)

                    coord[atom_a, xyz_a] -= DISP_SIZE
                    B_m2 = intcosMisc.Bmat(F.intcos, coord)

                    coord[atom_a, xyz_a] += 2 * DISP_SIZE  # restore coord to orig

                    for atom_b in range(Natom):
                        for xyz_b in range(3):
                            dq2dx2_fd[3 * atom_a + xyz_a, 3 * atom_b + xyz_b] = (
                                B_m2[i, 3 * atom_b + xyz_b]
                                - 8 * B_m[i, 3 * atom_b + xyz_b]
                                + 8 * B_p[i, 3 * atom_b + xyz_b]
                                - B_p2[i][3 * atom_b + xyz_b]
                            ) / (12.0 * DISP_SIZE)

            if op.Params.print_lvl >= 3:
                logger.info(
                    "\nNumerical B' (Dq2Dx2) matrix in au, DISP_SIZE = %f\n" % DISP_SIZE
                    + print_mat_string(dq2dx2_fd)
                )

            max_error = -1.0
            max_error_xyz = (-1, -1)
            for I in range(3 * Natom):
                for J in range(3 * Natom):
                    if fabs(dq2dx2_analytic[I, J] - dq2dx2_fd[I, J]) > max_error:
                        max_error = fabs(dq2dx2_analytic[I][J] - dq2dx2_fd[I][J])
                        max_error_xyz = (I, J)

            logger.info(
                "\t\tMax. difference is %.1e; 2nd derivative wrt %d and %d."
                % (max_error, max_error_xyz[0], max_error_xyz[1])
            )

            if max_error > MAX_ERROR:
                warn = True

    oMolsys.geom = geom_orig  # restore original
    oMolsys.unfix_bend_axes()

    if warn:
        logger.warning(
            """
        \tSome values did not agree.  However, numerical tests may fail for
        \ttorsions at 180 degrees and linear bond angles. This is OK
        \tIf discontinuities are interfering with a geometry optimization
        \ttry restarting your optimization at an updated geometry, and/or
        \tremove angular coordinates that are fixed by symmetry."""
        )
        return False
    else:
        logger.info("\t...Passed.")
        return True

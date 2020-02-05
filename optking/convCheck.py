""" keywords
Code to check convergence.
P.g_convergence = uod.get('G_CONVERGENCE', 'QCHEM')
P.max_force_g_convergence =  uod.get('MAX_FORCE_G_CONVERGENCE', 3.0e-4)
P.rms_force_g_convergence = uod.get('RMS_FORCE_G_CONVERGENCE', 3.0e-4)
P.max_energy_g_convergence = uod.get('MAX_ENERGY_G_CONVERGENCE', 1.0e-6)
P.max_disp_g_convergence = uod.get('MAX_DISP_G_CONVERGENCE', 1.2e-3)
P.rms_disp_g_convergence = uod.get('RMS_DISP_G_CONVERGENCE', 1.2e-3)
P.flexible_g_convergence = uod.get('FLEXIBLE_G_CONVERGENCE', False)
"""
import logging
import numpy as np
from math import fabs

from . import optparams as op
from .linearAlgebra import absMax, rms, symmMatRoot
from .intcosMisc import Gmat
from .printTools import printArrayString, printMatString

# Check convergence criteria and print status to output file.
# return True, if geometry is optimized
# By default, checks maximum force and (Delta(E) or maximum disp)


def convCheck(iterNum, oMolsys, dq, f, energies, q_pivot=None):
    logger = logging.getLogger(__name__)
    logger.info("Performing convergence check.")
    has_fixed = False
    for F in oMolsys._fragments:
        if any([ints.fixedEqVal for ints in F.intcos]):
            has_fixed = True
    for DI in oMolsys._dimer_intcos:
        if any([ints.fixedEqVal for ints in DI._pseudo_frag._intcos]):
            has_fixed = True
    energy = energies[-1]
    last_energy = energies[-2] if len(energies) > 1 else 0.0

    # Save original forces and put back in below.
    if op.Params.opt_type == 'IRC' or has_fixed:
        f_backup = np.copy(f)

    DE = energy - last_energy

    # Remove arbitrary forces for user-specified equilibrium values.
    if has_fixed:
        logger.info("Forces used to impose fixed constraints are not included in convergence check.")
        for i, ints in enumerate(oMolsys.intcos):
            if ints.fixedEqVal:
                f[i] = 0

    if op.Params.opt_type == 'IRC':
        G_m = Gmat(oMolsys, oMolsys.masses)
        G_m_inv = np.linalg.inv(G_m)
        q = qValues(oMolsys.intcos, oMolsys.geom)
        logger.info("Projecting out forces parallel to reaction path.")

        p = np.subtract(q, q_pivot)
        logger.info("\np, current step from pivot point (not previous point on rxnpath):\n"+
                    printArrayString(p))

        logger.info("\nForces at current point on hypersphere\n"+ printArrayString(f))

        # Gradient perpendicular to p and tangent to hypersphere is:
        # g_m' = g_m - (g_m^t p_m / (p_m^t p_m) p_m, or
        # g'   = g   - (g^t p / (p^t G^-1 p)) G^-1 p
        # Ginv_p = np.array(oMolsys.Nintco, float)
        G_m_inv_p = np.dot(G_m_inv, p)
        f[:] = np.subtract(f, np.multiply(np.dot(f, p)/np.dot(p, G_m_inv_p), G_m_inv_p)) 
        logger.debug("\nForces perpendicular to hypersphere.\n"+ printArrayString(f))

    # Compute forces after projection and removal above.
    max_force = absMax(f)
    rms_force = rms(f)
    max_disp = absMax(dq)
    rms_disp = rms(dq)

    if op.Params.opt_type != 'IRC2':
        conv_str = """\n\t==> Convergence Check <==
        \n\tMeasures of convergence in internal coordinates in au.
        \n\tCriteria marked as inactive (o), active & met (*), and active & unmet ( ).
        \n\t---------------------------------------------------------------------------------------------"""

        if iterNum == 0:
            conv_str += " ~\n"
        else:
            conv_str += "\n"
        conv_str += (
            "\t  Step     Total Energy     Delta E     MAX Force     RMS Force      "
            + "MAX Disp      RMS Disp   ")
        if iterNum == 0:
            conv_str += " ~\n"
        else:
            conv_str += "\n"
        conv_str += (
            "\t--------------------------------------------------------------------------"
            + "-------------------")
        if iterNum == 0:
            conv_str += " ~\n"
        else:
            conv_str += "\n"
        conv_str += "\t  Convergence Criteria "
        if op.Params.i_max_DE:
            conv_str += ("  %10.2e %1s" % (op.Params.conv_max_DE, "*"))
        else:
            conv_str += ("             %1s" % "o")
        if op.Params.i_max_force:
            conv_str += ("  %10.2e %1s" % (op.Params.conv_max_force, "*"))
        else:
            conv_str += ("             %1s" % "o")
        if op.Params.i_rms_force:
            conv_str += ("  %10.2e %1s" % (op.Params.conv_rms_force, "*"))
        else:
            conv_str += ("             %1s" % "o")
        if op.Params.i_max_disp:
            conv_str += ("  %10.2e %1s" % (op.Params.conv_max_disp, "*"))
        else:
            conv_str += ("             %1s" % "o")
        if op.Params.i_rms_disp:
            conv_str += ("  %10.2e %1s" % (op.Params.conv_rms_disp, "*"))
        else:
            conv_str += ("             %1s" % "o")
        if iterNum == 0:
            conv_str += "  ~\n"
        else:
            conv_str += "\n"
        conv_str += (
                "\t----------------------------------------------------------------------"
                + "-----------------------")
        if iterNum == 0:
            conv_str += " ~\n"
        else:
            conv_str += "\n"
        conv_str += (
            "\t  %4d %16.8f  %10.2e %1s  %10.2e %1s  %10.2e %1s  %10.2e %1s  %10.2e %1s  ~\n"
            % (iterNum + 1, energy, DE, ('*' if fabs(DE) < op.Params.conv_max_DE else "")
               if op.Params.i_max_DE else 'o', max_force,
               ('*' if fabs(max_force) < op.Params.conv_max_force else "")
               if op.Params.i_max_force else 'o', rms_force,
               ('*' if fabs(rms_force) < op.Params.conv_rms_force else "")
               if op.Params.i_rms_force else 'o', max_disp,
               ('*' if fabs(max_disp) < op.Params.conv_max_disp else "")
               if op.Params.i_max_disp else 'o', rms_disp,
               ('*' if fabs(rms_disp) < op.Params.conv_rms_disp else "")
               if op.Params.i_rms_disp else 'o'))
        conv_str += (
                "\t----------------------------------------------------------------------"
                + "-----------------------\n\n")
        logger.info(conv_str)

# Return forces to what they were when conv_check was called (Why?? What does this do? - Alex)
    if op.Params.opt_type == 'IRC' or has_fixed:
        f[:] = f_backup

    return test_for_convergence(DE, max_force, rms_force, max_disp, rms_disp)


def test_for_convergence(DE, max_force, rms_force, max_disp, rms_disp):
    """ Tests whether all condditions for convergence have been met"""

    logger = logging.getLogger(__name__)
    logger.debug("Testing convergence parameters.")

    converge = op.Params.g_convergence
    untampered = op.Params.i_untampered  # i_untampered see note below
    conv_max_force = (max_force < op.Params.conv_max_force)
    conv_rms_force = (rms_force < op.Params.conv_rms_force)
    conv_max_disp = (max_disp < op.Params.conv_max_disp)
    conv_rms_disp = (rms_force < op.Params.conv_rms_disp)
    conv_max_DE = (fabs(DE) < op.Params.conv_max_DE)
    flat_potential = 100 * rms_force < conv_rms_force

    DE_inactive = not op.Params.i_max_DE
    max_f_inactive = not op.Params.i_max_force
    rms_f_inactive = not op.Params.i_rms_force
    max_disp_inactive = not op.Params.i_max_disp
    rms_disp_inactive = not op.Params.i_rms_disp

    converge_type = converge == 'QCHEM' or converge == "MOLPRO"
    converge_gau = (converge == "GAU" or converge == "GAU_TIGHT"
                    or converge == 'GAU_VERYTIGHT' or converge == 'GAU_LOOSE')

    min_requirements = (conv_max_force and (conv_max_DE or conv_max_disp))
    min_requirements_gau = (flat_potential or (conv_max_force and conv_rms_force
                                               and conv_max_disp and conv_max_disp))

    # The requirement of untampered means that if a user explicitly adds any of the
    # 5 indiv. criteria on top of G_CONVERGENCE, it is required to be met.

    # If QChem or MOLPRO, forces, and either energy change or displacement met, convergence!
    if untampered and converge_type and min_requirements:
        return True
    # if max/rms forces/disp met or flat potential forces met, convergence!
    elif untampered and converge_gau and min_requirements_gau:
        return True
    # if criterion not active or criterion met, convergence!
    elif ((conv_max_DE or DE_inactive) and (conv_max_force or max_f_inactive)
            and (conv_rms_force or rms_f_inactive) and (conv_max_disp or max_disp_inactive)
            and (conv_rms_disp or rms_disp_inactive)):
        return True
    elif (op.Params.opt_type == 'IRC' and (DE_inactive or conv_max_DE)
            and (max_disp_inactive or conv_max_disp) and (rms_disp_inactive or conv_rms_disp)):
        return True
    else:
        return False


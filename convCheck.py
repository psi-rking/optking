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

import numpy as np
import optParams as op
from math import fabs
from linearAlgebra import absMax, rms

# Check convergence criteria and print status to output file.
# return True, if geometry is optimized
# By default, checks maximum force and (Delta(E) or maximum disp)

def convCheck(iterNum, intcos, dq, f, energies, masses=None):
    max_disp = absMax(dq)
    rms_disp = rms(dq)
    Nintcos = len(intcos)
    has_fixed = any([ints.fixedEqVal for ints in intcos])
    energy = energies[-1]
    last_energy = energies[-2] if len(energies)>1 else 0.0

#   if op.Params.opt_type == 'IRC'
#       if ircData.go:  return True

    # Save original forces and put back in below.
    if op.Params.opt_type == 'IRC' or has_fixed:
        f_backup = np.copy(f)

    DE = energy - last_energy

    # Remove arbitrary forces for user-specified equilibrium values. 
    if has_fixed:
        print "\tForces used to impose fixed constraints are not included in convergence check."
        for i,ints in enumerate(intcos):
            if ints.fixedEqVal:
                f[i] = 0

    if op.Params.opt_type == 'IRC':
        G = Gmat(masses)
        Ginv = np.linalg.inv(G)

        # compute p_m, mass-weighted hypersphere vector
        q_pivot = irc_data.g_q_pivot();
        q = mol.coord_values();
        p = np.array( Nintco, float)
        p[:] = q - q_pivot

        # gradient perpendicular to p and tangent to hypersphere is:
        # g_m' = g_m - (g_m^t p_m / p_m^t p_m) p_m, or
        # g'   = g   - (g^t p / (p^t G^-1 p)) G^-1 p
        Ginv_p = np.array(Nintco, float)
        for i in range(Nintco):
            Ginv_p[i] = np.dot(Ginv[i], p)

        overlap = np.dot(f, p) / np.dot(p, Ginv_p)

        for i in range(Nintco):
            f[i] -= overlap * Ginv_p[i];

        if op.Params.print_level > 1:
            print "Forces perpendicular to hypersphere."
            misc.printArray(f)

    # Compute forces after projection and removal above.
    max_force = absMax(f)
    rms_force = rms(f)

    if op.Params.opt_type != 'IRC':
        print "\n  ==> Convergence Check <==\n"
        print "  Measures of convergence in internal coordinates in au."
        print "  Criteria marked as inactive (o), active & met (*), and active & unmet ( )."
        print "  ---------------------------------------------------------------------------------------------",
        if iterNum == 0: print "~"
        else: print
        print "   Step     Total Energy     Delta E     MAX Force     RMS Force      MAX Disp      RMS Disp   ",
        if iterNum == 0: print "~"
        else: print
        print "  ---------------------------------------------------------------------------------------------",
        if iterNum == 0: print "~"
        else: print
        print "    Convergence Criteria",
        if op.Params.i_max_DE   : print " %10.2e %1s" % (op.Params.conv_max_DE,    "*"),
        else: print "            %1s" % "o",
        if op.Params.i_max_force: print " %10.2e %1s" % (op.Params.conv_max_force, "*"),
        else: print "            %1s" % "o",
        if op.Params.i_rms_force: print " %10.2e %1s" % (op.Params.conv_rms_force, "*"),
        else: print "            %1s" % "o",
        if op.Params.i_max_disp : print " %10.2e %1s" % (op.Params.conv_max_disp,  "*"),
        else: print "            %1s" % "o",
        if op.Params.i_rms_disp : print " %10.2e %1s" % (op.Params.conv_rms_disp,  "*"),
        else: print "            %1s" % "o",
        if iterNum == 0: print " ~"
        else: print
        print "  ---------------------------------------------------------------------------------------------",
        if iterNum == 0: print "~"
        else: print
        print "   %4d %16.8f  %10.2e %1s  %10.2e %1s  %10.2e %1s  %10.2e %1s  %10.2e %1s  ~" % (iterNum+1, energy,
          DE,        ('*' if fabs(DE)       < op.Params.conv_max_DE    else "") if op.Params.i_max_DE    else 'o',
          max_force, ('*' if fabs(max_force)< op.Params.conv_max_force else "") if op.Params.i_max_force else 'o',
          rms_force, ('*' if fabs(rms_force)< op.Params.conv_rms_force else "") if op.Params.i_rms_force else 'o',
          max_disp,  ('*' if fabs(max_disp) < op.Params.conv_max_disp  else "") if op.Params.i_max_disp  else 'o',
          rms_disp,  ('*' if fabs(rms_disp) < op.Params.conv_rms_disp  else "") if op.Params.i_rms_disp  else 'o' )
        print "  ---------------------------------------------------------------------------------------------\n"
#
  
    # Return forces to what they were when conv_check was called
    if op.Params.opt_type == 'IRC' or has_fixed:
        f[:] = f_backup

    #   The requirement of i_untampered means that if a user explicitly adds any of the
    #   5 indiv. criteria on top of G_CONVERGENCE, it is required to be met.
    # forces and either energy change or displacement met, convergence!
    if op.Params.i_untampered and \
       ( op.Params.g_convergence == 'QCHEM' or op.Params.g_convergence == "MOLPRO" ) and \
       max_force < op.Params.conv_max_force and  \
       ( fabs(DE) < op.Params.conv_max_DE or max_disp < op.Params.conv_max_disp):
           return True

    # if max/rms forces/disp met or flat potential forces met, convergence!
    if op.Params.i_untampered and \
       ( op.Params.g_convergence == "GAU" or op.Params.g_convergence == "GAU_TIGHT" or
       op.Params.g_convergence == 'GAU_VERYTIGHT' or op.Params.g_convergence == 'GAU_LOOSE') and \
       ((max_force < op.Params.conv_max_force and \
       rms_force < op.Params.conv_rms_force and \
       max_disp  < op.Params.conv_max_disp  and \
       rms_disp  < op.Params.conv_rms_disp) or \
       100 * rms_force < op.Params.conv_rms_force):
           return True

    # if criterion not active or criterion met, convergence!
    if (not op.Params.i_max_DE    or fabs(DE)  < op.Params.conv_max_DE) and \
       (not op.Params.i_max_force or max_force < op.Params.conv_max_force) and \
       (not op.Params.i_rms_force or rms_force < op.Params.conv_rms_force) and \
       (not op.Params.i_max_disp  or max_disp  < op.Params.conv_max_disp) and \
       (not op.Params.i_rms_disp  or rms_disp  < op.Params.conv_rms_disp) :
           return True

    if op.Params.opt_type == 'IRC' and \
       (not op.Params.i_max_DE   or fabs(DE) < op.Params.conv_max_DE)   and \
       (not op.Params.i_max_disp or max_disp < op.Params.conv_max_disp) and \
       (not op.Params.i_rms_disp or rms_disp < op.Params.conv_rms_disp):
           return True
  
    return False

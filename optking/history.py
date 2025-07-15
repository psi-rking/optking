import copy
import logging
import math
from typing import Union

import numpy as np
import qcelemental as qcel

from .exceptions import OptError
from .molsys import Molsys
from .linearAlgebra import abs_max, rms, sign_of_double
from .printTools import print_array_string, print_mat_string
from . import log_name
from . import op

logger = logging.getLogger(f"{log_name}{__name__}")


class Step(object):
    def __init__(self, geom, E, forces, cart_grad):
        self.geom = geom.copy()  # Store as 2D object
        self.E = E
        self.forces = forces.copy()
        self.cart_grad = cart_grad.copy()
        self.projectedDE = None
        self.Dq = np.array([])
        self.followedUnitVector = np.array([])
        self.oneDgradient: Union[float, None] = None
        self.oneDhessian: Union[float, None] = None
        self.hessian: Union[np.ndarray, None] = None
        self.decent = True

    def record(self, projectedDE, Dq, followedUnitVector, oneDgradient, oneDhessian):
        self.projectedDE = projectedDE
        self.Dq = Dq.copy()
        self.followedUnitVector = followedUnitVector.copy()
        self.oneDgradient = oneDgradient
        self.oneDhessian = oneDhessian

    def to_dict(self):
        d = {
            "geom": self.geom.copy(),
            "E": self.E,
            "forces": self.forces.copy(),
            "cart_grad": self.cart_grad.copy(),
            "projectedDE": self.projectedDE,
            "Dq": self.Dq.copy(),
            "followedUnitVector": self.followedUnitVector.copy(),
            "oneDgradient": self.oneDgradient,
            "oneDhessian": self.oneDhessian,
            "decent": self.decent,
        }
        return d

    @staticmethod
    def from_dict(d):
        if all(["geom" in d.keys(), "E" in d.keys(), "forces" in d.keys()]):
            s = Step(d["geom"], d["E"], d["forces"], d["cart_grad"])
        else:
            raise OptError("Missing necessary keywords to construct step")

        s.record(
            d.get("projectedDE", None),
            d.get("Dq", np.array([])),
            d.get("followedUnitVector", np.array([])),
            d.get("oneDgradient", None),
            d.get("oneDhessian", None),
        )
        return s

    def __str__(self):
        s = "Step Info\n"
        s += "Geometry     = \n"
        s += print_mat_string(self.geom)
        s += "Energy       = %15.10f\n" % self.E
        s += "forces       = "
        s += print_array_string(self.forces)
        if self.projectedDE is not None:
            s += "Projected DE = %15.10f\n" % self.projectedDE
        if len(self.Dq):
            s += "Dq           = "
            s += print_array_string(self.Dq)
        if len(self.followedUnitVector):
            s += "followedUnitVector       = "
            s += print_array_string(self.followedUnitVector)
        if self.oneDgradient is not None:
            s += "oneDgradient = %15.10f\n" % self.oneDgradient
        if self.oneDhessian is not None:
            s += "oneDhessian  = %15.10f\n" % self.oneDhessian
        return s


class History(object):
    def __init__(self, params=None):
        self.steps = []
        History.stepsSinceLastHessian = 0

        if params is None:
            params = op.OptParams()

        self.steps_since_last_hessian = 0
        self.consecutive_backsteps = 0
        # nuclear_repulsion_energy = 0

        self.hess_update = params.hess_update
        self.hess_update_use_last = params.hess_update_use_last
        self.hess_update_dq_tol = params.hess_update_dq_tol
        self.hess_update_den_tol = params.hess_update_den_tol
        self.hess_update_limit = params.hess_update_limit
        self.hess_update_limit_max = params.hess_update_limit_max
        self.hess_update_limit_scale = params.hess_update_limit_scale

    def __str__(self):
        s = "History of length %d\n" % len(self)
        for i, step in enumerate(self.steps):
            s += "Step %d\n" % (i + 1)
            s += step.__str__()
        return s

    def __len__(self):
        return len(self.steps)

    def __getitem__(self, index):
        return self.steps[index]

    def __setitem__(self, index, item):
        self.steps[index] = item

    def __delitem__(self, index):
        del self.steps[index]

    # Add new step.  We will store geometry as 1D in history.
    def append(self, geom, E, forces, cart_grad):
        """Create a new step geometry should be stored as a one D object

        Parameters
        ----------
        geom: np.ndarray
        E: float
        forces: np.ndarray
        cart_grad: np.ndarray
        """
        s = Step(geom, E, forces, cart_grad)
        self.steps.append(s)
        self.steps_since_last_hessian += 1

    # Fill in details of new step.
    def append_record(self, projectedDE, Dq, followedUnitVector, oneDgradient, oneDhessian):
        self.steps[-1].record(projectedDE, Dq, followedUnitVector, oneDgradient, oneDhessian)

    def to_dict(self):
        initial = self.__dict__
        d = {
            "steps_since_last_hessian": initial.pop("steps_since_last_hessian"),
            "consecutive_backsteps": initial.pop("consecutive_backsteps"),
            "steps": [s.to_dict() for s in initial.pop("steps")],
            "options": initial,
        }  # place everything else in options
        return d

    @classmethod
    def from_dict(cls, d):
        params = op.OptParams(**d.get("options", {}))
        new_history = cls(params)

        new_history.steps_since_last_hessian = d.get("steps_since_last_hessian", 0)
        new_history.consecutive_backsteps = d.get("consecutive_backsteps", 0)
        new_history.steps = [Step.from_dict(s) for s in d.get("steps", [])]

        return new_history

    def trajectory(self, Zs):
        t = []
        Zstring = [qcel.periodictable.to_E(i) for i in Zs]
        for iS, S in enumerate(self.steps):
            t.append((S.E, list(Zstring), S.geom.copy()))
        return t

    # Summarize key quantities and return steps or string
    def summary(self, printoption=False):
        opt_summary = ""
        steps = []

        for i, step in enumerate(self.steps):
            if i == 0:
                DE = step.E
            else:
                DE = step.E - self.steps[i - 1].E

            try:
                max_force = abs_max(step.forces)
                rms_force = rms(step.forces)
            except ValueError:
                max_force = None
                rms_force = None

            # For the summary Dq, we do not want to +2*pi for example for the angles,
            # so we read old Dq used during step.
            max_disp = abs_max(step.Dq)
            rms_disp = rms(step.Dq)

            steps.append(
                {
                    "Energy": step.E,
                    "DE": DE,
                    "max_force": max_force,
                    "max_disp": max_disp,
                    "rms_disp": rms_disp,
                }
            )

            if max_force is None or rms_force is None:
                opt_summary += "\t  %4d %20.12lf  %18.12f    %12s    %12s    %12.8lf    %12.8lf" "  ~\n" % (
                    (i + 1),
                    self.steps[i].E,
                    DE,
                    "o",
                    "o",
                    max_disp,
                    rms_disp,
                )
            else:
                opt_summary += "\t  %4d %20.12lf  %18.12lf    %12.8lf    %12.8lf    %12.8lf    %12.8lf" "  ~\n" % (
                    (i + 1),
                    self.steps[i].E,
                    DE,
                    max_force,
                    rms_force,
                    max_disp,
                    rms_disp,
                )

        opt_summary += "\t" + "-" * 112 + "\n\n"

        if printoption:
            return opt_summary
        else:
            return steps

    # Keep only most recent step
    def reset_to_most_recent(self):
        self.steps = self.steps[-1:]
        History.stepsSinceLastHessian = 0
        self.consecutive_backsteps = 0
        # self.nuclear_repulsion_energy = 0
        # The step included is not taken in an IRC.
        self.steps[-1].projectedDE = None
        return

    # Use History to update Hessian
    def hessian_update(self, H, f_q, molsys):

        if self.hess_update == "NONE" or len(self.steps) < 1:
            return H

        logger.info("\tPerforming %s update." % self.hess_update)
        Nintco = molsys.num_intcos  # working dimension

        q = molsys.q_array()

        # Fix configuration of torsions and out-of-plane angles,
        # so that Dq's are reasonable
        molsys.update_dihedral_orientations()

        # Don't go further back than the last Hessian calculation
        num_to_use = min(self.hess_update_use_last, len(self.steps), self.steps_since_last_hessian)
        logger.info("\tUsing %d previous steps for update.", num_to_use)

        # Make list of old geometries to update with.
        # Check each one to see if it is too close (so stable denominators).
        use_steps = []
        i_step = len(self.steps) - 1  # just in case called with only 1 pt.
        while i_step > -1 and len(use_steps) < num_to_use:
            step = self.steps[i_step]
            dq, dg, dqdg, dqdq, max_change = self.get_update_info(molsys, f_q, q, step)

            # If there is only one left, take it no matter what.
            if len(use_steps) == 0 and i_step == 0:
                use_steps.append(i_step)
            elif math.fabs(dqdg) < self.hess_update_den_tol or math.fabs(dqdq) < self.hess_update_den_tol:
                logger.warning("\tDenominators (dg)(dq) or (dq)(dq) are very small.")
                logger.warning("\tSkipping Hessian update for step %d.", i_step + 1)
                pass
            elif max_change > self.hess_update_dq_tol:
                logger.warning(
                    "\tChange in internal coordinate of %5.2e exceeds limit of %5.2e.",
                    max_change,
                    self.hess_update_dq_tol,
                )
                logger.warning("\tSkipping Hessian update for step %d.", i_step + 1)
                pass
            else:
                use_steps.append(i_step)
            i_step -= 1

        hessian_steps = "\tSteps to be used in Hessian update: "
        for i in use_steps:
            hessian_steps += " %d" % (i + 1)
        hessian_steps += "\n"

        logger.info(hessian_steps)

        # Don't update any modes if constraints are enacted.
        frozen = molsys.frozen_intco_list
        ranged = molsys.ranged_intco_list
        constrained = frozen + ranged
        C = np.diagflat(constrained)

        H_new = np.zeros(H.shape)
        for i_step in use_steps:
            step = self.steps[i_step]
            dq, dg, dqdg, dqdq, max_change = self.get_update_info(molsys, f_q, q, step)

            # See  J. M. Bofill, J. Comp. Chem., Vol. 15, pages 1-11 (1994)
            #  and Helgaker, JCP 2002 for formula.
            if self.hess_update == "BFGS":
                for i in range(Nintco):
                    for j in range(Nintco):
                        H_new[i, j] = H[i, j] + dg[i] * dg[j] / dqdg

                Hdq = np.dot(H, dq)
                dqHdq = np.dot(dq, Hdq)

                for i in range(Nintco):
                    for j in range(Nintco):
                        H_new[i, j] -= Hdq[i] * Hdq[j] / dqHdq

            elif self.hess_update == "MS":
                Z = -1.0 * np.dot(H, dq) + dg
                qz = np.dot(dq, Z)

                for i in range(Nintco):
                    for j in range(Nintco):
                        H_new[i, j] = H[i, j] + Z[i] * Z[j] / qz

            elif self.hess_update == "POWELL":
                Z = -1.0 * np.dot(H, dq) + dg
                qz = np.dot(dq, Z)

                for i in range(Nintco):
                    for j in range(Nintco):
                        H_new[i, j] = (
                            H[i, j] - qz / (dqdq * dqdq) * dq[i] * dq[j] + (Z[i] * dq[j] + dq[i] * Z[j]) / dqdq
                        )

            elif self.hess_update == "BOFILL":
                # Bofill = (1-phi) * MS + phi * Powell
                Z = -1.0 * np.dot(H, dq) + dg
                qz = np.dot(dq, Z)
                zz = np.dot(Z, Z)

                phi = 1.0 - qz * qz / (dqdq * zz)
                if phi < 0.0:
                    phi = 0.0
                elif phi > 1.0:
                    phi = 1.0

                for i in range(Nintco):  # (1-phi)*MS
                    for j in range(Nintco):
                        H_new[i, j] = H[i, j] + (1.0 - phi) * Z[i] * Z[j] / qz

                for i in range(Nintco):  # (phi * Powell)
                    for j in range(Nintco):
                        H_new[i, j] += phi * (
                            -1.0 * qz / (dqdq * dqdq) * dq[i] * dq[j] + (Z[i] * dq[j] + dq[i] * Z[j]) / dqdq
                        )

            # If the cooordinate is constrained. Don't allow the update to occur.
            for i in range(Nintco):
                if C[i, i] == 1:
                    H_new[i, :] = H_new[:, i] = np.zeros(len(f_q))
                    H_new[i, i] = H[i, i]

            if self.hess_update_limit:  # limit changes in H
                # Changes to the Hessian from the update scheme are limited to the larger of
                # (hess_update_limit_scale)*(the previous value) and hess_update_limit_max.
                max_limit = self.hess_update_limit_max
                scale_limit = self.hess_update_limit_scale

                # Compute change in Hessian
                H_new[:, :] = H_new - H

                for i in range(Nintco):
                    for j in range(Nintco):
                        val = math.fabs(scale_limit * H[i, j])
                        maximum = max(val, max_limit)

                        if math.fabs(H_new[i, j]) < maximum:
                            H[i, j] += H_new[i, j]
                        else:  # limit change to max
                            H[i, j] += maximum * sign_of_double(H_new[i, j])

            else:  # only copy H_new into H
                H[:, :] = H_new

            H_new[:, :] = 0  # zero for next step
            # end loop over old geometries

        logger.info("\tUpdated Hessian (in au) \n %s" % print_mat_string(H))
        return H

    def get_update_info(self, molsys: Molsys, f: np.ndarray, q: np.ndarray, step: Step):
        """Get gradient and displacement info for updating the Hessian

        Parameters
        ----------
        molsys: molsys.Molsys]
        f: np.ndarray
        q: np.ndarray
        step: Step

        """
        f_old = step.forces
        x_old = step.geom

        old_molsys = copy.deepcopy(molsys)
        old_molsys.geom = x_old
        q_old = old_molsys.q_array()

        dq = q - q_old
        dg = f_old - f  # gradients -- not forces!
        dqdg = np.dot(dq, dg)
        dqdq = np.dot(dq, dq)
        max_change = abs_max(dq)
        return dq, dg, dqdg, dqdq, max_change

    def summary_string(self):
        output_string = """\n\t==> Optimization Summary <==\n
        \n\tMeasures of convergence in internal coordinates in au. (Any backward steps not shown.)
        \n\t---------------------------------------------------------------------------------------------------------------  ~
        \n\t Step         Total Energy             Delta E       MAX Force       RMS Force        MAX Disp        RMS Disp   ~
        \n\t---------------------------------------------------------------------------------------------------------------  ~
        \n"""
        output_string += self.summary(printoption=True)
        return output_string


# oHistory = History()

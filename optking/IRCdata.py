import logging
import os
import copy

import numpy as np

from .exceptions import OptError
from .printTools import print_geom_string
from .linearAlgebra import symm_mat_inv
from . import log_name
from . import addIntcos
from . import molsys
from . import frag

logger = logging.getLogger(f"{log_name}{__name__}")


class IRCpoint(object):
    """Holds data for one step on the IRC.
    Parameters
    ----------
    step_number : int
        step number
    q_pivot :    ndarray
        pivot point for next step
    x_pivot :    ndarray
        pivot point for next step; save so that q_pivot can be recomputed if desired
    q       :    ndarray
        internal coordinate values
    x       :    ndarray
        cartesian coordinate values
    f_q     :    ndarray
        internal coordinate forces
    f_x     :    ndarray
        cartesian coordinate forces
    energy  :    float
        total energy
    step_dist :  float
    arc_dist  :  float
    line_dist :  float
    """

    def __init__(
        self,
        step_number,
        q,
        x,
        f_q,
        f_x,
        energy,
        q_pivot,
        x_pivot,
        step_dist,
        arc_dist,
        line_dist,
    ):
        self.step_number = step_number
        self.q = q
        self.x = x
        self.f_q = f_q
        self.f_x = f_x
        self.energy = energy
        self.q_pivot = q_pivot
        self.x_pivot = x_pivot
        self.step_dist = step_dist
        self.arc_dist = arc_dist
        self.line_dist = line_dist

    def add_pivot(self, q_p, x_p):
        self.q_pivot = q_p
        self.x_pivot = x_p

    def to_dict(self):
        s = {
            "step_number": self.step_number,
            "q": self.q.tolist(),
            "x": self.x.tolist(),
            "f_q": self.f_q.tolist(),
            "f_x": self.f_x.tolist(),
            "energy": self.energy,
            "q_pivot": self.q_pivot.tolist() if self.q_pivot is not None else [0] * len(self.q),
            "x_pivot": self.x_pivot.tolist() if self.x_pivot is not None else [0] * len(self.x),
            "step_dist": self.step_dist,
            "arc_dist": self.arc_dist,
            "line_dist": self.line_dist,
        }
        return s

    @classmethod
    def from_dict(cls, d):
        for key in ["q", "x", "f_q", "f_x", "q_pivot", "x_pivot"]:
            d[key] = np.asarray(d[key])
        return cls(**d)


class IRCHistory(object):
    """Stores obtained points along the IRC as well as information about
    the status of the IRC computation"""

    __step_size = 0.0
    __direction = None
    __running_step_dist = 0.0
    __running_arc_dist = 0.0
    __running_line_dist = 0.0

    def __init__(self):
        self.go = True
        self.irc_points: [IRCpoint] = []
        self.atom_symbols = None

    def set_atom_symbols(self, atom_symbols):  # just for printing
        self.atom_symbols = atom_symbols.copy()  # just for printing

    def set_step_size_and_direction(self, step_size, direction):
        self.__step_size = step_size
        self.__direction = direction

    def to_dict(self):
        d = {
            "irc_points": [point.to_dict() for point in self.irc_points],
            "go": self.go,
            "atom_symbols": self.atom_symbols,
            "direction": self.__direction,
            "step_size": self.__step_size,
        }
        return d

    @classmethod
    def from_dict(cls, d):
        irc_history = cls()
        irc_history.irc_points = [IRCpoint.from_dict(point) for point in d["irc_points"]]
        irc_history.go = d["go"]
        irc_history.atom_symbols = d["atom_symbols"]
        irc_history.__direction = d["direction"]
        irc_history.__step_size = d["step_size"]
        return irc_history

    def add_irc_point(self, step_number, q_in, x_in, f_q, f_x, E, lineDistStep=0, arcDistStep=0):
        if len(self.irc_points) != 0:
            if self.__direction == "FORWARD":
                sign = 1
            elif self.__direction == "BACKWARD":
                sign = -1
                step_number *= -1
            else:
                raise OptError("IRC direction must be set to FORWARD or BACKWARD")
            # step_dist = sum of all steps to and from pivot points, a multiple of
            # the step_size
            self.__running_step_dist += sign * self.__step_size
            # line distance is sum of all steps directly between rxnpath points, ignoring
            # pivot points
            self.__running_line_dist += sign * lineDistStep
            # distance along a circular arc connecting rxnpath points
            self.__running_arc_dist += sign * arcDistStep

        onepoint = IRCpoint(
            step_number,
            q_in,
            x_in,
            f_q,
            f_x,
            E,
            None,
            None,
            self.__running_step_dist,
            self.__running_arc_dist,
            self.__running_line_dist,
        )
        self.irc_points.append(onepoint)

        pindex = len(self.irc_points) - 1
        outstr = "\tAdding IRC point %d\n" % pindex
        outstr += print_geom_string(self.atom_symbols, x_in, "Angstroms")
        logger.info(outstr)

    def add_pivot_point(self, q_p, x_p, step=None):
        index = -1 if step is None else step
        pindex = (len(self.irc_points) - 1) if step is None else step
        logger.debug(
            "Adding pivot point (index %d) for finding rxnpath point %d" % (pindex, pindex + 1)
        )
        self.irc_points[index].add_pivot(q_p, x_p)

    # Return most recent IRC step data unless otherwise specified
    def step_number(self, step=None):
        index = -1 if step is None else step
        return self.irc_points[index].step_number

    @property
    def step_size(self):
        return self.__step_size

    def current_step_number(self):
        return len(self.irc_points)

    def q_pivot(self, step=None):
        index = -1 if step is None else step
        return self.irc_points[index].q_pivot

    def x_pivot(self, step=None):
        index = -1 if step is None else step
        return self.irc_points[index].x_pivot

    def q(self, step=None):
        index = -1 if step is None else step
        return self.irc_points[index].q

    def x(self, step=None):
        index = -1 if step is None else step
        return self.irc_points[index].x

    def f_q(self, step=None):
        index = -1 if step is None else step
        return self.irc_points[index].f_q

    def f_x(self, step=None):
        index = -1 if step is None else step
        return self.irc_points[index].f_x

    def energy(self, step=None):
        index = -1 if step is None else step
        return self.irc_points[index].energy

    def line_dist(self, step=None):
        index = -1 if step is None else step
        return self.irc_points[index].line_dist

    def arc_dist(self, step=None):
        index = -1 if step is None else step
        return self.irc_points[index].arc_dist

    def step_dist(self, step=None):
        index = -1 if step is None else step
        return self.irc_points[index].step_dist

    def test_for_dissociation(
        self, new_molsys: molsys.Molsys, old_molsys: molsys.Molsys, threshold=0.4
    ):
        """Check whether or not the connectivity has changed and multiple fragments have
        been created. This may be used to terminate the IRC. This method should only be
        called if frag_mode == 'SINGLE'.

        Returns
        -------
        bool
            True if the molecular system has separated into 2 (or more) fragments from 1
        """

        logger.debug("Checking connectivity for whether dissociation has occured")

        # Repeat the standard procedure for creating a single fragment molecular system
        orig_connectivity = addIntcos.connectivity_from_distances(old_molsys.geom, old_molsys.Z)
        orig_molsys = copy.deepcopy(old_molsys)
        orig_molsys.split_fragments_by_connectivity()
        scale_dist = orig_molsys.augment_connectivity_to_single_fragment(orig_connectivity)
        orig_molsys.consolidate_fragments()

        # Detect the number of fragments current molecular system using the old scale_dist plus
        # 0.4 angstroms. Not bullet proof, just attempts to detect changes in conenctivity while
        # not triggering for small increases in bond lengths
        new_molsys = molsys.Molsys([frag.Frag(old_molsys.Z, new_molsys.geom, old_molsys.masses)])
        new_molsys.split_fragments_by_connectivity(scale_dist + threshold)

        if orig_molsys.nfragments != new_molsys.nfragments:
            return True
        return False

    def test_for_irc_minimum(self, f_q, energy, irc_conv):
        """Given current forces, checks if we are at/near a minimum

        Parameters
        ----------
        f_q: np.ndarray
            (1, nintco) array of forces in internal coordinates evaluated at the geometry currently
            being evaulated for convergence.
        energy: float
            energy for the point at the geometry currently being evaluated for convergence
        irc_conv: float
            Threshold to compare the overlap between the previous and current forces along the
            reaction path.

        Notes
        -----
        Two checks are performed.
        1. If forces are opposite those are previous pivot point
            - The forces point in opposite directions due to stepping over the minima
            due to finite step size

        At the time this function is called, the point/geometry is assumed to not have been added
        to the history. Calling ``arc_dist()``, ``energy()``, ``f_q()``, etc... should return the
        last converged point along the IRC. This function should only be called after hypersphere
        optimization has been completed.
        """

        unit_f = f_q / np.linalg.norm(f_q)  # current forces (at guess_point)
        f_rxn = self.f_q(step=-2)  # forces at most recent rxnpath point (q0)
        unit_f_rxn = f_rxn / np.linalg.norm(f_rxn)
        overlap = np.dot(unit_f, unit_f_rxn)

        logger.info("Overlap of forces with previous rxnpath point %8.4f" % overlap)
        d_energy = energy - self.energy(step=-2)
        logger.info("Change in energy from last point %.4e", d_energy)

        if overlap < irc_conv:
            return True
        if abs(self.line_dist(step=-1) - self.line_dist(step=-2)) < self.step_size * 1e-3:
            return True
        # AH Found to cause early termination in some test cases
        # elif overlap < 0.0 and d_energy > 0.0 :
        #     return True

        return False

    def progress_report(self, return_str=False):
        blocks = 4  # TODO: make dynamic
        sign = 1
        Ncoord = len(self.q())

        irc_report = os.path.join(os.getcwd(), "ircprogress.log")  # prepare progress report
        with open(irc_report, "w+") as irc_prog:
            irc_prog.truncate(0)

        irc_log = logger.getChild("ircprogress")
        irc_handle = logging.FileHandler(os.path.join(os.getcwd(), "ircprogress.log"), "w")
        irc_handle.setLevel(logging.DEBUG)
        irc_log.addHandler(irc_handle)

        out = "\n"
        out += "@IRC ----------------------------------------------\n"
        out += "@IRC            ****      IRC Report      ****\n"
        out += "@IRC ----------------------------------------------\n"
        out += "@IRC  Step    Energy              Change in Energy \n"
        out += "@IRC ----------------------------------------------\n"
        for i in range(len(self.irc_points)):
            if i == 0:
                DE = self.energy(i)
            else:
                DE = self.energy(i) - self.energy(i - 1)
            out += "@IRC  %3d %18.12lf  %18.12lf\n" % (i, self.energy(i), DE)
        out += "@IRC ----------------------------------------------\n\n"

        # Print Internal Coordinates for Each step
        out += "@IRC -----------------------------------------------------\n"
        out += "@IRC              ****     IRC Steps     ****             \n"
        out += "@IRC -----------------------------------------------------"
        for j in range(Ncoord // blocks):
            out += "\n@IRC        |          Distance         |\n"
            out += "@IRC Step   | Step    Arc       Line    |"
            for i in range(j * blocks, (j + 1) * blocks):
                out += "    Coord %3d" % i
            out += "\n"
            out += "@IRC --------------------------------------"
            for i in range(j * blocks, (j + 1) * blocks):
                out += "-------------"
            out += "\n"
            for i in range(len(self.irc_points)):
                out += "@IRC  %3d %9.2lf %9.5lf  %9.5lf   " % (
                    i,
                    sign * self.step_dist(i),
                    sign * self.arc_dist(i),
                    sign * self.line_dist(i),
                )
                for k in range(j * blocks, (j + 1) * blocks):
                    out += "%13.8f" % self.q(i)[k]
                out += "\n"

            out += "@IRC --------------------------------------"
            for i in range(j * blocks, (j + 1) * blocks):
                out += "-------------"
        if Ncoord % blocks != 0:
            out += "\n@IRC         |          Distance         |\n"
            out += "@IRC  Step   | Step    Arc       Line    |"

            for i in range(Ncoord - (Ncoord % blocks), Ncoord):
                out += "    Coord %3d" % i
            out += "\n"
            out += "@IRC --------------------------------------"

            for i in range(Ncoord - (Ncoord % blocks), Ncoord):
                out += "-------------"
            out += "\n"

            for i in range(len(self.irc_points)):
                out += "@IRC  %3d %9.2lf %9.5lf  %9.5lf   " % (
                    i,
                    sign * self.step_dist(i),
                    sign * self.arc_dist(i),
                    sign * self.line_dist(i),
                )
                for k in range(Ncoord - (Ncoord % blocks), Ncoord):
                    out += "%13.8f" % self.q(i)[k]
                out += "\n"

            out += "@IRC --------------------------------------"

            for i in range(Ncoord - (Ncoord % blocks), Ncoord):
                out += "-------------"

        out += "\n"
        out += "\n"
        if return_str:
            return out
        irc_log.info(out)
        irc_handle.close()
        # out += mol.print_coords(psi_outfile, qc_outfile)
        # out += mol.print_simples(psi_outfile, qc_outfile)

    def rxnpath_dict(self):
        rp = [self.irc_points[i].to_dict() for i in range(len(self.irc_points))]
        return rp

    def _project_forces(self, f_q, o_molsys, threshold=1e-10):
        """Compute forces perpendicular to the second IRC halfstep and tangent to hypersphere

        Notes
        -----

        For IRC calculations the Gradient perpendicular to p and tangent to the hypersphere is:
        g_m' = g_m - (g_m^t . p_m / (p_m^t . p_m) * p_m, in massweighted coordinates
        or g'   = g   - (g^t . p / (p^t G^-1 p)) * G^-1 . p

        """

        G_m = o_molsys.Gmat(massWeight=True)
        G_m_inv = symm_mat_inv(G_m, redundant=True, threshold=threshold)

        q_vec = o_molsys.q_array()
        p_vec = q_vec - self.q_pivot()

        G_m_inv_p = G_m_inv @ p_vec
        orthog_f = f_q - (f_q @ p_vec) / (p_vec @ G_m_inv_p) * G_m_inv_p
        return orthog_f

    def recompute_all_internals(self, molsys: molsys.Molsys):
        # make a independent copy of molecular system to change all internal coordinate values
        # into the new internal basis set
        tmp_molsys = copy.deepcopy(molsys)

        # For each step update the irc_point coords, forces coords, and pivot points coords
        for step in self.irc_points:
            tmp_molsys.geom = step.x
            step.q = tmp_molsys.q_array()
            step.f_q = tmp_molsys.gradient_to_internals(step.f_x)

            # Last point may not be fully set (could be none)
            if isinstance(step.x_pivot, np.ndarray):
                tmp_molsys.geom = step.x_pivot
                step.q_pivot = tmp_molsys.q_array()

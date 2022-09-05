import logging
from abc import abstractmethod
from math import isclose
from typing import Union

import numpy as np

from .addIntcos import linear_bend_check
from .displace import displace_molsys
from .exceptions import AlgError, OptError
from .history import Step, History
from .stepAlgorithms import OptimizationInterface
from . import log_name

logger = logging.getLogger(f"{log_name}{__name__}")


class LineSearchStep(Step):
    """Extension of history.Step """

    def __init__(self, geom, energy, forces, distance, next_pt_dist):
        super().__init__(geom, energy, forces, np.zeros(geom.shape))
        self.distance = distance
        self.next_pt_dist = next_pt_dist


class LineSearch(OptimizationInterface):
    """Basic framework for performing Linesearches. Child classes must implement a fit method
    that determines the step_size from the previous point in the linesearch to either the new point
    in the linesearch or to the predicted minimum of the linesearch."""

    # Developer Note: step_size always refers to the length of the step from the previous point to the next point
    # distance (as in the Point class) refers to the distance from the original point in the line search.

    def __init__(self, molsys, history, params):
        super().__init__(molsys, history, params)

        self.linesearch_max_iter = 10
        self.linesearch_start = len(self.history.steps)
        self.linesearch_steps = 0
        self.points = []  # list of Points
        self.points_needed: Union[float, None] = None
        self.final_step = np.zeros(len(self.molsys.q_array()))
        self.final_distance = 0
        self.minimized = False
        self.direction: Union[np.ndarray, None] = None
        self.linesearch_history = History()
        self.active_point = None

        # # should have just taken a step. Continue in this direction
        # # stash the initial Hessian for use at end of linesearch
        # if len(self.history.steps) > 1:
        #     dq = self.history.steps[-1].Dq

        #     if H is not None:
        #         self.history.steps[-1].H = H
        # else:
        #     dq = np.ones(len(self.molsys.q_array()))

        # self.direction = np.linalg.norm(dq)
        self.step_size = params.linesearch_step
        if params.linesearch_step is None:
            self.step_size = np.linalg.norm(self.history[-2].Dq) / 2

    @abstractmethod
    def fit(self):
        """Determines where the next step should head. Remove points from self.points as needed
        Add the new points. Must set self.final_point if the linesearch has finished.

        Returns
        -------
        step_size: float
            length of step (could be negative) along dq to the next point from the last point
        converged: bool
            has fit found a minimum.
        """
        pass

    @abstractmethod
    def step(self, fq=None, energy=None, **kwargs):
        """Either take a step with the size dictated by the fit method. or take another step
        of the default size"""
        pass

    @abstractmethod
    def expected_energy(self, **kwargs):
        """Linesearch Algorithms should be able to compute the expected energy based only
        on the Points."""
        pass

    def take_step(self, fq=None, H=None, energy=None, return_str=False, **kwargs):

        if self.linesearch_steps < 10:
            dq, self.step_size = self.step(fq, energy, **kwargs)
            self.linesearch_steps += 1
        else:
            raise AlgError("Line search did not converge to a solution")

        if len(self.linesearch_history.steps) > 1:
            delta_energy = self.linesearch_history.steps[-1].E - self.linesearch_history.steps[0].E
            logger.debug("\tProjected energy change: %10.10lf\n" % delta_energy)
        else:
            delta_energy = 0

        self.molsys.interfrag_dq_discontinuity_correction(dq)
        achieved_dq, achieved_dx, return_str = displace_molsys(self.molsys, dq, fq, return_str=True)
        achieved_dq_norm = np.linalg.norm(achieved_dq)
        logger.info("\tNorm of achieved step-size %15.10f" % achieved_dq_norm)

        self.linesearch_history.append_record(delta_energy, achieved_dq, self.direction, None, None)

        linear_list = linear_bend_check(self.molsys, achieved_dq)
        if linear_list:
            raise AlgError("New linear angles", newLinearBends=linear_list)

        if not isclose(np.linalg.norm(dq), achieved_dq_norm, rel_tol=5, abs_tol=0):
            # Attempt to replicate step_size check in OptimizationAlgorithm
            # TODO create a check in displace for backtransformation failure
            raise OptError("Back transformation has failed spectacularly. Smaller step needed")

        if return_str:
            return achieved_dq, return_str
        return achieved_dq

    def reset(self):
        self.previous_step = self.compute_distance() * self.direction
        self.direction = np.zeros(len(self.direction))
        self.linesearch_steps = 0
        self.linesearch_start = 0
        self.linesearch_history = None
        self.step_size = self.final_distance / 2
        self.points = []

    def start(self, dq):
        logger.info("Starting linesearch in direction %s", dq)
        self.direction = dq / np.linalg.norm(dq)
        self.linesearch_start = len(self.history.steps)
        self.linesearch_history = History()
        self.minimized = False

    def compute_distance(self):
        if len(self.points) == 3:
            active_point = self.points[1] if self.points[-1] is None else self.points[0]
        else:
            active_point = self.points[-1]
        self.active_point = active_point  # save in case this is the final step to minima
        return active_point.distance + active_point.next_pt_dist


class ThreePointEnergy(LineSearch):
    def __init__(self, molsys, history, params):
        super().__init__(molsys, history, params)
        self.points_needed = 3
        self.points = []
        self.expected_energy = 0

    def step(self, fq=None, energy=None, **kwargs):
        """ Determine the new step to take in the linesearch procedure. (Could be stepping backwards
        from the previous point).

        Parameters
        ----------
        fq: np.ndarary
        energy: float
        kwargs

        Returns
        -------
        np.ndarray: new step

        """
        logger.info("\n\tTaking LINESEARCH optimization step.")

        distance = self.compute_distance()
        logger.debug("Adding new step at distance %s", distance)
        new_step = LineSearchStep(self.molsys.geom, energy, fq, distance, next_pt_dist=self.step_size)
        self.linesearch_history.steps.append(new_step)

        if self.linesearch_steps < self.points_needed:
            logger.debug("Taking one of initial set of steps")
            self.points.append(new_step)

        if len(self.points) == self.points_needed:
            if None in self.points:
                self.points[self.points.index(None)] = new_step
            self.step_size, self.minimized = self.fit()
            logger.info("Taking a step of length: %f along\n %s", self.step_size, self.direction)

        return self.step_size * self.direction, self.step_size

    def requires(self):
        return "energy"

    def expected_energy(self, **kwargs):
        return self.expected_energy

    def fit(self):
        """Three point parabolic fit. Returns the next point in linesearch.
        Returns
        -------
        step_size: float
            distance to the next point
        converged: boolean
            True if stepsize is distance to the projected minimum. False if linsearch goes on
        """

        converged = False
        energy_a, energy_b, energy_c = [point.E for point in self.points]

        sa = 0.0
        sb = self.points[1].distance
        sc = self.points[2].distance
        logger.info("\n\tCurrent linesearch bounds.\n")
        logger.info("\t s=%7.5f, Ea=%17.12f", 0, energy_a)
        logger.info("\t s=%7.5f, Eb=%17.12f", sb, energy_b)
        logger.info("\t s=%7.5f, Ec=%17.12f\n", sc, energy_c)

        if energy_b < energy_a and energy_b < energy_c:
            logger.debug("\tMiddle point is lowest energy. Good. Projecting minimum.")

            A = np.zeros((2, 2))
            A[0, 0] = sc * sc - sb * sb
            A[0, 1] = sc - sb
            A[1, 0] = sb * sb - sa * sa
            A[1, 1] = sb - sa

            B = np.zeros(2)
            B[0] = energy_c - energy_b
            B[1] = energy_b - energy_a

            x = np.linalg.solve(A, B)
            x_min = -x[1] / (2 * x[0])
            min_energy = x[0] * x_min ** 2 + x[1] * x_min + energy_a

            logger.info("Desired point %s", x_min)
            logger.info("Desired point %s", sb)
            logger.info("Desired point %s", sc)

            # active point corresponds to the old point (we just created a new one)
            if self.points.index(self.active_point) == 1:
                step_size = x_min - sc
            else:
                step_size = x_min - sb

            self.expected_energy = min_energy
            self.final_distance = x_min
            self.final_step = x_min * self.direction

            converged = True

        elif energy_c < energy_b and energy_c < energy_a:
            # unbounded.  increase step size
            # need to compute new Point 3 displacing from Point 2
            logger.debug("\tSearching with larger step beyond 3rd point.")
            step_size = sc
            self.points[1] = self.points[2]
            self.points[2] = None

            self.points[0].next_pt_dist = self.points[1].distance
            self.points[1].next_pt_dist = step_size

        else:
            # displace backwards from last_point 2
            # if the last step was a point 3 need to step back to sb and again to halfway point
            if self.linesearch_history.steps[-1].distance == sb:
                step_size = -1 * (sb / 2)
            else:
                step_size = -1 * (sb / 2 + (self.points[-1].distance - self.points[1].distance))

            self.points[2] = self.points[1]
            self.points[1] = None

            self.points[2].next_pt_dist = 0
            self.points[0].next_pt_dist = sb / 2

        return step_size, converged

    def reset(self):
        super().reset()
        self.points = []

    def compute_distance(self):
        if self.linesearch_steps == 0:
            return np.zeros(len(self.molsys.q_array()))
        else:
            return super().compute_distance()

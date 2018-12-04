### Class to store points on the IRC
import logging

class IRCpoint(object):
    """Holds data for one step on the IRC.
    Parameters
    ----------
    step_number : int
        step number
    q_pivot :    ndarray
        pivot point for step
    x_pivot :    ndarray
        pivot point for step; we save so that q_pivot can be recomputed if necessary
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

    def __init__(self, step_number, q_pivot, x_pivot, q, x, f_q, f_x, energy, step_dist, arc_dist, line_dist):
        self.step_number = step_number
        self.q_pivot = q_pivot
        self.x_pivot = x_pivot
        self.q = q
        self.x = x
        self.f_q = f_q
        self.f_x = f_x
        self.energy = energy
        self.step_dist = step_dist
        self.arc_dist = arc_dist
        self.line_dist = line_dist

class IRCdata(object):
    """ Stores obtained points along the IRC as well as information about
        the status of the IRC computation"""

    def __init__(self):
        self.go = True
        self.in_min_range = False
        self.step_length = 0.4 # TODO: fill with parameter value 
        self.sphere_step = 0
        self.running_arc_dist = 0.0
        self.running_line_dist = 0.0
        # If not too inconvenient, require as arguments in add_irc_point()
        self.arc_length = 0.0  # temporary storage for current step length
        self.line_length = 0.0  # temporary storage for current step length

        self.irc_points = []

    def add_irc_point(self, step_number, q_p, x_p, q, x, f_q, f_x, E):
        step_dist = step_number * self.step_length
        arc_dist  = self.running_arc_dist + self.arc_length # optionally computed
        line_dist = self.running_line_dist + self.line_length # optionally computed

        onepoint = IRCpoint(step_number, q_p, x_p, q, x, f_q, f_x, E,
                            step_dist, arc_dist, line_dist)
        self.irc_points.append(onepoint)

    # Return most recent IRC step data unless otherwise specified
    def step_number(self, step=None):
        index = -1 if step is None else step
        return self.irc_points[index].step_number

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

    def set_arc_length(self, arc_length_in, step=None):
        index = -1 if step is None else step
        self.irc_points[index].arc_length = arc_length_in 

    def set_line_length(self, line_length_in, step=None):
        index = -1 if step is None else step
        self.irc_points[index].line_length = line_length_in 

    def progress_report(self):
        blocks = 4 # TODO: make dynamic
        sign = 1
        # TODO: if OptParams.IRC_direction == 'backward': sign = -1
        Ncoord = len(self.q())

        logging.basicConfig(filename='ircprogress.log',level=logging.DEBUG)

        out = '\n' #string for output

        #Print Energies and Energy Changes for Each Step
        out += "@IRC ----------------------------------------------\n"
        out += "@IRC            ****      IRC Report      ****\n"
        out += "@IRC ----------------------------------------------\n"
        out += "@IRC  Step    Energy              Change in Energy \n"
        out += "@IRC ----------------------------------------------\n"
        for i in range(len(self.irc_points)):
            if i == 0:
                DE = self.energy(i)
            else:
                DE = self.energy(i) - self.energy(i-1)
            out += "@IRC  %3d %18.12lf  %18.12lf\n" % (i, self.energy(i), DE)
        out += "@IRC ----------------------------------------------\n\n"

        # Print Internal Coordinates for Each step
        out += "@IRC -----------------------------------------------------\n"
        out += "@IRC              ****     IRC Steps     ****             \n"
        out += "@IRC -----------------------------------------------------"
        for j in range(Ncoord//blocks):
            out += "\n@IRC        |          Distance         |\n"
            out +=   "@IRC Step   | Step    Arc       Line    |"
            for i in range(j*blocks, (j+1)* blocks):
                out += "    Coord %3d" % i
            out += "\n"
            out += "@IRC --------------------------------------"
            for i in range(j*blocks, (j+1)* blocks):
                out += "-------------"
            out += "\n"
            for i in range(len(self.irc_points)):
                out += "@IRC  %3d %9.2lf %9.5lf  %9.5lf   " % (i, sign*self.step_dist(i),
                    sign*self.arc_dist(i), sign*self.line_dist(i))
                for k in range(j*blocks, (j+1)*blocks):
                    out += "%13.8f" % self.q(i)[k]
                out += "\n"
          
            out += "@IRC --------------------------------------"
            for i in range(j*blocks, (j+1)* blocks):
                out += "-------------"
        if Ncoord % blocks != 0:
            out += "\n@IRC         |          Distance         |\n"
            out +=   "@IRC  Step   | Step    Arc       Line    |"

            for i in range(Ncoord - (Ncoord % blocks), Ncoord):
                out += "    Coord %3d" % i
            out += "\n"
            out += "@IRC --------------------------------------"

            for i in range(Ncoord - (Ncoord % blocks), Ncoord):
                out += "-------------"
            out += "\n"

            for i in range(len(self.irc_points)):
                out += "@IRC  %3d %9.2lf %9.5lf  %9.5lf   " % (i,
                    sign*self.step_dist(i), sign*self.arc_dist(i), sign*self.line_dist(i))
                for k in range(Ncoord - (Ncoord % blocks), Ncoord):
                    out += "%13.8f" % self.q(i)[k]
                out += "\n"

            out += "@IRC --------------------------------------"

            for i in range(Ncoord - (Ncoord % blocks), Ncoord):
                out += "-------------"
        
        out += "\n"
        out += "\n"

        #print(out,end="")
        logging.info(out)

        #out += mol.print_coords(psi_outfile, qc_outfile)
        #out += mol.print_simples(psi_outfile, qc_outfile)


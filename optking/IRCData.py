### Class to store points on the IRC

class IRCpoint(object):
    """Holds data for one step on the IRC.
    Parameters
    ----------
    coord_step : int
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

    def __init__(self, coord_step, q_pivot, x_pivot, q, x, f_q, f_x, energy, step_dist, arc_dist, line_dist):
        self.coord_step = coord_step
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
        self.step_length = 0.4 #fill with parameter value 

        self.sphere_step = 0
        #self.coord_step  needed?
        self.step_dist = 0.0
        self.arc_dist = 0.0
        self.line_dist = 0.0
        self.arc_length = 0.0
        self.line_length = 0.0

        self.irc_points = []

    # Return most recent IRC step data unless otherwise specified
    def q_pivot(self, step=None):
        index = -1 if step is None else step
        return self.irc_points[index].q_pivot

    def x_pivot(self, step=None):
        index = -1 if step is None else step
        return self.irc_points[index].x_pivot

    def g_q(self, step=None):
        index = -1 if step is None else step
        return self.irc_points[index].g_q

    def g_x(self, step=None):
        index = -1 if step is None else step
        return self.irc_points[index].q_x

    def f_q(self, step=None):
        index = -1 if step is None else step
        return self.irc_points[index].f_q

    def f_x(self, step=None):
        index = -1 if step is None else step
        return self.irc_points[index].f_x

    def line_dist(self, step=None):
        index = -1 if step is None else step
        return self.irc_points[index].line_dist

    def add_irc_point(self, coord_step, q_p, x_p, q, x, f_q, f_x, E):
        self.step_dist  = coord_step * self.step_length
        self.arc_dist  += self.arc_length
        self.line_dist += self.line_length

        onepoint = IRCpoint(coord_step, q_p, x_p, q, x, f_q, f_x, E,
                             self.step_dist, self.arc_dist, self.line_dist)
        self.irc_points.append(onepoint)

"""
    def progress_report(self):

  double DE;
  int dim = mol.Ncoord();
  int blocks = 4;
  int sign = 1;

  if(Opt_params.IRC_direction == OPT_PARAMS::BACKWARD)
    sign = -1;

//Printing Energies and Energy Changes for Each Step
  oprintf_out("@IRC ----------------------------------------------\n");
  oprintf_out("@IRC            ****      IRC Report      ****\n");
  oprintf_out("@IRC ----------------------------------------------\n");
  oprintf_out("@IRC  Step    Energy              Change in Energy \n");
  oprintf_out("@IRC ----------------------------------------------\n");
  for (std::size_t i=0; i<steps.size(); ++i)
  {
    if (i == 0) DE = g_step(i).g_energy();
    else DE = g_step(i).g_energy() - g_step(i-1).g_energy();

    oprintf_out("@IRC  %3d %18.12lf  %18.12lf\n", i, g_step(i).g_energy(), DE);
  }
  oprintf_out("@IRC ----------------------------------------------\n\n");

//Printing Internal Coordinates for Each step
  oprintf_out("@IRC -----------------------------------------------------\n");
  oprintf_out("@IRC              ****     IRC Steps     ****             \n");
  oprintf_out("@IRC -----------------------------------------------------");
  for(int j=0; j < dim/blocks; j++)
  {
    oprintf_out("\n@IRC        |          Distance         |\n");
    oprintf_out(  "@IRC Step   | Step    Arc       Line    |");
    for(int i = (j*blocks); i < ((j+1)* blocks); i++)
    {
      oprintf_out("    Coord %3d", i);
    }
    oprintf_out("\n");
    oprintf_out("@IRC --------------------------------------");
    for(int i = (j*blocks); i < ((j+1)* blocks); i++)
    {
      oprintf_out("-------------");
    }
    oprintf_out("\n");
    for (std::size_t i=0; i<steps.size(); ++i)
    {
      oprintf_out("@IRC  %3d %9.2lf %9.5lf  %9.5lf   ", i, sign*g_step(i).g_step_dist(), sign*g_step(i).g_arc_dist(), sign*g_step(i).g_line_dist());
      for(int k = (j*blocks); k < ((j+1)*blocks); k++)
        oprintf_out("%13.8f",g_step(i).g_q()[k]);
      oprintf_out("\n");

        oprintf_out("%13.8f",g_step(i).g_q()[k]);
      oprintf_out("\n");
    }
    oprintf_out("@IRC --------------------------------------");
    for(int i = (j*blocks); i < ((j+1)* blocks); i++)
    {
      oprintf_out("-------------");
    }
  }
  if(dim % blocks != 0)
  {
    oprintf_out("\n@IRC         |          Distance         |\n");
    oprintf_out(  "@IRC  Step   | Step    Arc       Line    |");
    for(int i = (dim - (dim % blocks)); i < dim; i++)
    {
      oprintf_out("    Coord %3d", i);
    }
    oprintf_out("\n");
    oprintf_out("@IRC --------------------------------------");
    for(int i = (dim - (dim % blocks)); i < dim; i++)
    {
      oprintf_out("-------------");
    }
    oprintf_out("\n");
    for (std::size_t i=0; i<steps.size(); ++i)
    {
      oprintf_out("@IRC  %3d %9.2lf %9.5lf  %9.5lf   ", i, sign*g_step(i).g_step_dist(), sign*g_step(i).g_arc_dist(), sign*g_step(i).g_line_dist());
      for(int k = (dim - (dim % blocks)); k < dim; k++)
        oprintf_out("%13.8f",g_step(i).g_q()[k]);
      oprintf_out("\n");
    }
    oprintf_out("@IRC --------------------------------------");
    for(int i = (dim - (dim % blocks)); i < dim; i++)
    {
      oprintf_out("-------------");
    }
  }

  oprintf_out("\n");
  oprintf_out("\n");

  mol.print_coords(psi_outfile, qc_outfile);
  mol.print_simples(psi_outfile, qc_outfile);
}
"""


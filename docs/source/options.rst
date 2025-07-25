Options
=======

Introduction
------------

Optking uses pydantic to validate user input for types and values. Brief descriptions of each option
is presented below. All options have a reasonable default. In general tightening or loosening
convergence, adding constraints, and changing optimization type should be the only options that
need to be adjusted.

Option Short Cuts
~~~~~~~~~~~~~~~~~

Some of the more important options for general use of ``OptKing``. See below for the full list.

.. grid:: 3

    .. grid-item-card::    General

        * :py:attr:`step_type <optking.v1.optparams.OptParams.step_type>`
        * :py:attr:`opt_type <optking.v1.optparams.OptParams.opt_type>`
        * :py:attr:`g_convergence <optking.v1.optparams.OptParams.g_convergence>`
        * :py:attr:`full_hess_every <optking.v1.optparams.OptParams.full_hess_every>`
        * :py:attr:`intrafrag_step_limit <optking.v1.optparams.OptParams.intrafrag_trust>`
        * :py:attr:`geom_maxiter <optking.v1.optparams.OptParams.geom_maxiter>`
        * :py:attr:`opt_coordinates <optking.v1.optparams.OptParams.opt_coordinates>`

    .. grid-item-card::    IRC

        * :py:attr:`irc_direction <optking.v1.optparams.OptParams.irc_direction>`
        * :py:attr:`irc_points <optking.v1.optparams.OptParams.irc_points>`
        * :py:attr:`irc_step_size <optking.v1.optparams.OptParams.irc_step_size>`

    .. grid-item-card::    Constraints

        * :py:attr:`frozen_distance <optking.v1.optparams.OptParams.frozen_distance>`
        * :py:attr:`frozen_bend <optking.v1.optparams.OptParams.frozen_bend>`
        * :py:attr:`frozen_dihedral <optking.v1.optparams.OptParams.frozen_dihedral>`
        * :py:attr:`frozen_oofp <optking.v1.optparams.OptParams.frozen_oofp>`
        * :py:attr:`frozen_cartesian <optking.v1.optparams.OptParams.frozen_cartesian>`
        * :py:attr:`ranged_distance <optking.v1.optparams.OptParams.ranged_distance>`
        * :py:attr:`ranged_bend <optking.v1.optparams.OptParams.ranged_bend>`
        * :py:attr:`ranged_dihedral <optking.v1.optparams.OptParams.ranged_dihedral>`
        * :py:attr:`ranged_oofp <optking.v1.optparams.OptParams.ranged_oofp>`
        * :py:attr:`ranged_cartesian <optking.v1.optparams.OptParams.ranged_cartesian>`
        * :py:attr:`ext_force_distance <optking.v1.optparams.OptParams.ext_force_distance>`
        * :py:attr:`ext_force_bend <optking.v1.optparams.OptParams.ext_force_bend>`
        * :py:attr:`ext_force_dihedral <optking.v1.optparams.OptParams.ext_force_dihedral>`
        * :py:attr:`ext_force_oofp <optking.v1.optparams.OptParams.ext_force_oofp>`
        * :py:attr:`ext_force_cartesian <optking.v1.optparams.OptParams.ext_force_cartesian>`


Global and Local Options
------------------------

For developers or users interacting with optking's internals, Optking historically has utilized
a single global options dictionary. Some parts of Optking still utilize this global dictionary
Most of the current code's classes and functions; however, accept local options by passing a
dictionary, options object, or explicit parameters. Note - the options reported here are the names
that the user should use when providing keywords to Optking. For developers, Optking may use
different names internally. For starters, variables (with the exception of matrices) should
adhere to PEP8 (lower snake-case).

OptParams
---------

Alphabetized Keywords
---------------------

.. autopydantic_model:: optking.v1.optparams.OptParams

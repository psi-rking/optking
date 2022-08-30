###########
Convergence
###########

.. |delta|  unicode:: U+0394

Optking utilizes a number of optimization presets which mirror and/or mimic the optmization behavior from a number
of popular Quantum Chemistry packages. These may be selected with the *G_CONVERGENCE* keyword. Descriptions of each
preset may be found below. See Notes [#fe] and [#ff] for clarification on what combinations of
criteria are required or allowed.

For greater control one or more ctriteria can be selectively activated through use of the *<CRITERIA>_G_CONVERGENCE* keywords.
In order to modify a preset both *FLEXIBLE_G_CONVERGENCE* and one or more *<CRITERIA>_G_CONVERGENCE* keywords must be
selected in addition to the preset. Specifying *<CRITERIA>_G_CONVERGENCE* without *FLEXIBLE_G_CONVERGENCE* will cause
the preset to be discarded and optking will ONLY consider the *<CRITERIA>_G_CONVERGENCE* keyword for convergence.

As an example the first set of options only changes the `rms_force` threshold. The second changes from `QCHEM` to `GAU_TIGHT` while
loosening the `rms_force` threshold

::

    {"g_convergence": "gau_tight", "rms_force_g_convergence": 3e-5}
    {"g_convergence": "gau_tight", "flexible_g_convergence": True, "rms_force_g_convergence": 3e-5}

.. _`table:optkingconv`:

.. table:: Summary of convergence criteria for *g_convergence* defaults in optking (Same as in Psi4)

    +-----------------------------+----------------------------+----------------------------+----------------------------+----------------------------+----------------------------+
    | *g_convergence*             | |delta| E                  | Max Force                  | RMS Force                  | Max Disp                   | RMS Disp                   |
    +=============================+============================+============================+============================+============================+============================+
    | NWCHEM_LOOSE [#fd]_         |                            | :math:`4.5 \times 10^{-3}` | :math:`3.0 \times 10^{-3}` | :math:`5.4 \times 10^{-3}` | :math:`3.6 \times 10^{-3}` |
    +-----------------------------+----------------------------+----------------------------+----------------------------+----------------------------+----------------------------+
    | GAU_LOOSE [#ff]_            |                            | :math:`2.5 \times 10^{-3}` | :math:`1.7 \times 10^{-3}` | :math:`1.0 \times 10^{-2}` | :math:`6.7 \times 10^{-3}` |
    +-----------------------------+----------------------------+----------------------------+----------------------------+----------------------------+----------------------------+
    | TURBOMOLE [#fd]_            | :math:`1.0 \times 10^{-6}` | :math:`1.0 \times 10^{-3}` | :math:`5.0 \times 10^{-4}` | :math:`1.0 \times 10^{-3}` | :math:`5.0 \times 10^{-4}` |
    +-----------------------------+----------------------------+----------------------------+----------------------------+----------------------------+----------------------------+
    | GAU [#fc]_ [#ff]_           |                            | :math:`4.5 \times 10^{-4}` | :math:`3.0 \times 10^{-4}` | :math:`1.8 \times 10^{-3}` | :math:`1.2 \times 10^{-3}` |
    +-----------------------------+----------------------------+----------------------------+----------------------------+----------------------------+----------------------------+
    | CFOUR [#fd]_                |                            |                            | :math:`1.0 \times 10^{-4}` |                            |                            |
    +-----------------------------+----------------------------+----------------------------+----------------------------+----------------------------+----------------------------+
    | QCHEM [#fa]_ [#fe]_         | :math:`1.0 \times 10^{-6}` | :math:`3.0 \times 10^{-4}` |                            | :math:`1.2 \times 10^{-3}` |                            |
    +-----------------------------+----------------------------+----------------------------+----------------------------+----------------------------+----------------------------+
    | MOLPRO [#fb]_ [#fe]_        | :math:`1.0 \times 10^{-6}` | :math:`3.0 \times 10^{-4}` |                            | :math:`3.0 \times 10^{-4}` |                            |
    +-----------------------------+----------------------------+----------------------------+----------------------------+----------------------------+----------------------------+
    | GAU_TIGHT [#fc]_ [#ff]_     |                            | :math:`1.5 \times 10^{-5}` | :math:`1.0 \times 10^{-5}` | :math:`6.0 \times 10^{-5}` | :math:`4.0 \times 10^{-5}` |
    +-----------------------------+----------------------------+----------------------------+----------------------------+----------------------------+----------------------------+
    | GAU_VERYTIGHT [#ff]_        |                            | :math:`2.0 \times 10^{-6}` | :math:`1.0 \times 10^{-6}` | :math:`6.0 \times 10^{-6}` | :math:`4.0 \times 10^{-6}` | 
    +-----------------------------+----------------------------+----------------------------+----------------------------+----------------------------+----------------------------+

.. rubric:: Footnotes

.. [#fa] Default
.. [#fb] Baker convergence criteria are the same.
.. [#fc] Counterpart NWCHEM convergence criteria are the same.
.. [#fd] Convergence achieved when all active criteria are fulfilled.
.. [#fe] Convergence achieved when **Max Force** and one of **Max Energy** or **Max Disp** are fulfilled.
.. [#ff] Normal convergence achieved when all four criteria (**Max Force**, **RMS Force**,
         **Max Disp**, and **RMS Disp**) are fulfilled. To help with flat 
         potential surfaces, alternate convergence achieved when 100\ :math:`\times`\ *rms force* is less 
         than **RMS Force** criterion.

# OptParams is a class to store all of the optimization parameters.
# The init function will receive a User Option Dictionary (uod) which can
# override default values.
# P = parameters ('self')
# Option keys in the input dictionary are interpreted case-insensitively.
# The enumerated string types are translated to all upper-case within the parameter object.
import json
import logging
import pathlib
import re
from packaging import version

from pydantic.v1 import (
    BaseModel,
    Field,
    ConfigDict,
    root_validator,
    validator,
    PrivateAttr,
)

from optking.exceptions import OptError
from optking import log_name

logger = logging.getLogger(f"{log_name}{__name__}")

CART_STR = r"(?:xyz|xy|yz|x|y|z)"


class InterfragCoords(BaseModel):
    """Validate that the string input to create interfragment coords is mostly correct
    (keys and types correct)"""

    model_config = ConfigDict()
    natoms_per_frag: list[int] = Field(alias="NATOMS PER FRAG")
    a_frag: int = Field(default=1, alias="A FRAG")
    a_ref_atoms: list[list[int]] = Field(alias="A REF ATOMS")
    a_label: str = Field(default="FRAGMENT A")
    a_weights: list[list[float]] = Field(default=[], alias="A WEIGHTS")
    b_frag: int = Field(default=2, alias="B FRAG")
    b_ref_atoms: list[list[int]] = Field(alias="B REF ATOMS")
    b_label: str = Field(default="FRAGMENT B")
    b_weights: list[list[float]] = Field(default=[], alias="B WEIGHTS")
    frozen: list[str] = Field(default=[], alias="FROZEN")

    @root_validator(pre=True)
    @classmethod
    def to_upper(cls, data):
        return {key.upper(): val for key, val in data.items()}

    def to_dict(self, by_alias=True):
        return self.dict(by_alias=by_alias)

    @classmethod
    def from_dict(cls, dict_obj):
        """ Provides consistent v1/v2 interface from optking """
        return cls.parse_obj(dict_obj).to_dict()

class OptParams(BaseModel):

    class Config:
        alias_generator = lambda field_name: field_name.upper()
        anystr_upper = True
        # validate_assignment=True,
        # extra="forbid",
        #  regexes need to use IGNORECASE flag since inputs won't be standardized until after
        # validation

    # SUBSECTION Optimization Algorithm
    # NOTE Internal (in some cases legacy documentation) goes above fields as a commend.
    # User documentation goes below as a string

    # Maximum number of geometry optimization steps
    geom_maxiter: int = Field(gt=0, default=50)
    """The maximum number of geometry optimization steps allowed - Technically this is
    the maximum number of gradients that Optking is allowed to calculate."""

    # If user sets one, assume this.
    alg_geom_maxiter: int = Field(gt=0, default=50)

    # Print level.  1 = normal
    print_lvl: int = Field(ge=1, le=5, default=1, alias="PRINT")
    """An integer between 1 (least printing) and 5 (most printing). This has been largely, but not
    entirely, replaced by using the logging modules ``DEBUG`` and ``INFO`` levels.
    Consider changing the logging handler in ``loggingconfig.py`` or if using Psi4 change the
    logging level from the command line. ``psi4 --loglevel=10...``"""

    # Print all optimization parameters.
    printxopt_params: bool = False
    # output_type: str = Field(pattern=r"FILE|STDOUT|NULL", default="FILE")

    # Specifies minimum search, transition-state search, or IRC following
    opt_type: str = Field(pattern=re.compile(r"MIN|TS|IRC", flags=re.IGNORECASE), default="MIN")
    """One of ``["MIN", "TS", or "IRC"]``. ``OPT_TYPE`` will be changed if ``OPT_TYPE``
    is not provided, but ``STEP_TYPE`` is provided, and the two are inconsistent. If both are
    provided but are inconsistent, an error will be raised.

    +------------------------------------------------------------------+
    | Allowed ``opt_type`` and ``step_type`` values                    |
    +==============+===================================================+
    | ``opt_type`` | compatible ``step_type``                          |
    +--------------+--------------+----+----+------------+-------------+
    | MIN          | **RFO**      | NR | SD | LINESEARCH | Conjugate   |
    +--------------+--------------+----+----+------------+-------------+
    | TS           | **RS_I_RFO** | P_RFO                              |
    +--------------+--------------+------------------------------------+
    | IRC          | N/A                                               |
    +--------------+---------------------------------------------------+
    """

    # Geometry optimization step type, e.g., Newton-Raphson or Rational Function Optimization
    step_type: str = Field(
        pattern=re.compile(r"RFO|RS_I_RFO|P_RFO|NR|SD|LINESEARCH|CONJUGATE", flags=re.IGNORECASE),
        default="RFO",
    )
    """One of ``["RFO", "RS_I_RFO", "P_RFO", "NR", "SD", "LINESEARCH", "CONJUGATE"]``. If ``OPT_TYPE``
    is ``TS`` and ``STEP_TYPE`` is not specified then ``STEP_TYPE`` will be set to ``RS_I_RFO``."""

    # What program to use for evaluating gradients and energies
    program: str = Field(default="psi4")
    """What program to use for running gradient and energy calculations through ``qcengine``."""

    # variation of steepest descent step size
    steepest_descent_type: str = Field(
        pattern=re.compile(r"OVERLAP|BARZILAI_BORWEIN", flags=re.IGNORECASE), default="OVERLAP"
    )
    """One of ``["OVERLAP", "BARZILAI_BORWEIN"]``. Change how the ``SD`` step is calculated (scaled)"""

    # Conjugate gradient step types. See Wikipedia on Nonlinear_conjugate_gradient
    # "POLAK" for Polak-Ribiere. Polak, E.; Ribière, G. (1969).
    # Revue Française d'Automatique, Informatique, Recherche Opérationnelle. 3 (1): 35–43.
    # "FLETCHER" for Fletcher-Reeves.  Fletcher, R.; Reeves, C. M. (1964).
    conjugate_gradient_type: str = Field(
        pattern=re.compile(r"FLETCHER|DESCENT|POLAK", flags=re.IGNORECASE), default="FLETCHER"
    )
    """One of ``["POLAK", "FLETCHER", "DESCENT"]``. Changes how the step direction is calculated."""

    # Geometry optimization coordinates to use.
    # REDUNDANT and INTERNAL are synonyms and the default.
    # DELOCALIZED are the coordinates of Baker.
    # NATURAL are the coordinates of Pulay.
    # CARTESIAN uses only Cartesian coordinates.
    # BOTH uses both redundant and Cartesian coordinates.
    opt_coordinates: str = Field(
        pattern=re.compile(
            r"REDUNDANT|INTERNAL|DELOCALIZED|NATURAL|CARTESIAN|BOTH", flags=re.IGNORECASE
        ),
        default="INTERNAL",
    )
    """One of ``["REDUNDANT", "INTERNAL", "CARTESIAN", "BOTH"]``. ``"INTERNAL"`` is just a synonym for
    ``"REDUNDANT"``. ``"BOTH"`` utilizes a full set of redundant internal coordinates and cartesian
    :math:`(3N - 6+) + (3N) = (6N - 6+)` coordinates."""

    # Do follow the initial RFO vector after the first step?
    rfo_follow_root: bool = False
    """Whether or not to optimize along the previously chosen mode of the augmented hessian matrix"""

    # Root for RFO to follow, 0 being lowest (typical for a minimum)
    rfo_root: int = Field(ge=0, default=0)
    """root for ``RFO`` or ``RS_I_RFO`` to follow. Changing rfo_root for a ``TS`` may lead to a
    higher-order stationary point."""

    # Whether to accept geometry steps that lower the molecular point group. DEFAULT=False
    accept_symmetry_breaking: bool = False
    """Whether to accept steps that lower the molecular point group. Note - as of 0.3.0,
    this is **only** effective when running through Psi4"""

    # TODO This needs a validator to check the allowed values as well as set dynamic_lvl_max depending
    # upon dynamic_lvl
    # Starting level for dynamic optimization (0=nondynamic, higher=>more conservative)
    # `dynamic_lvl=0 prevents changes to algorithm`
    dynamic_level: int = Field(ge=0, le=6, default=0, alias="DYNAMIC_LVL")
    """An integer between 0 and 6. Larger values reflect less aggressive optimization techniques
    If ``dynamic_lvl`` is not set, ``Optking`` will not change the ``dynamic_lvl``. The
    ``dynamic_lvl`` must be > 0 for alternative approaches to be tried.
    A backstep will be triggered (if allowed) by :math:`\\Delta E > 0` in a minimization.
    A step is considered "bad" if :math:`\\Delta E > 0` when no more backsteps are allowed **and**
    iterations :math:`> 5`, **or** there are badly defined internal coordinates or derivatives.
    Default = 0

    +-----------+------+-------+------------+--------------+-------------------------------+
    | dynamic   | step | coord | trust      | backsteps    | criteria to change dynamic_lvl|
    +===========+======+=======+============+==============+================+==============+
    |           |      |       |            |              | decrease       |  increase    |
    +-----------+------+-------+------------+--------------+----------------+--------------+
    |   0       | RFO  | RI    | dynamic    | no           |     none       | none         |
    +-----------+------+-------+------------+--------------+----------------+--------------+
    |   1       | RFO  | RI    | dynamic(d) | no           |     1 bad step | none         |
    +-----------+------+-------+------------+--------------+----------------+--------------+
    |   2       | RFO  | RI    | smaller    | yes (1)      |     1 bad step | none         |
    +-----------+------+-------+------------+--------------+----------------+--------------+
    |   3       | RFO  | BOTH  | small      | yes (1)      |     1 bad step | none         |
    +-----------+------+-------+------------+--------------+----------------+--------------+
    |   4       | RFO  | XYZ   | large      | yes (1)      |     1 bad step | none         |
    +-----------+------+-------+------------+--------------+----------------+--------------+
    |   5       | RFO  | XYZ   | small      | yes (1)      |     1 bad step | none         |
    +-----------+------+-------+------------+--------------+----------------+--------------+
    |   6       | SD   | XYZ   | large      | yes (1)      |     1 bad step | none         |
    +-----------+------+-------+------------+--------------+----------------+--------------+
    |   7       | SD   | XYZ   | small      | yes (1)      |     1 bad step | none         |
    +-----------+------+-------+------------+--------------+----------------+--------------+
    """

    dynamic_lvl_max: int = Field(ge=0, le=6, default=0)
    """How large ``dynamic_lvl`` is allowed to grow. If ``dynamic_lvl`` :math:`> 0`, ``dynamic_lvl_max``
    will default to 6"""

    # IRC step size in bohr(amu)^{1/2}$.
    irc_step_size: float = Field(gt=0.0, default=0.2)
    """Specifies the distance between each converged point along the IRC reaction path in
    :math:`bohr amu^{1/2}`"""

    # IRC mapping direction
    irc_direction: str = Field(
        pattern=re.compile("FORWARD|BACKWARD", flags=re.IGNORECASE), default="FORWARD"
    )
    """One of ``["FORWARD", "BACKWARD"]``. Whether to step in the forward (+) direction along
    the transition state mode (smallest mode of hessian) or backward (-)"""

    # Decide when to stop IRC calculations
    irc_points: int = Field(gt=0, default=20)
    """Maximum number of converged points along the IRC path to map out before quitting.
    For dissociation reactions, where the reaction path may not terminate in
    a minimum, this is needed to cap the number of step's Optking is allowed to take"""

    irc_convergence: float = Field(lt=-0.5, gt=-1.0, default=-0.7)
    """Main criteria for declaring convergence for an IRC. The overlap between the unit forces
    at two points of the IRC is compared to this value to assess whether a minimum has been stepped
    over. If :math:`overlap < irc_convergence`, declare convergence. If an IRC terminates too early,
    this may be symptomatic of a highly curved reaction-path, decrease try
    ``irc_converence = -0.9``"""

    irc_mode: str = Field(
        pattern=re.compile("NORMAL|CONFIRM", flags=re.IGNORECASE), default="NORMAL"
    )
    """Experimental - One of ['NORMAL', 'CONFIRM']. 'CONFIRM' is meant to be used for dissociation
    reactions. The IRC is terminated once the molecule's connectivity has changed. Convergence
    is declared once the original ``covalent_connect`` must be increased by more than 0.4 au."""

    # ------------- SUBSECTION ----------------
    # trust radius - need to write custom validator to check for sane combination
    # of values: One for intrafrag_trust, intrafrag_trust_min, and intrafrag_trust_max,
    # Another for interfrag_trust, interfrag_trust_min, interfrag_trust_max

    # Initial maximum step size in bohr or radian along an internal coordinate
    intrafrag_trust: float = Field(gt=0.0, default=0.5, alias="INTRAFRAG_STEP_LIMIT")
    """Initial maximum step size in bohr or radian in internal coordinates for trust region
    methods (``RFO`` and ``RS_I_RFO``). This value will be updated throughout optimization."""

    # Lower bound for dynamic trust radius [a/u]
    intrafrag_trust_min: float = Field(gt=0.0, default=0.001, alias="INTRAFRAG_STEP_LIMIT_MIN")
    """Lower bound for dynamic trust radius [au]"""

    # Upper bound for dynamic trust radius [au]
    intrafrag_trust_max: float = Field(gt=0.0, default=1.0, alias="INTRAFRAG_STEP_LIMIT_MAX")
    """Upper bound for dynamic trust radius [au]"""

    # Initial maximum step size in bohr or radian along an interfragment coordinate
    interfrag_trust: float = Field(gt=0.0, default=0.5)
    """Initial maximum step size in bohr or radian along an interfragment coordinate"""

    # Lower bound for dynamic trust radius [a/u] for interfragment coordinates
    interfrag_trust_min: float = Field(gt=0.0, default=0.001, alias="INTERFRAG_STEP_LIMIT_MIN")
    """Lower bound for dynamic trust radius [au] for interfragment coordinates"""

    # Upper bound for dynamic trust radius [au] for interfragment coordinates
    interfrag_trust_max: float = Field(gt=0.0, default=1.0, alias="INTERFRAG_STEP_LIMIT_MAX")
    """Upper bound for dynamic trust radius [au] for interfragment coordinates"""

    # Reduce step size as necessary to ensure convergence of back-transformation of
    # internal coordinate step to Cartesian coordinates.
    ensure_bt_convergence: bool = False
    """Reduces step size as necessary to ensure convergence of back-transformation of
    internal coordinate step to Cartesian coordinates"""

    # Do simple, linear scaling of internal coordinates to step limit (not RS-RFO)
    simple_step_scaling: bool = False
    """Do simple, linear scaling of internal coordinates to limit step instead of restricted-step
	(dynamic trust radius) approaches like ``RS_RFO`` or ``RS_I_RFO``"""

    # Set number of consecutive backward steps allowed in optimization
    consecutive_backsteps_allowed: int = Field(ge=0, default=0, alias="CONSECUTIVE_BACKSTEPS")
    """Sets the number of consecutive backward steps allowed in an optimization. This option can be
    updated by ``Optking`` if ``dynamic_lvl`` is > 0. Not recommended for general use."""

    _working_consecutive_backsteps = 0

    # Eigenvectors of RFO matrix whose final column is smaller than this are ignored.
    rfo_normalization_max: float = 100
    """Eigenvectors of RFO matrix with elements greater than this are ignored as candidates for
	the step direction."""

    # Absolute maximum value of step scaling parameter in RS-RFO.
    rsrfo_alpha_max: float = 1e8
    """Absolute maximum value of step scaling parameter in ``RFO`` and ``RS_I_RFO``."""

    # New in python version
    print_trajectory_xyz_file: bool = False

    # Specify distances between atoms to be frozen (unchanged)
    frozen_distance: str = Field(default="", pattern=r"(?:\d\s+){2}*")
    """A string of white-space separated atomic indices to specify that the distances between the
    atoms should be frozen (unchanged).
    ``OptParams({"frozen_distance": "1 2 3 4"})`` Freezes ``Stre(1, 2)`` and ``Stre(3, 4)``"""

    # Specify angles between atoms to be frozen (unchanged)
    frozen_bend: str = Field(default="", pattern=r"(?:\d\s+){3}*")
    """A string of white-space separated atomic indices to specify that the distances between the
    atoms should be frozen (unchanged).
    ``OptParams({"frozen_bend": "1 2 3 2 3 4"})`` Freezes ``Bend(1 2 3)`` and ``Bend(2 3 4)``"""

    # Specify dihedral angles between atoms to be frozen (unchanged)
    frozen_dihedral: str = Field(default="", pattern=r"(?:\d\s+){4}*")
    """ A string of white-space separated atomic indices to specify that the corresponding dihedral
    angle should be frozen (unchanged).
    ``OptParams({"frozen_tors": "1 2 3 4 2 3 4 5"})`` Freezes ``Tors(1, 2, 3, 4)`` and ``Tors(2, 3, 4, 5)``"""

    # Specify out-of-plane angles between atoms to be frozen (unchanged)
    frozen_oofp: str = Field(default="", pattern=r"(?:\d\s+){4}*")
    """A string of white-space separated atomic indices to specify that the corresponding
    out-of-plane angle should be frozen.
    atoms should be frozen (unchanged).
    ``OptParams({"frozen_oofp": "1 2 3 4"})`` Freezes ``OOFP(1, 2, 3, 4)``"""

    # Specify atom and X, XY, XYZ, ... to be frozen (unchanged)
    frozen_cartesian: str = Field(default="", pattern=rf"(?:\d\s{CART_STR}\s?)*")
    """A string of white-space separated atomic indices and Cartesian labels to specify that the
    Cartesian coordinates for a given atom should be frozen (unchanged).
    ``OptParams({"frozen_cartesian": "1 XYZ 2 XY 2 Z"})`` Freezes ``CART(1, X)``,
    ``CART(1, Y)``, ``CART(1, Z)``, ``CART(2, X)``, etc..."""

    # constrain ALL torsions to be frozen.
    freeze_all_dihedrals: bool = False
    """A shortcut to request that all dihedrals should be frozen."""

    # For use only with ``freeze_all_dihedrals`` unfreeze a small subset of dihedrals
    unfreeze_dihedrals: str = Field(default="", pattern=r"(?:\d\s+){4}*")
    """A string of white-space separated atomic indices to specify that the corresponding dihedral
    angle should be **unfrozen**. This keyword is meant to be used in conjunction with
    ``FREEZE_ALL_DIHEDRALS``"""

    # Specify distance between atoms to be ranged
    ranged_distance: str = Field(default="", pattern=rf"(?:(?:\d\s*){2}(?:\d+\.\d+\s*){2})*")
    """A string of white-space separated atomic indices and bounds for the distance between two
    atoms.
    ``OptParams({"ranged_distance": 1 2 2.3 2.4})`` Forces :math:`2.3 <`  ``Stre(1, 2)``
    :math:`< 2.4` """

    # Specify angles between atoms to be ranged
    ranged_bend: str = Field(default="", pattern=rf"(?:(?:\d\s*){3}(?:\d+\.\d+\s*){2})*")
    """A string of white-space separated atomic indices and bounds for the angle between three
    atoms.
    ``OptParams({1 2 3 100 110})`` Forces :math:`100^{\\circ} <` ``Bend(1, 2, 3)``
    :math:`< 110^{\\circ}`"""

    # Specify dihedral angles between atoms to be ranged
    ranged_dihedral: str = Field(default="", pattern=rf"(?:(?:\d\s*){4}(?:\d+\.\d+\s*){2})*")
    """A string of white-space separated atomic indices and bounds for the torsion angle of four
    atoms. The order of specification determines whether the dihedral is a proper or improper
    torsion/dihedral.
    ``OptParams({"ranged_dihedral": "1 2 3 4 100 110"})`` Forces
    :math:`100^{\\circ} <` ``Tors(1, 2, 3, 4)`` :math:`< 110^{\\circ}`"""

    # Specify out-of-plane angles between atoms to be ranged
    ranged_oofp: str = Field(default="", pattern=rf"(?:(?:\d\s*){4}(?:\d+\.\d+\s*){2})*")
    """A string of white-space separated atomic indices and bounds for the out of plane angle
    defined by four atoms where the second atom is the central atom.
    ``OptParams({"ranged_oofp": "1 2 3 4 100 110"})`` Forces
    :math:`100^{\\circ} <` ``Oofp(1, 2, 3, 4)`` :math:`< 110^{\\circ}`"""

    # Specify atom and X, XY, XYZ, ... to be ranged
    ranged_cartesian: str = Field(default="", pattern=rf"(?:\d\s+{CART_STR}\s+(?:\d+\.\d+\s*){2})*")
    """A string of white-space separated atomic indices, Cartesian labels, and bounds for the
    Cartesian coordinates of a given atom. ``OptParams({"ranged_cart": "1 XYZ 2.0 2.1"})`` Forces
    :math:`2.0 <` ``Cart(1, X), Cart(1, Y), Cart(1, Z)`` :math:`< 2.1` (Angstroms)"""

    # Specify distances for which extra force will be added
    ext_force_distance: str = Field(default="", pattern=rf"(?:(?:\d\s*){2}\(.*?\))*")
    """A string of white-space separated, atomic indices (2) followed by a single variable equation
    surrounded in either a single or double quotation mark.
    Example: ``"1 2 'Sin(x)'"`` or ``'1 2 "Sin(x)"'`` evaluates the force along the coordinate
    as a 1-dimensional sinusoidal function where x is the "value" (distance [bohr]) of the
    coordinate (stretch)."""

    # Specify angles for which extra force will be added
    ext_force_bend: str = Field(default="", pattern=rf"(?:(?:\d\s*){3}\(.*?\))*")
    """A string of white-space separated atomic indices (3) followed by a single variable equation
    surrounded in either a single or double quotation mark.
    Example: ``"1 2 3 'Sin(x)'"`` evaluates the force along the coordinate as a 1-D
    sinusoidal function where x is the "value" (angle [radians]) of the coordinate (bend)"""

    # Specify dihedral angles for which extra force will be added
    ext_force_dihedral: str = Field(default="", pattern=rf"(?:(?:\d\s*){4}\(.*?\))*")
    """A string of white-space separated atomic indices (4) followed by a single variable equation
    surrounded in either a single or double quotation mark.
    Example: ``"1 2 3 4 'Sin(x)'"`` evaluates the force along the coordinate as a 1-D
    sinusoidal function where x is the "value" (angle [radians]) of the coordinate (torsion)"""
    
    # Specify out-of-plane angles for which extra force will be added
    ext_force_oofp: str = Field(default="", pattern=rf"(?:(?:\d\s*){4}\(.*?\))*")
    """A string of white-space separated atomic indices (4) followed by a single variable equation
    surrounded in either a single or double quotation mark.
    Example: ``"1 2 3 4 'Sin(x)'"`` evaluates the force along the coordinate as a 1-D
    sinusoidal function where x is the "value" (angle [radians]) of the coordinate (oofp)"""

    # Specify Cartesian coordinates for which extra force will be added
    ext_force_cartesian: str = Field(default="", pattern=rf"(?:(?:\d\s+{CART_STR})\(.*?\))*")
    """A string of whitecaps separated atomic indices (1) and Cartesian labels, followed by a
    single variable equation surrounded in either a single or double quotation mark.
    Example: ``"1 X 'Sin(x)'"`` evaluates the force along the coordinate as 1 1-D sinusoidal
    function where x is the "value" (angle [bohr]) of the coordinate (bohr)"""

    # Should an xyz trajectory file be kept (useful for visualization)?
    # P.print_trajectory_xyz = uod.get('PRINT_TRAJECTORY_XYZ', False)
    # Symmetry tolerance for testing whether a mode is symmetric.
    # P.symm_tol("SYMM_TOL", 0.05)
    #
    # SUBSECTION Convergence Control.

    g_convergence: str = Field(
        pattern=re.compile(
            r"QCHEM|MOLPRO|GAU|GAU_LOOSE|GAU_TIGHT|GAU_VERYTIGHT|TURBOMOLE|CFOUR|NWCHEM_LOOSE|INTERFRAG_TIGHT",
            flags=re.IGNORECASE,
        ),
        default="QCHEM",
    )
    """A set of optimization criteria covering the change in energy, magnitude of the forces and
    the step_size. One of ``["QCHEM", "MOLPRO", "GAU", "GAU_LOOSE", "GAU_TIGHT", "GAU_VERYTIGHT",
    "TURBOMOLE", "CFOUR", "NWCHEM_LOOSE", "INTERFRAG_TIGHT"]``. Specification of any
    ``MAX_*_G_CONVERGENCE`` or ``RMS_*_G_CONVERGENCE`` options will overwrite the criteria set here.
    If ``flexible_g_convergence`` is also on then the specified keyword will be appended.
    See Table :ref:`Geometry Convergence <table:optkingconv>` for details."""

    # _conv_rms_force = -1
    # _conv_rms_disp = -1
    # _conv_max_DE = -1
    # _conv_max_force = -1
    # _conv_max_disp = -1
    conv_max_force: float = Field(default=3.0e-4, alias="MAX_FORCE_G_CONVERGENCE")
    """Convergence criterion for geometry optimization: maximum force (internal coordinates, au)"""

    conv_rms_force: float = Field(default=3.0e-4, alias="RMS_FORCE_G_CONVERGENCE")
    """Convergence criterion for geometry optimization: maximum force (internal coordinates, au)"""

    conv_max_DE: float = Field(default=1.0e-6, alias="MAX_ENERGY_G_CONVERGENCE")
    """Convergence criterion for geometry optimization: maximum energy change"""

    conv_max_disp: float = Field(default=1.2e-3, alias="MAX_DISP_G_CONVERGENCE")
    """Convergence criterion for geometry optimization: maximum displacement (internal coordinates, au)"""

    conv_rms_disp: float = Field(default=1.2e-3, alias="RMS_DISP_G_CONVERGENCE")
    """Convergence criterion for geometry optimization: rms displacement (internal coordinates, au)"""
    # Even if a user-defined threshold is set, allow for normal, flexible convergence criteria

    flexible_g_convergence: bool = False
    """Normally, any specified ``*_G_CONVERGENCE`` keyword like ``MAX_FORCE_G_CONVERGENCE`` will be
    obeyed exclusively. If active, ``FLEXIBLE_G_CONVERGENCE`` appends to ``G_CONVERGENCE`` with the
    value from ``*_G_CONVERGENCE`` instead of overriding.
    """

    #
    # SUBSECTION Hessian Update
    # Hessian update scheme
    hess_update: str = Field(pattern=r"NONE|BFGS|MS|POWELL|BOFILL", default="BFGS")
    """one of: ``[NONE, "BFGS", "MS", "POWELL", "BOFILL"]``
    Update scheme for the hessian. Default depends on ``OPT_TYPE``"""

    # Number of previous steps to use in Hessian update, 0 uses all
    hess_update_use_last: int = Field(ge=0, default=4)
    """Number of previous steps to use in Hessian update, 0 uses all steps."""

    # Do limit the magnitude of changes caused by the Hessian update?
    hess_update_limit: bool = True
    """Do limit the magnitude of changes caused by the Hessian update?
    If ``hess_update_limit`` is True, changes to the Hessian from the update are limited
    to the larger of ``hess_update_limit_scale`` * (current value) and
    ``hess_update_limit_max`` [au].  By default, a Hessian value cannot be changed by more
    than 50% and 1 au."""

    # If |hess_update_limit| is True, changes to the Hessian from the update are limited
    # to the larger of |hess_update_limit_scale| * (current value) and
    # |hess_update_limit_max| [au].  By default, a Hessian value cannot be changed by more
    # than 50% and 1 au.
    hess_update_limit_max: float = Field(ge=0.0, default=1.00)
    """Absolute upper limit for how much any given Hessian value can be changed when updating"""

    hess_update_limit_scale: float = Field(ge=0.0, le=1.0, default=0.50)
    """Relative upper limit for how much any given Hessian value can be changed when updating"""

    hess_update_den_tol: float = Field(gt=0.0, default=1e-7)
    """Denominator check for hessian update."""

    hess_update_dq_tol: float = Field(ge=0.0, default=0.5)
    """Hessian update is avoided if any internal coordinate has changed by more than this in
    radians/au"""

    # SUBSECTION Using external Hessians
    # Do read Cartesian Hessian?  Only for experts - use
    # |Optking__full_hess_every| instead.
    cart_hess_read: bool = False
    """Do read Cartesian Hessian? Recommended to use ``full_hess_every`` instead.
    cfour format or ``.json`` file (`AtomicResult <https://molssi.github.io/QCElemental/>`__)
    allowed. The filetype is determined by the presence of a ``.json`` extension. The cfour hessian
    format specifies that the first line contains the number of atoms. Each subsequent line
    contains three hessian values provided in
    `row-major order <https://en.wikipedia.org/wiki/Row-_and_column-major_order>`__."""
    
    # accompanies cart_hess_read. The default is not validated
    # Need two options here because str_to_upper cannot be turned of for individual members of the Model
    # _hessian_file avoids str_to_upper. Captitalization does not seem to be an issue for V1.
    hessian_file: pathlib.Path = Field(default=pathlib.Path(""), validate_default=False)
    """Accompanies ``CART_HESS_READ``. path to file where hessian has been saved.
    WARNING: As of Psi4 v1.10~nightly psi4.optimize() overrides this variable. If you have written
    a hessian to disk, copy the file to
    ``psi4.core.write_file_prefix(psi4.core.get_active_molecule().name())`` or use
    ``optking.optimize_psi4()``
    """
    # _hessian_file: pathlib.Path = pathlib.Path("")

    # Frequency with which to compute the full Hessian in the course
    # of a geometry optimization. 0 means to compute the initial Hessian only,
    # 1 means recompute every step, and N means recompute every N steps. The
    # default (-1) is to never compute the full Hessian.
    full_hess_every: int = Field(ge=-1, default=-1)
    """Frequency with which to compute the full Hessian in the course
    of a geometry optimization. 0 means to compute the initial Hessian only,
    1 means recompute every step, and N means recompute every N steps. -1 indicates that the
    full hessian should never be computed."""

    # Model Hessian to guess intrafragment force constants
    intrafrag_hess: str = Field(
        pattern=re.compile(r"SCHLEGEL|FISCHER|SIMPLE|LINDH|LINDH_SIMPLE", flags=re.IGNORECASE),
        default="SCHLEGEL",
    )
    """Model Hessian to guess intrafragment force constants. One of ``["SCHLEGEL", "FISCHER",
    "SIMPLE", "LINDH", "LINDH_SIMPLE"]``"""

    # Re-estimate the Hessian at every step, i.e., ignore the currently stored Hessian.
    h_guess_every: bool = False
    """Re-estimate the Hessian at every step, i.e., ignore the currently stored Hessian. This is NOT
    recommended"""
    _working_steps_since_last_H = 0

    #
    # SUBSECTION Back-transformation to Cartesian Coordinates Control
    bt_max_iter: int = Field(gt=0, default=25)
    """Maximum number of iterations allowed to converge back-transformation"""

    bt_dx_conv: float = Field(gt=0.0, default=1.0e-7)
    """Threshold for the change in any given Cartesian coordinate during iterative
    back-transformation."""

    bt_dx_rms_change_conv: float = Field(gt=0.0, default=1.0e-12)
    """Threshold for RMS change in Cartesian coordinates during iterative back-transformation."""

    # The following should be used whenever redundancies in the coordinates
    # are removed, in particular when forces and Hessian are projected and
    # in back-transformation from delta(q) to delta(x).
    bt_pinv_rcond: float = Field(gt=0.0, default=1.0e-6)
    """Threshold to remove redundancies from generalized inverse. Corresponds to the ``rcond`` from
    `numpy <https://numpy.org/doc/stable/reference/generated/numpy.linalg.pinv.html>`__.
    The following should be used whenever redundancies in the coordinates
    are removed, in particular when forces and Hessian are projected and
    in back-transformation from delta(q) to delta(x)."""

    #
    # For multi-fragment molecules, treat as single bonded molecule or via interfragment
    # coordinates. A primary difference is that in ```MULTI``` mode, the interfragment
    # coordinates are not redundant.
    frag_mode: str = Field(
        pattern=re.compile(r"SINGLE|MULTI", flags=re.IGNORECASE), default="SINGLE"
    )
    """For multi-fragment molecules, treat as single bonded molecule or via interfragment
    coordinates. A primary difference is that in ``MULTI`` mode, the interfragment
    coordinates are not redundant."""

    # Which atoms define the reference points for interfragment coordinates?
    frag_ref_atoms: list[list[list[int]]] = []
    """Which atoms define the reference points for interfragment coordinates?
    Example for a simple diatomic dimer like :math:`Ne_2` ``[[[1]], [[2]]]``. Please see the section
    on multi-fragment optimizations for more information. :ref:`Multi-Fragment Optimizations<DimerFrag>` """

    # Do freeze all fragments rigid?
    freeze_intrafrag: bool = False
    """Whether to freeze all intrafragment coordinates (rigid molecules). Only optimize the
    interfragment coordinates."""

    # Do freeze all interfragment modes?
    # P.inter_frag = uod.get('FREEZE_INTERFRAG', False)
    # When interfragment coordinates are present, use as reference points either
    # principal axes or fixed linear combinations of atoms.
    interfrag_mode: str = Field(
        pattern=re.compile(r"FIXED|PRINCIPAL_AXES", re.IGNORECASE), default="FIXED"
    )
    """One of ``['FIXED', 'PRINCIPAL_AXES']``. Use either principal axes or fixed linear combinations
    of atoms as reference points for generating the interfragment coordinates."""

    add_auxiliary_bonds: bool = False
    """Add bond coordinates for atoms separated by less than :math:`2.5 \\times` their covalent
    radii"""

    auxiliary_bond_factor: float = Field(gt=1.0, default=2.5)
    """This factor times the standard covalent distance is used to add extra stretch coordinates."""

    interfrag_dist_inv: bool = False
    """Do use 1/R for the interfragment stretching coordinate instead of R?"""

    interfrag_collinear_tol: float = Field(gt=0.0, default=0.01)
    """Used for determining which atoms in a system are too collinear to be chosen as default
    reference atoms. We avoid collinearity. Greater is more restrictive."""

    # Let the user submit a dictionary (or array of dictionaries) for
    # the interfrag coordinates. Validation occurs below
    interfrag_coords: list[dict] = []
    """Let the user submit a dictionary (or array of dictionaries) for
    the interfrag coordinates. The input may also be given as a string, but the string input must be
    "loadable" as a python dictionary. See input examples
    :ref:`Multi-Fragment Optimzations <DimerFrag>`."""

    interfrag_hess: str = Field(
        pattern=re.compile(r"DEFAULT|FISCHER_LIKE", flags=re.IGNORECASE), default="DEFAULT"
    )
    """Model Hessian to guess interfragment force constants. One of ``["DEFAULT", "FISCHER_LIKE"]``
    """

    covalent_connect: float = Field(gt=0.0, default=1.3)
    """When determining connectivity, a bond is assigned if the interatomic distance
    is less than ``COVANLENT_CONNECT`` :math:` \\times ` the sum of covalent radii.
    When connecting disparate fragments and ``FRAG_MODE`` is SINGLE, a "bond"
    is assigned if interatomic distance is less than (``COVALENT_CONNECT``) :math:`\\times` sum of
    covalent radii. The value is then increased until all the fragments are connected directly
    or indirectly."""

    # interfragment_connect: float = Field(gt=0.0, default=1.8)
    """When connecting disparate fragments and ``FRAG_MODE`` is ``SINGLE``, a "bond"
    is assigned if interatomic distance is less than (``INTERFRAGMENT_CONNECT``) :math:`\\times` the
    sum of covalent radii. The value is then increased until all the fragments are connected
    directly or indirectly."""

    h_bond_connect: float = Field(gt=0.0, default=4.3)
    """General, maximum distance for the definition of H-bonds."""

    include_oofp: bool = False
    """Add out-of-plane angles (usually not needed)"""

    #
    #
    # SUBSECTION Misc.
    # Do save and print the geometry from the last projected step at the end
    # of a geometry optimization? Otherwise (and by default), save and print
    # the previous geometry at which was computed the gradient that satisfied
    # the convergence criteria.
    # P.final_geom_write = uod.get('FINAL_GEOM_WRITE', False)

    test_B: bool = False
    """Do test B matrix?"""
    test_derivative_B: bool = False
    """Do test derivative B matrix?"""

    # Only generate the internal coordinates and then stop (boolean) UNUSED
    # generate_intcos_exit: bool = False
    # Keep internal coordinate definition file.
    # keep_intcos: bool = False UNUSED
    linesearch_step: float = Field(gt=0.0, default=0.100)
    """Initial stepsize  when performing a 1-D linesearch"""

    linesearch: bool = False
    # Guess at Hessian in steepest-descent direction.
    """performs linesearch on top of current ``STEP_TYPE``."""

    sd_hessian: float = Field(gt=0.0, default=1.0)
    """Guess at Hessian in steepest-descent direction (acts as a stepsize control)."""

    # # -- Items below are unlikely to need modified

    # Boundary to guess if a torsion or out-of-plane angle has passed through 180
    # during a step.
    fix_val_near_pi: float = 1.57

    # Torsional angles will not be computed if the contained bond angles are within
    # this many radians of zero or 180. (< ~1 and > ~179 degrees)
    # only used in v3d.py
    v3d_tors_angle_lim: float = 0.017

    # cos(torsional angle) must be this close to -1/+1 for angle to count as 0/pi
    # only used in v3d.py
    v3d_tors_cos_tol: float = 1e-10

    # if bend exceeds this value, then also create linear bend complement
    linear_bend_threshold: float = 3.05  # about 175 degrees

    # If bend is smaller than this value, then never fix its associated vectors
    # this allows iterative steps through and near zero degrees.
    small_bend_fix_threshold: float = 0.35

    # Threshold for which entries in diagonalized redundant matrix are kept and
    # inverted while computing a generalized inverse of a matrix
    redundant_eval_tol: float = 1.0e-10  # to be deprecated.

    # threshold for which eigenvalues, eigenvector values, and other floating point
    # values are considered to be zero. Silences numeric noise that can cause issues
    # with matrix inversion. Replaces redundant_eval_tol
    linear_algebra_tol = 1e-10

    # --- SET INTERNAL OPTIMIZATION PARAMETERS ---
    _i_max_force: bool = False
    _i_rms_force: bool = False
    _i_max_DE: bool = False
    _i_max_disp: bool = False
    _i_rms_disp: bool = False
    _i_untampered: bool = False

    def to_dict(self, by_alias=True):
        """ Specialized form of __dict__. Makes sure to include convergence keys that are hidden """
        save = self.dict(by_alias=by_alias)
        # This was used in the v2 option. Prevents anyone from setting the vars explicity
        # But during serialization, need to know what these are.
        # include = {
        #     "_i_max_force": self._i_max_force,
        #     "_i_rms_force": self._i_rms_force,
        #     "_i_max_DE": self._i_max_DE,
        #     "_i_max_disp": self._i_max_disp,
        #     "_i_rms_disp": self._i_rms_disp,
        #     "_i_untampered": self._i_untampered# or key in include:
        # }
        # for key in include:
            # save.update(include)
        return save

    def __str__(self):
        s = "\n\t\t -- Optimization Parameters --\n"
        for name, value in self.dict(by_alias=True).items():
            if "_i_" in name[:3]:
                continue
            s += "\t%-30s = %15s\n" % (name, value)
        s += "\n"
        return s

    @root_validator(pre=True)
    @classmethod
    def save_raw_input(cls, data):
        """Stash user input before any user input checking or transformations are performed (for
        instance str_to_upper)

        Notes
        -----
        model_set_fields is only set after validation so it can't be used to determine
        what option B should be set to if option A is set by the user.

        By running this before
        validation we can cache all the user inputs and then compare against later. Need to be
        careful that any of validation is before after

        """

        upper_data = {key.upper(): val for key, val in data.items()}
        cls._raw_input = upper_data
        cls._special_defaults = {}
        # create a special dict to hold keywords that were changed by validation
        return upper_data

    @root_validator()
    def validate_algorithm(cls, fields):
        """Ensure that if the user has selected both an opt_type and step_type that they are
        compatible. If the user has selected `opt_type=TS` OR a ``step_type`` consistent with ``TS``
        then change the other keyword to have the appropriate keyword"""

        min_step_types = ["RFO", "NR", "SD", "CONJUGATE", "LINESEARCH"]
        ts_step_types = ["RS_I_RFO", "P_RFO"]
        set_vars = cls._raw_input

        if "OPT_TYPE" in set_vars and "STEP_TYPE" in set_vars:
            if fields["opt_type"] == "TS":
                assert fields["step_type"] not in min_step_types
            elif fields["opt_type"] == "MIN":
                assert fields["step_type"] not in ts_step_types
        elif "OPT_TYPE" in set_vars and fields["opt_type"] == "TS":
            # User has selected TS. Change algorithm to RS_I_RFO
            fields.update({"step_type": "RS_I_RFO"})
        elif "STEP_TYPE" in set_vars and fields["step_type"] in ts_step_types:
            fields.update({"opt_type": "TS"})
        return fields

    @root_validator()
    def validate_convergence(cls, fields: dict):
        """Set active variables depending upon the PRESET that has been provided and whether any
        specific values were individually specified by the user."""

        # stash so that __setattr__ doesn't affect which variables have been changed
        # Start by setting each individual convergence option from preset
        set_vars = cls._raw_input

        conv_spec = CONVERGENCE_PRESETS[fields["g_convergence"]]
        for key, val in conv_spec.items():
            fields.update({key: val})
            # cls.__setattr__(key, val)

        # Table to easily correlate the user / psi4 name, internal keyword name,
        # and internal active flag for keyword
        keywords = [
            ("MAX_FORCE_G_CONVERGENCE", "conv_max_force", "_i_max_force"),
            ("RMS_FORCE_G_CONVERGENCE", "conv_rms_force", "_i_rms_force"),
            ("MAX_ENERGY_G_CONVERGENCE", "conv_max_DE", "_i_max_DE"),
            ("MAX_DISP_G_CONVERGENCE", "conv_max_disp", "_i_max_disp"),
            ("RMS_DISP_G_CONVERGENCE", "conv_rms_disp", "_i_rms_disp"),
        ]

        keys_present = [True if keyword_set[0] in set_vars else False for keyword_set in keywords]

        # Skip if no tampering occured. The following code sets active to False for any values the
        # user didn't explicitly specify. 
        if any(keys_present):
            # Summary: If ANY convergence options were specified by the user
            # (without flexible convergence being on), turn untampered on and set all options to inactive
            for keyword_set in keywords:
                if keyword_set[0] in set_vars:
                    # Keyword was specified by the user. Set value and active flag
                    fields.update({keyword_set[1]: set_vars[keyword_set[0]]})
                    fields.update({keyword_set[2]: True})
                    fields.update({"_i_untampered": False}) # Tampering has occured!!!!
                    if fields["flexible_g_convergence"]:
                        # use flexible conv criteria don't leave criteria preset active except for mods
                        fields.update({"_i_untampered": True})
                    else:
                        fields.update({"_i_untampered": False})
                else:
                    # Keyword was not specified by user. Deactivate all other keywords if running in normal mode
                    # if in flexible mode, leave other criteria active
                    if not fields["flexible_g_convergence"]:
                        fields.update({keyword_set[2]: False})
        return fields

    @root_validator()
    @classmethod
    def validate_iter(cls, fields: dict):

        if fields["opt_type"] == "IRC" and "GEOM_MAXITER" not in cls._raw_input:
            fields.update({"geom_maxiter": fields["irc_points"] * 15})
        elif fields["geom_maxiter"] < fields["alg_geom_maxiter"]:
            fields.update({"alg_geom_maxiter": fields["geom_maxiter"]})
        return fields

    @root_validator()
    def validate_trustregion(cls, fields):
        # Initial Hessian guess for Cartesian's with coordinates BOTH is stupid, so don't scale
        #   step size down too much.  Steepest descent has no good hessian either.
        set_vars = cls._raw_input

        if "INTRAFRAG_TRUST_MIN" not in set_vars:
            intra_trust = fields["intrafrag_trust"]
            if fields["opt_coordinates"] == "BOTH":
                fields.update({"intrafrag_trust_min": intra_trust})
            elif fields["step_type"] == "SD":  # steepest descent, use constant stepsize
                fields.update({"intrafrag_trust_min": intra_trust})
            elif any(
                [
                    fields["ext_force_distance"],
                    fields["ext_force_bend"],
                    fields["ext_force_dihedral"],
                    fields["ext_force_oofp"],
                    fields["ext_force_cartesian"],
                ]
            ):
                # with external forces, the check for trust radius will be inapt
                # so don't let minimum step get shrunk too much.
                fields.update({"intrafrag_trust_min": intra_trust / 2.0})

        # breakpoint()
        if (fields["opt_type"] in ["IRC", "TS"] or fields["step_type"] == "RS_I_RFO") and "INTRAFRAG_STEP_LIMIT" not in set_vars:
            fields.update({"intrafrag_trust": 0.2})  # start with smaller intrafrag_trust

        if fields["intrafrag_trust_max"] < fields["intrafrag_trust"]:
            fields.update({"intrafrag_trust": fields["intrafrag_trust_max"]})
        return fields

    @root_validator()
    @classmethod
    def validate_hessian(cls, fields):

        set_vars = cls._raw_input
        opt_type = fields["opt_type"]  # fetch once

        # Original Lindh specification was to redo at every step.
        if "H_GUESS_EVERY" not in set_vars and fields["intrafrag_hess"] == "LINDH":
            fields.update({"h_guess_every": True})

        # Default for cartesians: use Lindh force field for initial guess, then BFGS.
        if fields["opt_coordinates"] == "CARTESIAN":
            if "INTRAFRAG_HESS" not in set_vars:
                fields.update({"intrafrag_hess": "LINDH"})
                if "H_GUESS_EVERY" not in set_vars:
                    fields.update({"h_guess_every": False})

        # Set Bofill as default for TS optimizations.
        if opt_type == "TS" or opt_type == "IRC":
            if "HESS_UPDATE" not in set_vars:
                fields.update({"hess_update": "BOFILL"})

        # Make trajectory file printing the default for IRC.
        if opt_type == "IRC" and "PRINT_TRAJECTORY_XYZ_FILE" not in set_vars:
            fields.update({"print_trajectory_xyz_file": True})

        # Read cartesian Hessian by default for IRC.
        # Changed to turn cart_hess_read on only if a file path was provided.
        # otherwise full_hess_every will handle providing hessian
        if opt_type == "IRC" and "CART_HESS_READ" not in set_vars:
            if fields["hessian_file"] != pathlib.Path(""):
                fields.update({"cart_hess_read": True})

        if fields["cart_hess_read"] and fields["hessian_file"] == pathlib.Path(""):
            try:
                import psi4
            except ImportError as e:
                logger.error("CART_HESS_READ was turned on but ``HESSIAN_FILE`` was left empty."
                    "Attempting to read from ``psi4.writer_file_prefix`` has failed. Please"
                    "explicitly provide a file or ensure that psi4 is importable"
                )
                raise e
            name = psi4.core.get_active_molecule().name()
            fields.update(
                {"hessian_file": pathlib.Path(f"{psi4.core.get_writer_file_prefix(name)}.hess")}
            )

        # inactive option
        # if fields.get("generate_intcos_exit"):
        #     fields.get("keep_intcos") = True

        # For IRC, we WILL need a Hessian.  Compute it if not provided.
        # Set full_hess_every to 0 if -1
        if opt_type == "IRC" and fields["full_hess_every"] < 0:
            fields.update({"full_hess_every":  0})
            # fields.get("cart_hess_read") = True  # not sure about this one - test

        # if steepest-descent, then make much larger default
        if fields["step_type"] == "SD" and "CONSECUTIVE_BACKSTEPS" not in set_vars:
            fields.update({"consecutive_backsteps_allowed": 10})

        # For RFO step, eigenvectors of augmented Hessian are divided by the last
        # element unless it is smaller than this value {double}.  Can be used to
        # eliminate asymmetric steps not otherwise detected (e.g. in degenerate
        # point groups). For multi-fragment modes, we presume that smaller
        # Delta-E's are possible, and this threshold should be made larger.
        # if P.fragment_mode == 'MULTI' and 'RFO_NORMALIZATION_MAX' not in uod:
        #     P.rfo_normalization_max = 1.0e5
        # If arbitrary user forces, don't shrink step_size if Delta(E) is poor.
        return fields

    # @root_validator()
    # def validate_hessian_file(cls, fields):
    #     # Stash value of hessian_file in _hessian_file for internal use
    #     # mode before required so that we stash before str_to_upper is called

    #     set_vars = cls._raw_input
    #     hess_file = set_vars.get("hessian_file")
    #     breakpoint()
    #     if hess_file:
    #         fields.update({"hessian_file": pathlib.Path(hess_file)})
    #     return fields

    @root_validator()
    def validate_frag(cls, fields: dict):
        # Finish multi-fragment option setup by forcing frag_mode: MULTI if DimerCoords are provided

        input = fields["interfrag_coords"]
        if input:
            # if interfrag_coords is not empty. Consider whether it is just [{}]
            if isinstance(input, list) and len(input) > 0:
                if isinstance(input[0], dict) and len(input[0]) == 0:
                    # empty dict in list
                    return fields
            fields.update({"frag_mode": "MULTI"})
        return fields

    @validator("interfrag_coords", pre=True)
    @classmethod
    def check_interfrag_coords(cls, val):
        """Make sure required fields and types are sensible for interfrag_coords dict."""

        tmp = val
        if tmp:

            def to_uppercase_key_str(tmp: dict):
                """Convert dict to string object with uppercase keys"""
                tmp = {key.upper(): item for key, item in tmp.items()}
                return json.dumps(tmp)

            # convert to (presumably) dict or list[dict]
            if isinstance(tmp, str):
                tmp = tmp.replace("'", '"')
                tmp = json.loads(tmp)

            # ensure that keys are uppercase and standardize to list of dict
            if isinstance(tmp, dict):
                tmp = [to_uppercase_key_str(tmp)]
            elif isinstance(tmp, (list, tuple)):
                tmp = [to_uppercase_key_str(item) for item in tmp]

            # Validate string as matching InterfragCoords Spec
            for item in tmp:
                if item and item != "{}":
                    assert InterfragCoords.parse_raw(item)
                    # assert InterfragCoords.model_validate_json(item)

            # Now that everything is validated. Convert to dict for storage
            tmp = [json.loads(item) for item in tmp]
            return tmp
        else:
            return [{}]

    @root_validator()
    @classmethod
    def validate_case(cls, fields: dict):
        for key, val in fields.items():
            if isinstance(val, str):
                fields[key] = val.upper()
        return fields

    @classmethod
    def from_internal_dict(cls, params):
        """Assumes that params does not use the input key and syntax, but uses the internal names and
        internal syntax. Meant to be used for recreating options# o dict
        It's probably preferable to dump by alias and then recreate instead of using this"""

        options = cls()  # basic default options
        opt_dict = options.__dict__
        for key, val in opt_dict.items():
            option = params.get(key, val)
            if isinstance(option, str):
                option = option.upper()
            options.__dict__[key] = option

        return options

    # for specialists
    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def conv_criteria(self) -> dict:
        """Returns the currently active values for each convergence criteria. Not the original
        user input / presets"""
        return {
            "conv_max_force": self.conv_max_force,
            "conv_rms_force": self.conv_rms_force,
            "conv_max_disp": self.conv_max_disp,
            "conv_rms_disp": self.conv_rms_disp,
            "conv_max_DE": self.conv_max_DE,
            "i_max_force": self._i_max_force,
            "i_rms_force": self._i_rms_force,
            "i_max_disp": self._i_max_disp,
            "i_rms_disp": self._i_rms_disp,
            "i_max_DE": self._i_max_DE,
        }

    def update_dynamic_level_params(self, run_level):
        logger = logging.getLogger(__name__)  # TODO?
        """
        *dynamic  step   coord   trust      backsteps         criteria
        * run_level                                           for downmove    for upmove
        *  0      RFO    RI      dynamic         no           none            none
        *  1      RFO    RI      dynamic(d)      no           1 bad step
        *  2      RFO    RI      smaller         yes (1)      1 bad step
        *  3      RFO    BOTH    small           yes (1)      1 bad step
        *  4      RFO    XYZ     large           yes (1)      1 bad step
        *  5      RFO    XYZ     small           yes (1)      1 bad step
        *  6      SD     XYZ     large           yes (1)      1 bad step
        *  7      SD     XYZ     small           yes (1)      1 bad step
        *  8  abort
        *  BackStep:
        *   DE > 0 in minimization
        *  BadStep:
        *   DE > 0 and backsteps exceeded and iterations > 5  ** OR **
        *   badly defined internal coordinate or derivative
        """
        if run_level == 0:
            pass
        elif run_level == 1:
            self.opt_coordinates = "REDUNDANT"
            self.consecutiveBackstepsAllowed = 0
            self.step_type = "RFO"
            logger.info(
                "Going to run_level 1: Red. Int., RFO, no backsteps, default, dynamic trust. ~"
            )
        elif run_level == 2:
            self.opt_coordinates = "REDUNDANT"
            self.consecutiveBackstepsAllowed = 2
            self.step_type = "RFO"
            self.intrafrag_trust = 0.2
            self.intrafrag_trust_min = 0.2
            self.intrafrag_trust_max = 0.2
            logger.warning("Going to run_level 2: Red. Int., RFO, 2 backstep, smaller trust. ~")
        elif run_level == 3:
            self.opt_coordinates = "BOTH"
            self.consecutiveBackstepsAllowed = 2
            self.step_type = "RFO"
            self.intrafrag_trust = 0.1
            self.intrafrag_trust_min = 0.1
            self.intrafrag_trust_max = 0.1
            logger.warning(
                "Going to run_level 3: Red. Int. + XYZ, RFO, 2 backstep, smaller trust. ~"
            )
        elif run_level == 4:
            self.opt_coordinates = "CARTESIAN"
            self.consecutiveBackstepsAllowed = 2
            self.step_type = "RFO"
            self.intrafrag_hess = "LINDH"
            self.intrafrag_trust = 0.2
            self.intrafrag_trust_min = 0.2
            self.intrafrag_trust_max = 0.2
            logger.warning("Going to run_level 4: XYZ, RFO, 2 backstep, smaller trust. ~")
        elif run_level == 5:
            self.opt_coordinates = "CARTESIAN"
            self.consecutiveBackstepsAllowed = 2
            self.step_type = "SD"
            self.sd_hessian = 0.3
            self.intrafrag_trust = 0.3
            self.intrafrag_trust_min = 0.3
            self.intrafrag_trust_max = 0.3
            logger.warning("Going to run_level 5: XYZ, SD, 2 backstep, average trust. ~")
        elif run_level == 6:
            self.opt_coordinates = "CARTESIAN"
            self.consecutiveBackstepsAllowed = 2
            self.step_type = "SD"
            self.sd_hessian = 0.6
            self.intrafrag_trust = 0.1
            self.intrafrag_trust_min = 0.1
            self.intrafrag_trust_max = 0.1
            logger.warning("Moving to run_level 6: XYZ, SD, 2 backstep, smaller trust. ~")
        else:
            raise OptError("Unknown value of run_level")


CONVERGENCE_PRESETS = {
    "QCHEM": {
        "_i_untampered": True,
        "conv_max_force": 3.0e-4,
        "_i_max_force": True,
        "conv_max_DE": 1.0e-6,
        "_i_max_DE": True,
        "conv_max_disp": 1.2e-3,
        "_i_max_disp": True,
    },
    "MOLPRO": {
        "_i_untampered": True,
        "conv_max_force": 3.0e-4,
        "_i_max_force": True,
        "conv_max_DE": 1.0e-6,
        "_i_max_DE": True,
        "conv_max_disp": 3.0e-4,
        "_i_max_disp": True,
    },
    "GAU": {
        "_i_untampered": True,
        "conv_max_force": 4.5e-4,
        "_i_max_force": True,
        "conv_rms_force": 3.0e-4,
        "_i_rms_force": True,
        "conv_max_disp": 1.8e-3,
        "_i_max_disp": True,
        "conv_rms_disp": 1.2e-3,
        "_i_rms_disp": True,
    },
    "GAU_TIGHT": {
        "_i_untampered": True,
        "conv_max_force": 1.5e-5,
        "_i_max_force": True,
        "conv_rms_force": 1.0e-5,
        "_i_rms_force": True,
        "conv_max_disp": 6.0e-5,
        "_i_max_disp": True,
        "conv_rms_disp": 4.0e-5,
        "_i_rms_disp": True,
    },
    "GAU_VERYTIGHT": {
        "_i_untampered": True,
        "conv_max_force": 2.0e-6,
        "_i_max_force": True,
        "conv_rms_force": 1.0e-6,
        "_i_rms_force": True,
        "conv_max_disp": 6.0e-6,
        "_i_max_disp": True,
        "conv_rms_disp": 4.0e-6,
        "_i_rms_disp": True,
    },
    "GAU_LOOSE": {
        "_i_untampered": True,
        "conv_max_force": 2.5e-3,
        "_i_max_force": True,
        "conv_rms_force": 1.7e-3,
        "_i_rms_force": True,
        "conv_max_disp": 1.0e-2,
        "_i_max_disp": True,
        "conv_rms_disp": 6.7e-3,
        "_i_rms_disp": True,
    },
    "TURBOMOLE": {
        "_i_untampered": True,
        "conv_max_force": 1.0e-3,
        "_i_max_force": True,
        "conv_rms_force": 5.0e-4,
        "_i_rms_force": True,
        "conv_max_DE": 1.0e-6,
        "_i_max_DE": True,
        "conv_max_disp": 1.0e-3,
        "_i_max_disp": True,
        "conv_rms_disp": 5.0e-4,
        "_i_rms_disp": True,
    },
    "CFOUR": {
        "_i_untampered": True,
        "conv_rms_force": 1.0e-4,
        "_i_rms_force": True,
    },
    "NWCHEM_LOOSE": {
        "_i_untampered": True,
        "conv_max_force": 4.5e-3,
        "_i_max_force": True,
        "conv_rms_force": 3.0e-3,
        "_i_rms_force": True,
        "conv_max_disp": 5.4e-3,
        "_i_max_disp": True,
        "conv_rms_disp": 3.6e-3,
        "_i_rms_disp": True,
    },
    "INTERFRAG_TIGHT": {
        "conv_max_DE": 1.0e-5,
        "_i_max_DE": True,
        "conv_max_force": 1.5e-5,
        "_i_max_force": True,
        "conv_rms_force": 1.0e-5,
        "_i_rms_force": True,
        "conv_max_disp": 6.0e-4,
        "_i_max_disp": True,
        "conv_rms_disp": 4.0e-4,
        "_i_rms_disp": True,
    },
}

FLOATR = r"(:?\d+\.\d+)"
INT = r"(:?\d)"
SEP = r"(:?\d+\.\d+)"
SEP2 = r"\d+\.\d+"
SEP3 = r"\d+\.\d+"
FLOATR = r"\d+\.\d+"
CART_STR = r"(?:xyz|xy|yz|x|y|z)"
LABEL = r"(?:[SRABTDO]|STRE|STRETCH|BOND|BEND|ANGLE|TORS|TORSION|DIHEDRAL)"

# Create a module level, default, options object
Params = OptParams()

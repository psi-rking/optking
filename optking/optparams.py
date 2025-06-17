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
from typing import Union
from pprint import pformat

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    ConfigDict,
)

from .exceptions import OptError
from . import log_name

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

    @model_validator(mode="before")
    @classmethod
    def to_upper(cls, data):
        return {key.upper(): val for key, val in data.items()}

    def to_dict(self):
        return self.model_dump(by_alias=True)


class OptParams(BaseModel):
    model_config = ConfigDict(
        alias_generator=lambda field_name: field_name.upper(),
        # extra="forbid",
        str_to_upper=True,
        # regexes need to use IGNORECASE flag since inputs won't be standardized until after
        # validation
    )

    # SUBSECTION Optimization Algorithm
    # Maximum number of geometry optimization steps
    geom_maxiter: int = Field(gt=0, default=50)
    # If user sets one, assume this.
    alg_geom_maxiter: int = Field(gt=0, default=50)
    # Print level.  1 = normal
    print_lvl: int = Field(ge=1, le=5, default=1, alias="PRINT")
    # Print all optimization parameters.
    printxopt_params: bool = False
    # output_type: str = Field(pattern=r"FILE|STDOUT|NULL", default="FILE")

    # Specifies minimum search, transition-state search, or IRC following
    opt_type: str = Field(pattern=re.compile(r"MIN|TS|IRC", flags=re.IGNORECASE), default="MIN")

    # Geometry optimization step type, e.g., Newton-Raphson or Rational Function Optimization
    step_type: str = Field(
        pattern=re.compile(r"RFO|RS_I_RFO|P_RFO|NR|SD|LINESEARCH|CONJUGATE", flags=re.IGNORECASE),
        default="RFO",
    )

    # What program to use for evaluating gradients and energies
    program: str = Field(default="psi4")

    # variation of steepest descent step size
    steepest_descent_type: str = Field(
        pattern=re.compile(r"OVERLAP|BARZILAI_BORWEIN", flags=re.IGNORECASE), default="OVERLAP"
    )

    # Conjugate gradient step types. See wikipedia on Nonlinear_conjugate_gradient
    # "POLAK" for Polak-Ribiere. Polak, E.; Ribière, G. (1969).
    # Revue Française d'Automatique, Informatique, Recherche Opérationnelle. 3 (1): 35–43.
    # "FLETCHER" for Fletcher-Reeves.  Fletcher, R.; Reeves, C. M. (1964).
    conjugate_gradient_type: str = Field(
        pattern=re.compile(r"FLETCHER|DESCENT|POLAK", flags=re.IGNORECASE), default="FLETCHER"
    )
    # Geometry optimization coordinates to use.
    # REDUNDANT and INTERNAL are synonyms and the default.
    # DELOCALIZED are the coordinates of Baker.
    # NATURAL are the coordinates of Pulay.
    # CARTESIAN uses only cartesian coordinates.
    # BOTH uses both redundant and cartesian coordinates.
    opt_coordinates: str = Field(
        pattern=re.compile(
            r"REDUNDANT|INTERNAL|DELOCALIZED|NATURAL|CARTESIAN|BOTH", flags=re.IGNORECASE
        ),
        default="INTERNAL",
    )

    # Do follow the initial RFO vector after the first step?
    rfo_follow_root: bool = False
    # Root for RFO to follow, 0 being lowest (typical for a minimum)
    rfo_root: int = Field(ge=0, default=0)
    # Whether to accept geometry steps that lower the molecular point group. DEFAULT=False
    accept_symmetry_breaking: bool = False

    # TODO This needs a validator to check the allowed values as well as set dynamic_lvl_max depending
    # upon dynamic_lvl
    # Starting level for dynamic optimization (0=nondynamic, higher=>more conservative)
    # `dynamic_lvl=0 prevents changes to algorithm`
    dynamic_level: int = Field(ge=0, le=6, default=0, alias="DYNAMIC_LVL")
    dynamic_lvl_max: int = Field(ge=0, le=6, default=0)

    # IRC step size in bohr(amu)\ $^{1/2}$.
    irc_step_size: float = Field(gt=0.0, default=0.2)

    # IRC mapping direction
    irc_direction: str = Field(
        pattern=re.compile("FORWARD|BACKWARD", flags=re.IGNORECASE), default="FORWARD"
    )

    # Decide when to stop IRC calculations
    irc_points: int = Field(gt=0, default=20)

    # ------------- SUBSECTION ----------------
    # trust radius - need to write custom validator to check for sane combination
    # of values: One for intrafrag_trust, intrafrag_trust_min, and intrafrag_trust_max,
    # Another for interfrag_trust, interfrag_trust_min, interfrag_trust_max

    # Initial maximum step size in bohr or radian along an internal coordinate
    intrafrag_trust: float = Field(gt=0.0, default=0.5, alias="INTRAFRAG_STEP_LIMIT")

    # Lower bound for dynamic trust radius [a/u]
    intrafrag_trust_min: float = Field(gt=0.0, default=0.001, alias="INTRAFRAG_STEP_LIMIT_MIN")
    # self.intrafrag_trust_min = uod.get("INTRAFRAG_STEP_LIMIT_MIN", 0.001)

    # Upper bound for dynamic trust radius [au]
    intrafrag_trust_max: float = Field(gt=0.0, default=1.0, alias="INTRAFRAG_STEP_LIMIT_MAX")

    # Initial maximum step size in bohr or radian along an interfragment coordinate
    interfrag_trust: float = Field(gt=0.0, default=0.5)

    # Lower bound for dynamic trust radius [a/u] for interfragment coordinates
    interfrag_trust_min: float = Field(gt=0.0, default=0.001, alias="INTERFRAG_TRUST_MIN")
    # Upper bound for dynamic trust radius [au] for interfragment coordinates
    interfrag_trust_max: float = Field(gt=0.0, default=1.0, alias="INTERFRAG_TRUST_MAX")

    # Reduce step size as necessary to ensure convergence of back-transformation of
    # internal coordinate step to cartesian coordinates.
    ensure_bt_convergence: bool = False

    # Do simple, linear scaling of internal coordinates to step limit (not RS-RFO)
    simple_step_scaling: bool = False

    # Set number of consecutive backward steps allowed in optimization
    consecutive_backsteps_allowed: int = Field(ge=0, default=0, alias="CONSECUTIVE_BACKSTEPS")
    _working_consecutive_backsteps = 0

    # Eigenvectors of RFO matrix whose final column is smaller than this are ignored.
    rfo_normalization_max: float = 100

    # Absolute maximum value of step scaling parameter in RS-RFO.
    rsrfo_alpha_max: float = 1e8

    # New in python version
    print_trajectory_xyz_file: bool = False

    # Specify distances between atoms to be frozen (unchanged)
    frozen_distance: str = Field(default="", pattern=r"(?:\d\s+){2}*")
    # Specify angles between atoms to be frozen (unchanged)
    frozen_bend: str = Field(default="", pattern=r"(?:\d\s+){3}*")
    # Specify dihedral angles between atoms to be frozen (unchanged)
    frozen_dihedral: str = Field(default="", pattern=r"(?:\d\s+){4}*")
    # Specify out-of-plane angles between atoms to be frozen (unchanged)
    frozen_oofp: str = Field(default="", pattern=r"(?:\d\s+){4}*")
    # Specify atom and X, XY, XYZ, ... to be frozen (unchanged)

    frozen_cartesian: str = Field(default="", pattern=rf"(?:\d\s{CART_STR}\s?)*")

    # constrain ALL torsions to be frozen.
    freeze_all_dihedrals: bool = False
    # For use only with `freeze_all_dihedrals` unfreeze a small subset of dihedrals
    unfreeze_dihedrals: str = Field(default="", pattern=r"(?:\d\s+){4}*")

    # Specify distance between atoms to be ranged
    ranged_distance: str = Field(default="", pattern=rf"(?:(?:\d\s*){2}(?:\d+\.\d+\s*){2})*")
    # Specify angles between atoms to be ranged
    ranged_bend: str = Field(default="", pattern=rf"(?:(?:\d\s*){3}(?:\d+\.\d+\s*){2})*")
    # Specify dihedral angles between atoms to be ranged
    ranged_dihedral: str = Field(default="", pattern=rf"(?:(?:\d\s*){4}(?:\d+\.\d+\s*){2})*")
    # Specify out-of-plane angles between atoms to be ranged
    ranged_oofp: str = Field(default="", pattern=rf"(?:(?:\d\s*){4}(?:\d+\.\d+\s*){2})*")
    # Specify atom and X, XY, XYZ, ... to be ranged
    ranged_cartesian: str = Field(default="", pattern=rf"(?:\d\s+{CART_STR}\s+(?:\d+\.\d+\s*){2})*")

    # Specify distances for which extra force will be added
    ext_force_distance: str = Field(default="", pattern=rf"(?:(?:\d\s*){2}\(.*?\))*")
    # Specify angles for which extra force will be added
    ext_force_bend: str = Field(default="", pattern=rf"(?:(?:\d\s*){3}\(.*?\))*")
    # Specify dihedral angles for which extra force will be added
    ext_force_dihedral: str = Field(default="", pattern=rf"(?:(?:\d\s*){4}\(.*?\))*")
    # Specify out-of-plane angles for which extra force will be added
    ext_force_oofp: str = Field(default="", pattern=rf"(?:(?:\d\s*){4}\(.*?\))*")
    # Specify cartesian coordinates for which extra force will be added
    ext_force_cartesian: str = Field(default="", pattern=rf"(?:(?:\d\s+{CART_STR})\(.*?\))*")

    # Should an xyz trajectory file be kept (useful for visualization)?
    # P.print_trajectory_xyz = uod.get('PRINT_TRAJECTORY_XYZ', False)
    # Symmetry tolerance for testing whether a mode is symmetric.
    # P.symm_tol("SYMM_TOL", 0.05)
    #
    # SUBSECTION Convergence Control.
    # Set of optimization criteria. Specification of any MAX_*_G_CONVERGENCE
    # RMS_*_G_CONVERGENCE options will append to overwrite the criteria set here
    # |optking__flexible_g_convergence| is also on.
    # See Table :ref:`Geometry Convergence <table:optkingconv>` for details.
    g_convergence: str = Field(
        pattern=re.compile(
            r"QCHEM|MOLPRO|GAU|GAU_LOOSE|GAU_TIGHT|GAU_VERYTIGHT|TURBOMOLE|CFOUR|NWCHEM_LOOSE|INTERFRAG_TIGHT",
            flags=re.IGNORECASE,
        ),
        default="QCHEM",
    )

    # _conv_rms_force = -1
    # _conv_rms_disp = -1
    # _conv_max_DE = -1
    # _conv_max_force = -1
    # _conv_max_disp = -1
    # Convergence criterion for geometry optmization: maximum force (internal coordinates, au)
    conv_max_force: float = Field(default=3.0e-4, alias="MAX_FORCE_G_CONVERGENCE")
    # Convergence criterion for geometry optmization: rms force  (internal coordinates, au)
    conv_rms_force: float = Field(default=3.0e-4, alias="RMS_FORCE_G_CONVERGENCE")
    # Convergence criterion for geometry optmization: maximum energy change
    conv_max_DE: float = Field(default=1.0e-6, alias="MAX_ENERGY_G_CONVERGENCE")
    # Convergence criterion for geometry optmization:
    # maximum displacement (internal coordinates, au)
    conv_max_disp: float = Field(default=1.2e-3, alias="MAX_DISP_G_CONVERGENCE")
    # Convergence criterion for geometry optmization:
    # rms displacement (internal coordinates, au)
    conv_rms_disp: float = Field(default=1.2e-3, alias="RMS_DISP_G_CONVERGENCE")
    # Even if a user-defined threshold is set, allow for normal, flexible convergence criteria
    flexible_g_convergence: bool = False

    #
    # SUBSECTION Hessian Update
    # Hessian update scheme
    hess_update: str = Field(pattern=r"NONE|BFGS|MS|POWELL|BOFILL", default="BFGS")

    # Number of previous steps to use in Hessian update, 0 uses all
    hess_update_use_last: int = Field(ge=0, default=4)
    # Do limit the magnitude of changes caused by the Hessian update?
    hess_update_limit: bool = True
    # If |hess_update_limit| is True, changes to the Hessian from the update are limited
    # to the larger of |hess_update_limit_scale| * (current value) and
    # |hess_update_limit_max| [au].  By default, a Hessian value cannot be changed by more
    # than 50% and 1 au.
    hess_update_limit_max: float = Field(ge=0.0, default=1.00)
    hess_update_limit_scale: float = Field(ge=0.0, le=1.0, default=0.50)

    # Denominator check for hessian update.
    hess_update_den_tol: float = Field(gt=0.0, default=1e-7)

    # Hessian update is avoided if any internal coordinate has changed by
    # more than this in radians/au
    hess_update_dq_tol: float = Field(ge=0.0, default=0.5)

    # SUBSECTION Using external Hessians
    # Do read Cartesian Hessian?  Only for experts - use
    # |optking__full_hess_every| instead.
    cart_hess_read: bool = False
    # accompanies cart_hess_read. The default is not validated
    # Need two options here because str_to_upper cannot be turned of for members of the Model
    # _hessian_file avoids str_to_upper
    hessian_file: pathlib.Path = Field(default=pathlib.Path(""), validate_default=False)
    _hessian_file = pathlib.Path("")

    # Frequency with which to compute the full Hessian in the course
    # of a geometry optimization. 0 means to compute the initial Hessian only,
    # 1 means recompute every step, and N means recompute every N steps. The
    # default (-1) is to never compute the full Hessian.
    full_hess_every: int = Field(ge=-1, default=-1)

    # Model Hessian to guess intrafragment force constants
    intrafrag_hess: str = Field(
        pattern=re.compile(r"SCHLEGEL|FISCHER|SIMPLE|LINDH|LINDH_SIMPLE", flags=re.IGNORECASE),
        default="SCHLEGEL",
    )
    # Re-estimate the Hessian at every step, i.e., ignore the currently stored Hessian.
    h_guess_every: bool = False
    _working_steps_since_last_H = 0

    #
    # SUBSECTION Backtransformation to Cartesian Coordinates Control
    bt_max_iter: int = Field(gt=0, default=25)
    bt_dx_conv: float = Field(gt=0.0, default=1.0e-7)
    bt_dx_rms_change_conv: float = Field(gt=0.0, default=1.0e-12)
    # The following should be used whenever redundancies in the coordinates
    # are removed, in particular when forces and Hessian are projected and
    # in back-transformation from delta(q) to delta(x).
    bt_pinv_rcond: float = Field(gt=0.0, default=1.0e-6)

    #
    # For multi-fragment molecules, treat as single bonded molecule or via interfragment
    # coordinates. A primary difference is that in ``MULTI`` mode, the interfragment
    # coordinates are not redundant.
    frag_mode: str = Field(
        pattern=re.compile(r"SINGLE|MULTI", flags=re.IGNORECASE), default="SINGLE"
    )
    # Which atoms define the reference points for interfragment coordinates?
    frag_ref_atoms: list[list[list[int]]] = []
    # Do freeze all fragments rigid?
    freeze_intrafrag: bool = False
    # Do freeze all interfragment modes?
    # P.inter_frag = uod.get('FREEZE_INTERFRAG', False)
    # When interfragment coordinates are present, use as reference points either
    # principal axes or fixed linear combinations of atoms.
    interfrag_mode: str = Field(
        pattern=re.compile(r"FIXED|PRINCIPAL_AXES", re.IGNORECASE), default="FIXED"
    )

    # Do add bond coordinates at nearby atoms for non-bonded systems?
    add_auxiliary_bonds: bool = False
    # This factor times standard covalent distance is used to add extra stretch coordinates.
    auxiliary_bond_factor: float = Field(gt=1.0, default=2.5)
    # Do use 1/R for the interfragment stretching coordinate instead of R?
    interfrag_dist_inv: bool = False
    # Used for determining which atoms in a system are too collinear to
    # be chosen as default reference atoms. We avoid collinearity. Greater
    # is more restrictive.
    interfrag_collinear_tol: float = Field(gt=0.0, default=0.01)

    # Let the user submit a dictionary (or array of dictionaries) for
    # the interfrag coordinates. Validation occurs below
    interfrag_coords: list[dict] = []

    # Model Hessian to guess interfragment force constants
    interfrag_hess: str = Field(
        pattern=re.compile(r"DEFAULT|FISCHER_LIKE", flags=re.IGNORECASE), default="DEFAULT"
    )
    # P.interfrag_hess = uod.get('INTERFRAG_HESS', 'DEFAULT')
    # When determining connectivity, a bond is assigned if interatomic distance
    # is less than (this number) * sum of covalent radii.
    covalent_connect: float = Field(gt=0.0, default=1.3)
    # When connecting disparate fragments when frag_mode = SIMPLE, a "bond"
    # is assigned if interatomic distance is less than (this number) * sum of covalent radii.
    # The value is then increased until all the fragments are connected directly
    # or indirectly.
    interfragment_connect: float = Field(gt=0.0, default=1.8)
    # General, maximum distance for the definition of H-bonds.
    h_bond_connect: float = Field(gt=0.0, default=4.3)
    # Add out-of-plane angles (usually not needed)
    include_oofp: bool = False

    #
    #
    # SUBSECTION Misc.
    # Do save and print the geometry from the last projected step at the end
    # of a geometry optimization? Otherwise (and by default), save and print
    # the previous geometry at which was computed the gradient that satisfied
    # the convergence criteria.
    # P.final_geom_write = uod.get('FINAL_GEOM_WRITE', False)
    # Do test B matrix?
    test_B: bool = False
    # Do test derivative B matrix?
    test_derivative_B: bool = False
    # Only generate the internal coordinates and then stop (boolean) UNUSED
    # generate_intcos_exit: bool = False
    # Keep internal coordinate definition file.
    # keep_intcos: bool = False UNUSED
    linesearch_step: float = Field(gt=0.0, default=0.100)
    linesearch: bool = False
    # Guess at Hessian in steepest-descent direction.
    sd_hessian: float = Field(gt=0.0, default=1.0)

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

    # --- SET INTERNAL OPTIMIZATION PARAMETERS ---
    _i_max_force = False
    _i_rms_force = False
    _i_max_DE = False
    _i_max_disp = False
    _i_rms_disp = False
    _i_untampered = False

    def to_dict(self):
        """ Specialized form of __dict__. Makes sure to include convergence keys that are hidden """
        save = self.model_dump()
        include = {
            "_i_max_force": self._i_max_force,
            "_i_rms_force": self._i_rms_force,
            "_i_max_DE": self._i_max_DE,
            "_i_max_disp": self._i_max_disp,
            "_i_rms_disp": self._i_rms_disp,
            "_i_untampered": self._i_untampered,
        }
        for key in include:
            save.update(include)
        return save

    def __str__(self):
        s = "\n\t\t -- Optimization Parameters --\n"
        for name, value in self.model_dump(by_alias=True).items():
            s += "\t%-30s = %15s\n" % (name, value)
        s += "\n"
        return s

    @model_validator(mode="before")
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

    @model_validator(mode="after")
    def validate_algorithm(self):
        """Ensure that if the user has selected both an opt_type and step_type that they are
        compatible. If the user has selected `opt_type=TS` OR a `step_type` consistent with `TS`
        then change the other keyword to have the appropriate keyword"""

        min_step_types = ["RFO", "NR", "SD", "CONJUGATE", "LINESEARCH"]
        ts_step_types = ["RS_I_RFO", "P_RFO"]

        if "OPT_TYPE" in self._raw_input and "STEP_TYPE" in self._raw_input:
            if self.opt_type == "TS":
                assert self.step_type not in min_step_types
            elif self.opt_type == "MIN":
                assert self.step_type not in ts_step_types
        elif "OPT_TYPE" in self._raw_input and self.opt_type == "TS":
            # User has selected TS. Change algorithm to RS_I_RFO
            self.step_type = "RS_I_RFO"
        elif "STEP_TYPE" in self._raw_input and "STEP_TYPE" in ts_step_types:
            self.opt_type = "TS"
        return self

    @model_validator(mode="after")
    def validate_convergence(self):
        """Set active variables depending upon the PRESET that has been provided and whether any
        specific values were individually specified by the user."""

        # stash so that __setattr__ doesn't affect which variables have been changed
        # Start by setting each individual convergence option from preset
        conv_spec = CONVERGENCE_PRESETS.get(self.g_convergence)

        for key, val in conv_spec.items():
            self.__setattr__(key, val)

        # Table to easily correlate the user / psi4 name, internal keyword name,
        # and internal active flag for keyword
        keywords = [
            ("MAX_FORCE_G_CONVERGENCE", "conv_max_force", "_i_max_force"),
            ("RMS_FORCE_G_CONVERGENCE", "conv_rms_force", "_i_rms_force"),
            ("MAX_ENERGY_G_CONVERGENCE", "conv_max_DE", "_i_max_DE"),
            ("MAX_DISP_G_CONVERGENCE", "conv_max_disp", "_i_max_disp"),
            ("RMS_DISP_G_CONVERGENCE", "conv_rms_disp", "_i_rms_disp"),
        ]

        # if ANY convergence options were specified by the user,  turn untampered on and set all
        # options to inactive
        for keyword_set in keywords:
            if keyword_set[0] in self._raw_input:
                # mark keyword as "active" through _i_keyword variable
                # mark untampered as False (tampering has occured!)
                self.__setattr__(keyword_set[1], self._raw_input.get(keyword_set[0]))
                self.__setattr__(keyword_set[2], True)
                self._i_untampered = False
                if self.flexible_g_convergence:
                    # use flexible conv criteria don't leave criteria preset active except for mods
                    self._i_untampered = True
                else:
                    self._i_untampered = False
        return self

    @model_validator(mode="after")
    def validate_iter(self):
        if self.opt_type == "IRC" and "GEOM_MAXITER" not in self._raw_input:
            self.geom_maxiter = self.irc_points * 15
        elif self.geom_maxiter < self.alg_geom_maxiter:
            self.alg_geom_maxiter = self.geom_maxiter
        return self

    @model_validator(mode="after")
    def validate_trustregion(self):
        # Initial Hessian guess for cartesians with coordinates BOTH is stupid, so don't scale
        #   step size down too much.  Steepest descent has no good hessian either.
        if "INTRAFRAG_TRUST_MIN" not in self._raw_input:
            if self.opt_coordinates == "BOTH":
                self.intrafrag_trust_min = self.intrafrag_trust / 2.0
            elif self.step_type == "SD":  # steepest descent, use constant stepsize
                self.intrafrag_trust_min = self.intrafrag_trust
            elif any(
                [
                    self.ext_force_distance,
                    self.ext_force_bend,
                    self.ext_force_dihedral,
                    self.ext_force_oofp,
                    self.ext_force_cartesian,
                ]
            ):
                # with external forces, the check for trust radius will be inapt
                # so don't let minimum step get shrunk too much.
                self.intrafrag_trust_min = self.intrafrag_trust / 2.0

        if self.opt_type in ["IRC", "TS"] and "INTRAFRAG_STEP_LIMIT" not in self._raw_input:
            self.intrafrag_trust = 0.2  # start with smaller intrafrag_trust

        if self.intrafrag_trust_max < self.intrafrag_trust:
            self.intrafrag_trust = self.intrafrag_trust_max
        return self

    @model_validator(mode="after")
    def validate_hessian(self):
        set_vars = self._raw_input

        # Original Lindh specification was to redo at every step.
        if "h_guess_every" not in set_vars and self.intrafrag_hess == "LINDH":
            self.h_guess_every = True

        # Default for cartesians: use Lindh force field for initial guess, then BFGS.
        if self.opt_coordinates == "CARTESIAN":
            if "intrafrag_hess" not in set_vars:
                self.intrafrag_hess = "LINDH"
                if "h_guess_every" not in set_vars:
                    self.h_guess_every = False

        # Set Bofill as default for TS optimizations.
        if self.opt_type == "TS" or self.opt_type == "IRC":
            if "hess_update" not in set_vars:
                self.hess_update = "BOFILL"

        # Make trajectory file printing the default for IRC.
        if self.opt_type == "IRC" and "print_trajectory_xyz_file" not in set_vars:
            self.print_trajectory_xyz_file = True

        # Read cartesian Hessian by default for IRC.
        # Changed to turn cart_hess_read on only if a file path was provided.
        # otherwise full_hess_every will handle providing hessian
        if self.opt_type == "IRC" and "cart_hess_read" not in set_vars:
            if self._hessian_file != pathlib.Path(""):
                self.cart_hess_read = True

        # inactive option
        # if self.generate_intcos_exit:
        #     self.keep_intcos = True

        # For IRC, we WILL need a Hessian.  Compute it if not provided.
        # Set full_hess_every to 0 if -1
        if self.opt_type == "IRC" and self.full_hess_every < 0:
            self.full_hess_every = 0
            # self.cart_hess_read = True  # not sure about this one - test

        # if steepest-descent, then make much larger default
        if self.step_type == "SD" and "consecutive_backsteps" not in set_vars:
            self.consecutive_backsteps_allowed = 10

        # For RFO step, eigenvectors of augmented Hessian are divided by the last
        # element unless it is smaller than this value {double}.  Can be used to
        # eliminate asymmetric steps not otherwise detected (e.g. in degenerate
        # point groups). For multi-fragment modes, we presume that smaller
        # Delta-E's are possible, and this threshold should be made larger.
        # if P.fragment_mode == 'MULTI' and 'RFO_NORMALIZATION_MAX' not in uod:
        #     P.rfo_normalization_max = 1.0e5
        # If arbitrary user forces, don't shrink step_size if Delta(E) is poor.
        return self

    @model_validator(mode="after")
    def validate_hessian_file(self):
        # Stash value of hessian_file in _hessian_file for internal use
        # mode before required so that we stash before str_to_upper is called
        orig_vars = self._raw_input
        if orig_vars.get("HESSIAN_FILE"):
            self._hessian_file = pathlib.Path(orig_vars.get("HESSIAN_FILE"))
        return self

    @model_validator(mode="after")
    def validate_frag(self):
        # Finish multifragment option setup by forcing frag_mode: MULTI if DimerCoords are provided

        input = self.interfrag_coords
        if input:
            # if interfrag_coords is not empty. Consider whether it is just [{}]
            if isinstance(input, list) and len(input) > 0:
                if isinstance(input[0], dict) and len(input[0]) == 0:
                    # empty dict in list
                    return self
            self.frag_mode = "MULTI"
        return self

    @field_validator("interfrag_coords", mode="before")
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
                    assert InterfragCoords.model_validate_json(item)

            # Now that everything is validated. Convert to dict for storage
            tmp = [json.loads(item) for item in tmp]
            return tmp
        else:
            return [{}]

    @model_validator(mode="after")
    def validate_case(self):
        for attr in self.model_dump():
            if isinstance(self.__getattribute__(attr), str):
                self.__setattr__(attr, self.__getattribute__(attr).upper())
        return self

    @classmethod
    def from_internal_dict(cls, params):
        """Assumes that params does not use the input key and syntax, but uses the internal names and
        internal syntax. Meant to be used for recreating options object after dump to dict
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
        """Returns the currently active values for each convegence criteria. Not the original
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
Params = OptParams(**{})

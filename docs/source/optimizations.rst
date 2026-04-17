.. _`start`:

Getting Started
===============

OptKing (also known as pyoptking) is a rewrite of the c++ OptKing module in psi4 to enable future development
and for use with recent interoperability efforts (e.g. MolSSI QCArchive and QCDB). OptKing is focused on
optimization of molecular geometries: finding minima, transition states, and reaction paths. Current work
is focused especially on expanding the reaction path methods.

Installation and Setup
~~~~~~~~~~~~~~~~~~~~~~

The recommended method of installation is through conda. To install::
    
    conda install optking -c conda-forge

The project is hosted on `github <https://github.com/psi-rking/optking/>`_ Source code can be downloaded with
git or with the tarballs provided under `releases <https://github.com/psi-rking/optking/releases/latest>`_.

To install from source make sure all dependencies are installed via conda or pip and run::

    pip install -e .

from the installation directory.

Dependencies
------------

To run optking without ``QCEngine`` or ``Psi4``, the ``CustomHelper`` class may be used though the python API.
This ``Helper`` allows for the use of arbitrary packages and/or modified gradients to be used.
If even finer grain control is needed an ``OptimizationManager`` class can be used - this is not
likely to be the case unless implementing a new, complex optimization algorithm.
Gradients, energies, and possibly hessians can be provided directly.
To use the most basic representation of the algorithms with no reference to molecules one of the classes
inheriting from OptimizationAlgorithm will be needed.

Otherwise (and for most use cases), ``QCEngine`` and your QC/MM program of choice OR ``Psi4`` is required.
If using QCEngine see `Install QCEngine <https://molssi.github.io/QCEngine/install.html>`__. 
to ensure proper setup. Any QC or MM programs will need to be installed such that ``QCEngine`` can find them.

.. _`qcengine_running`:

Running Through QCEngine
~~~~~~~~~~~~~~~~~~~~~~~~

A basic driver has been implemented in ``QCEngine``. ``QCEngine`` is built upon ``QCElemental`` which provides input
validation and standardized input/output. To see the requirements for an ``OptimziationInput`` check MolSSI's
`qcelemental documentation <https://molssi.github.io/QCElemental/>`_. NOTE ``QCElemental`` assumes atomic units by default:

As of optking v0.5, it can run either QCSchema v1 or v2. For
direct optking runs, input v1 emits v1 and input v2 emits v2. For
qcengine.compute(), the same is true, but one can also pass a
``return_version`` to return 2 from 1 or 1 from 2. Note that due to
Pydantic restrictions, v1 is never available for Python 3.14+. See
https://molssi.github.io/QCElemental/next/models.html#qcschema-v2 for
conversion resources.

This is QCSchema v1.

.. code-block:: python

    import qcengine as qcng

    opt_input = { 
        "initial_molecule": {
            "symbols": ["O", "O", "H", "H"],
            "geometry": [
                0.0000000000,
                0.0000000000,
                0.0000000000,
                -0.0000000000,
                -0.0000000000,
                2.7463569188,
                1.3013018774,
                -1.2902977124,
                2.9574871774,
                -1.3013018774,
                1.2902977124,
                -0.2111302586,
            ],  
            "fix_com": True,
            "fix_orientation": True,
        },  
        "input_specification": {
            "model": {"method": "hf", "basis": "sto-3g"},
            "driver": "gradient",
            "keywords": {"d_convergence": "1e-7"},
        },  
        "keywords": {"g_convergence": "GAU_TIGHT", "program": "psi4"},
    }

    result = qcng.compute_procedure(opt_input, "optking")

And this is QCSchema v2.

.. code-block:: python

    import qcengine as qcng

    opt_input = {
        "initial_molecule": {
            "symbols": ["O", "O", "H", "H"],
            "geometry": [
                0.0000000000,
                0.0000000000,
                0.0000000000,
                -0.0000000000,
                -0.0000000000,
                2.7463569188,
                1.3013018774,
                -1.2902977124,
                2.9574871774,
                -1.3013018774,
                1.2902977124,
                -0.2111302586,
            ],
            "fix_com": True,
            "fix_orientation": True,
        },
        "specification": {
            "specification": {
                "model": {"method": "hf", "basis": "sto-3g"},
                "driver": "gradient",
                "keywords": {"d_convergence": "1e-7"},
            },
            "keywords": {"g_convergence": "GAU_TIGHT", "program": "psi4"},
        },
    }

    result = qcng.compute(opt_input, "optking")

An explicit example of creating and running an OptimizationInput. Note: Molecule.from_data seems to be the only
place Angstroms are expected:

This is QCSchema v1.

.. code-block:: python

    import qcengine as qcng

    from qcelemental.models import Molecule, OptimizationInput
    from qcelemental.models.common_models import Model
    from qcelemental.models.procedures import QCInputSpecification
    
    # WARNING. The user MUST set fix_com and fix_orientation to True.
    # optimization will almost certainly fail otherwise
    molecule = Molecule.from_data(
        """ 
        O        0.0000000000      0.0000000000      0.0000000000
        O       -0.0000000000     -0.0000000000      1.4533095991
        H        0.6886193476     -0.6827961938      1.5650349285
        H       -0.6886193476      0.6827961938     -0.1117253294""",
        fix_com=True,
        fix_orientation=True,
    )
    
    model = Model(method="hf", basis="sto-3g")
    input_spec = QCInputSpecification(
        driver="gradient", model=model, keywords={"d_convergence": 1e-7}  # QC program options
    )
    
    opt_input = OptimizationInput(
        initial_molecule=molecule,
        input_specification=input_spec,
        keywords={"g_convergence": "GAU_TIGHT", "program": "psi4"},  # optimizer options
    )
    
    config = qcng.get_config()  # get machine info (e.g. number of cores) can specify explicitly
    opt = qcng.get_procedure("optking")
    result = opt.compute(opt_input, config)

This is QCSchema v2.

.. code-block:: python

    import qcengine as qcng

    from qcelemental.models.v2 import Molecule, OptimizationInput, Model, AtomicSpecification

    # WARNING. The user MUST set fix_com and fix_orientation to True.
    # optimization will almost certainly fail otherwise
    molecule = Molecule.from_data(
        """
        O        0.0000000000      0.0000000000      0.0000000000
        O       -0.0000000000     -0.0000000000      1.4533095991
        H        0.6886193476     -0.6827961938      1.5650349285
        H       -0.6886193476      0.6827961938     -0.1117253294""",
        fix_com=True,
        fix_orientation=True,
    )

    model = Model(method="hf", basis="sto-3g")
    input_spec = AtomicSpecification(
        driver="gradient", model=model, keywords={"d_convergence": 1e-7}  # QC program options
    )

    opt_input = OptimizationInput(
        initial_molecule=molecule,
        specification={
            "specification": input_spec,
            "keywords": {"g_convergence": "GAU_TIGHT", "program": "psi4"},  # optimizer options
        },
    )

    config = qcng.get_config()  # get machine info (e.g. number of cores) can specify explicitly
    opt = qcng.get_procedure("optking")
    result = opt.compute(opt_input, config)

Running through Psi4
~~~~~~~~~~~~~~~~~~~~

pyoptking replaced the c++ OptKing module in Psi4 as of Psi4 1.7. To run an optimization, simply
call ``optimize()``::

    molecule hooh {
        0 1
        O        0.0000000000      0.0000000000      0.0000000000
        O       -0.0000000000     -0.0000000000      1.4533095991
        H        0.6886193476     -0.6827961938      1.5650349285
        H       -0.6886193476      0.6827961938     -0.1117253294
    }

    set {
        d_convergence 1e-7
        g_convergence GAU_TIGHT
    }

    optimize("hf/sto-3g")

.. note::
  As of ``v1.9``, Psi4 maintains its own list of options corresponding to OptKing's options. If an
  Optking is not available in your version of Psi4, please update your version of Psi4. Alternatively,
  options can be passed directly to the optimizer through ``optimizer_keywords``.
  See `psi4.driver.optimize <https://psicode.org/psi4manual/master/api/psi4.driver.optimize.html#psi4.driver.optimize>`__.

Running through an OptHelper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For users looking to run optimizations from python, an examples of QCEngine's python API has already been shown.
To run optking through Psi4's python API checkout the `Psi4 API docs <https://psicode.org/psi4manual/master/psiapi.html>`__.
These two options should be sufficient for the majority of users.

If direct control over the optimizer is desired two ``OptHelper`` classes are provided to streamline performing an optimization.
The molecular system, optimization coordinates, history, etc are all accessible through their respective classes and may be accessed
as attributes of the OptHelper instance.
``EngineHelper`` takes an ``OptimizationHelper`` and calls ``qcengine.compute()`` to perform basic calculations with the provided ``input_specification``
``CustomHelper`` accepts ``QCElemental`` and ``Psi4`` molecules while requiring user provided gradients, energies, and possibly hessians. This may
be useful for implementing a custom optimization driver or procedure using OptKing.

``EngineHelper``:

This is QCSchema v1.

.. code-block:: python

    import qcengine as qcng

    from qcelemental.models import Molecule, OptimizationInput
    from qcelemental.models.common_models import Model
    from qcelemental.models.procedures import QCInputSpecification


    molecule = Molecule.from_data(
        """ 
        O        0.0000000000      0.0000000000      0.0000000000
        O       -0.0000000000     -0.0000000000      1.4533095991
        H        0.6886193476     -0.6827961938      1.5650349285
        H       -0.6886193476      0.6827961938     -0.1117253294""",
        fix_com=True,
        fix_orientation=True,
    )
    
    model = Model(method="hf", basis="sto-3g")
    input_spec = QCInputSpecification(
        driver="gradient", model=model, keywords={"d_convergence": 1e-7}  # QC program options
    )
    
    opt_input = OptimizationInput(
        initial_molecule=molecule,
        input_specification=input_spec,
        keywords={"g_convergence": "GAU_TIGHT", "program": "psi4"},  # optimizer options
    )

    opt = optking.EngineHelper(opt_input)
    
    for step in range(30):

        # Compute one's own energy and gradient
        opt.compute() # process input. Get ready to take a step
        opt.take_step() 
        
        conv = opt.test_convergence()

        if conv is True:
            print("Optimization SUCCESS:")
        else:
            print("Optimization FAILURE:\n")

    json_output = opt.close() # create an unvalidated OptimizationOutput like object
    E = json_output["energies"][-1]

This is QCSchema v2.

.. code-block:: python

    import optking
    import qcengine as qcng

    from qcelemental.models.v2 import Molecule, OptimizationInput, Model, AtomicSpecification

    # WARNING. The user MUST set fix_com and fix_orientation to True.
    # optimization will almost certainly fail otherwise
    molecule = Molecule.from_data(
        """
        O        0.0000000000      0.0000000000      0.0000000000
        O       -0.0000000000     -0.0000000000      1.4533095991
        H        0.6886193476     -0.6827961938      1.5650349285
        H       -0.6886193476      0.6827961938     -0.1117253294""",
        fix_com=True,
        fix_orientation=True,
    )

    opt_input = OptimizationInput(
        initial_molecule=molecule,
        specification={
            "specification": {
                "driver": "gradient",
                "model": Model(method="hf", basis="sto-3g"),
                "keywords": {"d_convergence": 1e-7}  # QC program options
            },
            "keywords": {"g_convergence": "GAU_TIGHT", "program": "psi4"},  # optimizer options
        },
    )

    opt = optking.EngineHelper(opt_input)

    for step in range(30):

        # Compute one's own energy and gradient
        opt.compute() # process input. Get ready to take a step
        opt.take_step()

        conv = opt.test_convergence()

        if conv is True:
            print("Optimization SUCCESS:")
        else:
            print("Optimization FAILURE:\n")

    json_output = opt.close() # create an unvalidated OptimizationOutput like object
    E = json_output["trajectory_properties"][-1]["return_energy"]

``CustomHelper`` can take ``psi4`` or ``qcelemental`` molecules. A simple example of a custom optimization loop is
shown where the gradients are provided from a simple lennard jones potential:

.. code-block:: python

    import psi4
    import optking

    h2o = psi4.geometry(
    """
     O
     H 1 1.0
     H 1 1.0 2 104.5
    """
    )

    psi4_options = {
        "basis": "sto-3g",
    }
    optking_options = {"g_convergence": "gau_verytight", "intrafrag_hess": "SIMPLE"}

    psi4.set_options(psi4_options)

    opt = optking.CustomHelper(h2o, optking_options)

    for step in range(130):

        # Compute one's own energy and gradient
        E, gX = optking.lj_functions.calc_energy_and_gradient(opt.geom, 2.5, 0.01, True)
        # Insert these values into the 'user' computer.
        opt.E = E
        opt.gX = gX

        opt.compute() # process input. Get ready to take a step
        opt.take_step()

        conv = opt.test_convergence()

        if conv is True:
            print("Optimization SUCCESS:")
            break
    else:
        print("Optimization FAILURE:\n")

    json_output = opt.close() # create an unvalidated OptimizationOutput like object
    E = json_output["energies"][-1]


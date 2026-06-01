Build and Testing
=================

Optking uses `hatch: <https://hatch.pypa.io/dev/>` for its build system to simplify installing,
publishing, testing, doc generation, and style enforcement.

Installing
----------

To install optking obtain source code from github `OptKing <https://github.com/psi-rking>`. Please
consider forking if opening a PR.

To install optking in development mode into an existing environment::

	pip install -e .

Optking uses hatch-vcs and hatch-conda to link hatch with our git version tags and to create conda
environments for testing::

	pip install pipx
	pipx install hatch
	pipx inject hatch hatch-vcs hatch-conda  # install build requirements into hatch venv

Testing
-------

pytest can be used on its own as normal; however a set of tests has also been configured in `hatch`
These tests should be run from the projects root directory.

A default testsuite run can be triggered with::

	hatch test optking/tests

A conda environment will be automatically created with python 3.10, pydantic 1.10.17, Psi4, and
required dependencies to run the tests in. Long tests will be exlcuded. This should take a few
minutes.

Before creating a PR it is recommended that all tests be run::

	hatch test --all optking/tests

This will run OptKing's tests with multiple python and pydantic versions, long tests and Psi4's
optimization tests will be included for python 3.13 and pydantic 2. These same tests will be run by
optking's github workflows. This may take up to half and hour. On my workstation about 20 minutes.
If running for the first time, up to four conda environments will be made for the various test runs.
If `hatch test --all` passes OptKing's and Psi4's CI should pass.

Building and Publishing
-----------------------

Before building, create a new version tag with git tag "<major>.<minor>.<micro>". See the up-to-date
PEP 440
`version spec <https://packaging.python.org/en/latest/specifications/version-specifiers/#version-specifiers>`
for more information. Tags should be pushed to github with `git push -u <main-repo> --tags` when
the final PR has been accepted for a given version.

Hatch can build and publish to pypi with `hatch build` and `hatch publish`. Conda should autodetect
that a new release was published and auto-generate a PR::

	git tag -a "<new_version>" -m "message"
	hatch build
	hatch publish -r test  # publish to pypi test

In general, the git tags can simply be pushed to GitHub and the release workflow will handle
building and publishing. `hatch publish` is only needed for tests.

Docs
----

Hatch will install dependencies for creating documentation by running either `hatch run docs:build`
or `hatch run docs:serve`. The former will simply build the html files required for the documentation,
while the latter will build the docs, open the default web browser to view the docs, and rebuild
the docs whenever changes to `source/` are detected.

Docs are built automatically by readthedocs for any PRs.

Style
-----

hatch also defines format and lint commands that can be run with::

	hatch fmt -l --check  # run linting check with flake8 
	hatch fmt -f --check  # run format check with black

Omitting --check from `hatch fmt -f` will exectute changes.

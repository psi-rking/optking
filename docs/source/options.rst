Keywords and Options
====================

Optking uses pydantic to validate user input for types, values, and, when strings are expected,
for content with regexes. Brief descriptions of each option is presented below.

For developers or users interacting with optking's internals, Optking historically has utilized
a single global options dictionary. Some parts of Optking still utilize this global dictionary
Most of the current code's classes and functions; however, accept local options by passing a
dictionary, options object, or explicit parameters. Note - the options reported here are the names
that the user should use when providing keywords to Optking. For developers, Optking may use
different names internally. For starters, variables (with the exception of matrices) should
adhere to PEP8 (lower snake-case).

Alphabetized Keywords
=====================

.. autopydantic_model:: optking.optparams.OptParams

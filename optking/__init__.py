import logging.config
import logging
import os
import sys

from .optimize import optimize
from . import lj_functions
import loggingconfig

print(os.getcwd())

try:
    with open(os.path.join(os.getcwd(), sys.argv[0][:-3]) + ".out", "r+") as output_file:
        print(os.path.join(os.getcwd(), sys.argv[0][:-3]) + ".out")
        output_file.seek(0)
        output_file.truncate()
except FileNotFoundError:
    pass

logging.config.dictConfig(loggingconfig.logging_configuration)
logger = logging.getLogger(__name__)

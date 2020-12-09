import argparse
import os
import shutil
import subprocess as sp

# Args
parser = argparse.ArgumentParser(
    description="Creates a conda environment from file for a given Python version."
)

parser.add_argument(
    "-n", "--name", type=str, nargs=1, help="The name of the created Python environment"
)

args = parser.parse_args()

with open('../conda-envs/dev.yaml', "r") as handle:
    script = handle.read()

tmp_file = "tmp_env.yaml"

with open(tmp_file, "w") as handle:
    handle.write(script)

conda_path = shutil.which("conda")

sp.call(
    "{} env create -n {} -f {}".format(conda_path, args.name[0], tmp_file), shell=True
)
os.unlink(tmp_file)


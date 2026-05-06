import subprocess
import sys
import os

# Super simple script for running tests via scratch
p1 = subprocess.Popen(["pytest", "-rv", "--check_iter=1", "--show-capture=stderr", "--cov=optking", "--color=yes", "--cov-report=xml", "optking/"] + sys.argv[1:], env=os.environ)
p1_ec = p1.wait()
p2 = subprocess.Popen(["psi4", "--test", "-m", "opt"], env=os.environ)
p2_ec = p2.wait()

if p1_ec == 0 and p2_ec == 0:
    code = 0
else:
    code = max(p1_ec, p2_ec)
sys.exit(code)


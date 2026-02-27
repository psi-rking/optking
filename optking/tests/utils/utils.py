import optking
from packaging.version import Version
import sys

def compare_iterations(json_output, expected_steps, assert_iter):
    logger = optking.logger
    try:
        steps_taken = len(json_output["trajectory"])  # OptimizationResult.schema_version = 1
    except KeyError:
        steps_taken = len(json_output["trajectory_results"])  # OptimizationResult.schema_version = 2
    if steps_taken != expected_steps:
        logger.warning(
            f"TEST - Number of steps taken {steps_taken}, Previous required steps {expected_steps}"
        )
    else:
        logger.info(f"TEST - Number of steps taken matches expected {steps_taken}")

    if int(assert_iter):
        try:
            assert steps_taken == expected_steps
        except AssertionError:
            if steps_taken < expected_steps:
                case = "fewer"
            else:
                case = "more"
            print(f"Test required {case} steps than expected. Expected: {expected_steps} Actual: {steps_taken}", file=sys.stderr)
            raise

def psi4_runs_v2_qcschema(p4ver):
    return Version(p4ver) >= Version("1.11a1.dev2")

def qcel_impl_v2_qcschema():
    try:
        from qcelemental.models import v2
    except ImportError:
        return False
    else:
        return True

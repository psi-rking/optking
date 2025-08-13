import optking
import sys

def compare_iterations(json_output, expected_steps, assert_iter):
    logger = optking.logger
    steps_taken = len(json_output["trajectory"])
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

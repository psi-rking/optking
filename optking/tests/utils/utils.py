import optking


def compare_iterations(json_output, expected_steps, assert_iter):

    logger = optking.logger
    steps_taken = len(json_output["trajectory"])
    if steps_taken != expected_steps:
        logger.warning(f"TEST - Number of steps taken {steps_taken}, Previous required steps {expected_steps}")
    else:
        logger.info(f"TEST - Number of steps taken matches expected {steps_taken}")

    if int(assert_iter):
        assert steps_taken == expected_steps

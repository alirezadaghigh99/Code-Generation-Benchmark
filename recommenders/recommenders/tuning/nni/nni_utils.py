def check_experiment_status(wait=WAITING_TIME, max_retries=MAX_RETRIES):
    """Checks the status of the current experiment on the NNI REST endpoint.

    Waits until the tuning has completed.

    Args:
        wait (numeric) : time to wait in seconds
        max_retries (int): max number of retries
    """
    i = 0
    while i < max_retries:
        nni_status = get_experiment_status(NNI_STATUS_URL)
        if nni_status["status"] in ["DONE", "TUNER_NO_MORE_TRIAL"]:
            break
        elif nni_status["status"] not in ["RUNNING", "NO_MORE_TRIAL"]:
            raise RuntimeError(
                "NNI experiment failed to complete with status {} - {}".format(
                    nni_status["status"], nni_status["errors"][0]
                )
            )
        time.sleep(wait)
        i += 1
    if i == max_retries:
        raise TimeoutError("check_experiment_status() timed out")

def check_stopped(wait=WAITING_TIME, max_retries=MAX_RETRIES):
    """Checks that there is no NNI experiment active (the URL is not accessible).
    This method should be called after `nnictl stop` for verification.

    Args:
        wait (numeric) : time to wait in seconds
        max_retries (int): max number of retries
    """
    i = 0
    while i < max_retries:
        try:
            get_experiment_status(NNI_STATUS_URL)
        except Exception:
            break
        time.sleep(wait)
        i += 1
    if i == max_retries:
        raise TimeoutError("check_stopped() timed out")

def check_metrics_written(wait=WAITING_TIME, max_retries=MAX_RETRIES):
    """Waits until the metrics have been written to the trial logs.

    Args:
        wait (numeric) : time to wait in seconds
        max_retries (int): max number of retries
    """
    i = 0
    while i < max_retries:
        all_trials = requests.get(NNI_TRIAL_JOBS_URL).json()
        if all(["finalMetricData" in trial for trial in all_trials]):
            break
        time.sleep(wait)
        i += 1
    if i == max_retries:
        raise TimeoutError("check_metrics_written() timed out")


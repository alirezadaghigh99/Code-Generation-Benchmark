class JobStatus:
    job_name: str = ""
    jobs: list[Any] = []
    current_status: Any = None
    job_statuses: list[Any] = []
    filtered_statuses: list[Any] = []
    failure_chain: list[Any] = []
    flaky_jobs: list[Any] = []

    def __init__(self, job_name: str, job_statuses: list[Any]) -> None:
        self.job_name = job_name
        self.job_statuses = job_statuses

        self.filtered_statuses = list(
            filter(lambda j: not is_job_skipped(j), job_statuses)
        )
        self.current_status = self.get_current_status()
        self.failure_chain = self.get_most_recent_failure_chain()
        self.flaky_jobs = self.get_flaky_jobs()

    def get_current_status(self) -> Any:
        """
        When getting the current status, we want the latest status which is not pending,
        be it success or failure
        """
        for status in self.filtered_statuses:
            if status["conclusion"] != PENDING:
                return status
        return None

    def get_unique_failures(self, jobs: list[Any]) -> dict[str, list[Any]]:
        """
        Returns list of jobs grouped by failureCaptures from the input list
        """
        failures = defaultdict(list)
        for job in jobs:
            if job["conclusion"] == "failure":
                found_similar_failure = False
                if "failureCaptures" not in job:
                    failures["unclassified"] = [job]
                    continue

                # This is now a list returned by HUD API, not a string
                failureCaptures = " ".join(job["failureCaptures"])

                for failure in failures:
                    seq = SequenceMatcher(None, failureCaptures, failure)
                    if seq.ratio() > SIMILARITY_THRESHOLD:
                        failures[failure].append(job)
                        found_similar_failure = True
                        break
                if not found_similar_failure:
                    failures[failureCaptures] = [job]

        return failures

    # A flaky job is if it's the only job that has that failureCapture and is not the most recent job
    def get_flaky_jobs(self) -> list[Any]:
        unique_failures = self.get_unique_failures(self.filtered_statuses)
        flaky_jobs = []
        for failure in unique_failures:
            failure_list = unique_failures[failure]
            if (
                len(failure_list) == 1
                and failure_list[0]["sha"] != self.current_status["sha"]
            ):
                flaky_jobs.append(failure_list[0])
        return flaky_jobs

    # The most recent failure chain is an array of jobs that have the same-ish failures.
    # A success in the middle of the chain will terminate the chain.
    def get_most_recent_failure_chain(self) -> list[Any]:
        failures = []
        found_most_recent_failure = False

        for job in self.filtered_statuses:
            if is_job_failed(job):
                failures.append(job)
                found_most_recent_failure = True
            if found_most_recent_failure and not is_job_failed(job):
                break

        return failures

    def should_alert(self) -> bool:
        # Group jobs by their failures. The length of the failure chain is used
        # to raise the alert, so we can do a simple tweak here to use the length
        # of the longest unique chain
        unique_failures = self.get_unique_failures(self.failure_chain)

        return (
            self.current_status is not None
            and self.current_status["conclusion"] != SUCCESS
            and any(
                len(failure_chain) >= FAILURE_CHAIN_THRESHOLD
                for failure_chain in unique_failures.values()
            )
            and all(
                disabled_alert not in self.job_name
                for disabled_alert in DISABLED_ALERTS
            )
        )

    def __repr__(self) -> str:
        return f"jobName: {self.job_name}"


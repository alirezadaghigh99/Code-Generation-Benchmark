class DataMonitor:
    """
    An object that can be used to audit new data using the statistics from a fitted Datalab instance.

    Parameters
    ----------
    datalab :
        The Datalab object fitted to the original training dataset.
    """

    def __init__(self, datalab: Datalab):
        if str(datalab.task) != "classification":
            raise NotImplementedError(
                f"Currently, only classification tasks are supported for DataMonitor."
                f' The task of the provided Datalab instance is "{str(datalab.task)}", which is not supported by DataMonitor.'
            )

        self.label_map = datalab._label_map

        self.info = datalab.get_info()
        datalab_issue_summary = datalab.get_issue_summary()
        issue_names_checked = datalab_issue_summary["issue_type"].tolist()
        for issue_name in issue_names_checked:
            # lab.get_info() is an alias for lab.info, but some keys are handled differently via lab.get_info(key) method.
            _missing_keys = set(datalab.get_info(issue_name).keys()) - set(self.info.keys())
            self.info[issue_name].update(
                {k: v for (k, v) in datalab.get_info(issue_name).items() if k in _missing_keys}
            )

        # TODO: Compare monitors and the issue types that Datalab managed to check. Print types that DataMonitor won't consider.
        # TODO: If label issues were checked by Datalab, but with features, then the monitor will skip the label issue check, explaining that it won't support that argument for now. Generalize this for all issue types.
        # TODO: Fail on issue types that DataMonitor is asked to check, but Datalab didn't check.

        # Set up monitors
        self.monitors: Dict[str, IssueMonitor] = {}

        # Helper lmabda to figure out what inputs where used for the issue checks in Datalab
        _check_issue_input = (
            lambda n, i: self.info.get(n, {}).get("find_issues_inputs", {}).get(i, False)
        )

        # Only consider label error detection if checked by Datalab, using pred_probs
        label_issue_checked = "label" in issue_names_checked
        pred_probs_checked_for_label = _check_issue_input("label", "pred_probs")

        if label_issue_checked and pred_probs_checked_for_label:
            self.monitors["label"] = LabelIssueMonitor(self.info)

        # Only consider outlier detection if checked by Datalab, using features
        outliers_checked = "outlier" in issue_names_checked
        features_checked_for_outlier = _check_issue_input("outlier", "features")

        if outliers_checked and features_checked_for_outlier:
            self.monitors["outlier"] = OutlierIssueMonitor(self.info)

        if not self.monitors:
            if issue_names_checked:
                # No monitors were created, so we can't check for any issues.
                raise ValueError("No issue types checked by Datalab are supported by DataMonitor.")
            # Datalab didn't check any issues with the expected inputs, so we can't check for any issues.
            error_msg = (
                "No issue types checked by Datalab. DataMonitor requires at least one issue type to be checked."
                " The following issue types are supported by DataMonitor: label, outlier."
            )
            raise ValueError(error_msg)

        # Set up dictionary of lists for efficiently constructing the issues DataFrame
        issue_names = self.monitors.keys()

        # This issue dictionary will collect the issues for the entire stream of data.
        self.issues_dict: Dict[str, Union[List[bool], List[float]]] = {
            col: []
            for cols in zip(
                [f"is_{name}_issue" for name in issue_names],
                [f"{name}_score" for name in issue_names],
            )
            for col in cols
        }

    @property
    def issues(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(self.issues_dict)

    @property
    def issue_summary(self) -> pd.DataFrame:
        issue_summary_dict: Dict[str, Union[List[str], List[int], List[float]]] = {
            "issue_type": [],
            "num_issues": [],
            "score": [],
        }
        issue_names = self.monitors.keys()
        issue_summary_dict["issue_type"] = list(issue_names)
        issue_summary_dict["num_issues"] = [
            np.sum(self.issues_dict[f"is_{issue_name}_issue"]) for issue_name in issue_names
        ]
        issue_summary_dict["score"] = [
            float(np.mean(self.issues_dict[f"{issue_name}_score"])) for issue_name in issue_names
        ]
        return pd.DataFrame.from_dict(issue_summary_dict)

    def find_issues(
        self,
        *,
        labels: Optional[np.ndarray] = None,
        pred_probs: Optional[np.ndarray] = None,
        features: Optional[np.ndarray] = None,
    ) -> None:
        # TODO: Simplifying User Input: Ensure that users can pass input in the simplest form possible.
        # See FindIssuesKwargs._adapt_to_singletons TODO for more details.

        str_to_int_map: Dict[Any, Any] = {v: k for (k, v) in self.label_map.items()}
        find_issues_kwargs = FindIssuesKwargs(
            labels=labels,
            pred_probs=pred_probs,
            features=features,
            _label_map=str_to_int_map,
        )
        issues_dict: Dict[str, Union[List[float], List[bool], np.ndarray]] = {
            k: [] for k in self.issues_dict.keys()
        }

        # Flag to track if any monitor has found issues
        display_results = False
        for issue_name, issue_monitor in self.monitors.items():
            issue_monitor.find_issues(find_issues_kwargs)

            # Update issues_dict based on the current monitor's findings for the current batch
            issues_dict[f"is_{issue_name}_issue"] = issue_monitor.issues_dict["is_issue"]
            issues_dict[f"{issue_name}_score"] = issue_monitor.issues_dict["score"]

            if issue_monitor.batch_has_issues:
                display_results = True

            # Clear the current monitor's issues dictionary immediately after processing
            issue_monitor.clear_issues_dict()

        if display_results:
            self._display_batch_issues(issues_dict, labels=labels, pred_probs=pred_probs)

        # Append the issues to the existing issues dictionary
        for k, v in issues_dict.items():
            self.issues_dict[k].extend(v)  # type: ignore[arg-type]

    def _display_batch_issues(
        self, issues_dicts: Dict[str, Union[List[float], List[bool], np.ndarray]], **kwargs
    ) -> None:
        start_index = len(
            next(iter(self.issues_dict.values()))
        )  # TODO: Abstract this into a method for checking how many examples have been processed/checked. E.g. __len__ or a property.
        end_index = start_index + len(next(iter(issues_dicts.values())))
        index = np.arange(start_index, end_index)

        issue_columns = [
            col for col in issues_dicts.keys() if (col.startswith("is_") and col.endswith("_issue"))
        ]
        score_columns = [
            col.replace("is_", "").replace("_issue", "_score") for col in issue_columns
        ]
        issue_score_pairs = zip(issue_columns, score_columns)

        pairs_to_keep = [
            (issue_col, score_col)
            for issue_col, score_col in issue_score_pairs
            if np.any(issues_dicts[issue_col])
        ]
        filtered_issues_dicts = {}
        for issue_col, score_col in pairs_to_keep:
            filtered_issues_dicts.update(
                {issue_col: issues_dicts[issue_col], score_col: issues_dicts[score_col]}
            )

        df_issues = pd.DataFrame(filtered_issues_dicts, index=index)

        is_issue_columns = [
            col for col in df_issues.columns if (col.startswith("is_") and col.endswith("_issue"))
        ]
        if "is_label_issue" in is_issue_columns:
            df_issues["given_label"] = kwargs["labels"]
            df_issues["suggested_label"] = np.vectorize(self.label_map.get)(
                np.argmax(kwargs["pred_probs"], axis=1)
            )

        df_subset = df_issues.query(f"{' | '.join([f'{col} == True' for col in is_issue_columns])}")

        print(
            "Detected issues in the current batch:\n",
            (
                df_issues.query(
                    f"{' | '.join([f'{col} == True' for col in is_issue_columns])}"
                ).to_string()
            ),
            "\n",
        )


class DataIssues:
    """
    Class that collects and stores information and statistics on issues found in a dataset.

    Parameters
    ----------
    data :
        The data object for which the issues are being collected.
    strategy :
        Strategy used for processing info dictionaries.

    Attributes
    ----------
    issues : pd.DataFrame
        Stores information about each individual issue found in the data,
        on a per-example basis.
    issue_summary : pd.DataFrame
        Summarizes the overall statistics for each issue type.
    info : dict
        A dictionary that contains information and statistics about the data and each issue type.
    """

    def __init__(self, data: Data, strategy: Type[_InfoStrategy]) -> None:
        self.issues: pd.DataFrame = pd.DataFrame(index=range(len(data)))
        self.issue_summary: pd.DataFrame = pd.DataFrame(
            columns=["issue_type", "score", "num_issues"]
        ).astype({"score": np.float64, "num_issues": np.int64})
        self.info: Dict[str, Dict[str, Any]] = {
            "statistics": get_data_statistics(data),
        }
        self._data = data
        self._strategy = strategy

    def get_info(self, issue_name: Optional[str] = None) -> Dict[str, Any]:
        return self._strategy.get_info(data=self._data, info=self.info, issue_name=issue_name)

    @property
    def statistics(self) -> Dict[str, Any]:
        """Returns the statistics dictionary.

        Shorthand for self.info["statistics"].
        """
        return self.info["statistics"]

    def get_issues(self, issue_name: Optional[str] = None) -> pd.DataFrame:
        """
        Use this after finding issues to see which examples suffer from which types of issues.

        Parameters
        ----------
        issue_name : str or None
            The type of issue to focus on. If `None`, returns full DataFrame summarizing all of the types of issues detected in each example from the dataset.

        Raises
        ------
        ValueError
            If `issue_name` is not a type of issue previously considered in the audit.

        Returns
        -------
        specific_issues :
            A DataFrame where each row corresponds to an example from the dataset and columns specify:
            whether this example exhibits a particular type of issue and how severely (via a numeric quality score where lower values indicate more severe instances of the issue).

            Additional columns may be present in the DataFrame depending on the type of issue specified.
        """
        if self.issues.empty:
            raise ValueError(
                """No issues available for retrieval. Please check the following before using `get_issues`:
                1. Ensure `find_issues` was executed. If not, please run it with the necessary parameters.
                2. If `find_issues` was run but you're seeing this message,
                    it may have encountered limitations preventing full analysis.
                    However, partial checks can still provide valuable insights.
            Review `find_issues` output carefully for any specific actions needed
            to facilitate a more comprehensive analysis before calling `get_issues`.
            """
            )
        if issue_name is None:
            return self.issues

        columns = [col for col in self.issues.columns if issue_name in col]
        if not columns:
            raise ValueError(
                f"""No columns found for issue type '{issue_name}'. Ensure the following:
                1. `find_issues` has been executed. If it hasn't, please run it.
                2. Check `find_issues` output to verify that the issue type '{issue_name}' was included in the checks to
                    ensure it was not excluded accidentally before the audit.
                3. Review `find_issues` output for any errors or warnings that might indicate the check for '{issue_name}' issues failed to complete.
                    This can provide better insights into what adjustments may be necessary.
            """
            )
        specific_issues = self.issues[columns]
        info = self.get_info(issue_name=issue_name)

        if issue_name == "label":
            specific_issues = specific_issues.assign(
                given_label=info["given_label"], predicted_label=info["predicted_label"]
            )

        if issue_name == "near_duplicate":
            column_dict = {
                k: info.get(k)
                for k in ["near_duplicate_sets", "distance_to_nearest_neighbor"]
                if info.get(k) is not None
            }
            specific_issues = specific_issues.assign(**column_dict)

        if issue_name == "class_imbalance":
            specific_issues = specific_issues.assign(given_label=info["given_label"])
        return specific_issues

    def get_issue_summary(self, issue_name: Optional[str] = None) -> pd.DataFrame:
        """Summarize the issues found in dataset of a particular type,
        including how severe this type of issue is overall across the dataset.

        Parameters
        ----------
        issue_name :
            Name of the issue type to summarize. If `None`, summarizes each of the different issue types previously considered in the audit.

        Returns
        -------
        issue_summary :
            DataFrame where each row corresponds to a type of issue, and columns quantify:
            the number of examples in the dataset estimated to exhibit this type of issue,
            and the overall severity of the issue across the dataset (via a numeric quality score where lower values indicate that the issue is overall more severe).
        """
        if self.issue_summary.empty:
            raise ValueError(
                "No issues found in the dataset. "
                "Call `find_issues` before calling `get_issue_summary`."
            )

        if issue_name is None:
            return self.issue_summary

        row_mask = self.issue_summary["issue_type"] == issue_name
        if not any(row_mask):
            raise ValueError(f"Issue type {issue_name} not found in the summary.")
        return self.issue_summary[row_mask].reset_index(drop=True)

    def collect_statistics(self, issue_manager: Union[IssueManager, "Imagelab"]) -> None:
        """Update the statistics in the info dictionary.

        Parameters
        ----------
        statistics :
            A dictionary of statistics to add/update in the info dictionary.

        Examples
        --------

        A common use case is to reuse the KNN-graph across multiple issue managers.
        To avoid recomputing the KNN-graph for each issue manager,
        we can pass it as a statistic to the issue managers.

        >>> from scipy.sparse import csr_matrix
        >>> weighted_knn_graph = csr_matrix(...)
        >>> issue_manager_that_computes_knn_graph = ...

        """
        key = "statistics"
        statistics: Dict[str, Any] = issue_manager.info.get(key, {})
        if statistics:
            self.info[key].update(statistics)

    def _update_issues(self, issue_manager):
        overlapping_columns = list(set(self.issues.columns) & set(issue_manager.issues.columns))
        if overlapping_columns:
            warnings.warn(
                f"Overwriting columns {overlapping_columns} in self.issues with "
                f"columns from issue manager {issue_manager}."
            )
            self.issues.drop(columns=overlapping_columns, inplace=True)
        self.issues = self.issues.join(issue_manager.issues, how="outer")

    def _update_issue_info(self, issue_name, new_info):
        if issue_name in self.info:
            warnings.warn(f"Overwriting key {issue_name} in self.info")
        self.info[issue_name] = new_info

    def collect_issues_from_issue_manager(self, issue_manager: IssueManager) -> None:
        """
        Collects results from an IssueManager and update the corresponding
        attributes of the Datalab object.

        This includes:
        - self.issues
        - self.issue_summary
        - self.info

        Parameters
        ----------
        issue_manager :
            IssueManager object to collect results from.
        """
        self._update_issues(issue_manager)

        if issue_manager.issue_name in self.issue_summary["issue_type"].values:
            warnings.warn(
                f"Overwriting row in self.issue_summary with "
                f"row from issue manager {issue_manager}."
            )
            self.issue_summary = self.issue_summary[
                self.issue_summary["issue_type"] != issue_manager.issue_name
            ]
        issue_column_name: str = f"is_{issue_manager.issue_name}_issue"
        num_issues: int = int(issue_manager.issues[issue_column_name].sum())
        self.issue_summary = pd.concat(
            [
                self.issue_summary,
                issue_manager.summary.assign(num_issues=num_issues),
            ],
            axis=0,
            ignore_index=True,
        )
        self._update_issue_info(issue_manager.issue_name, issue_manager.info)

    def collect_issues_from_imagelab(self, imagelab: "Imagelab", issue_types: List[str]) -> None:
        pass  # pragma: no cover

    def set_health_score(self) -> None:
        """Set the health score for the dataset based on the issue summary.

        Currently, the health score is the mean of the scores for each issue type.
        """
        self.info["statistics"]["health_score"] = self.issue_summary["score"].mean()


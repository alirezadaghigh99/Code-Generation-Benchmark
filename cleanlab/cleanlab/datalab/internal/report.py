def _write_summary(self, summary: pd.DataFrame) -> str:
        statistics = self.data_issues.get_info("statistics")
        num_examples = statistics["num_examples"]
        num_classes = statistics.get(
            "num_classes"
        )  # This may not be required for all types of datasets  in the future (e.g. unlabeled/regression)

        dataset_information = f"Dataset Information: num_examples: {num_examples}"
        if num_classes is not None:
            dataset_information += f", num_classes: {num_classes}"

        if not self.show_all_issues:
            # Drop any items in the issue_summary that have no issues (any issue detected in data needs to have num_issues > 0)
            summary = summary.query("num_issues > 0")

        report_header = (
            f"{dataset_information}\n\n"
            + "Here is a summary of various issues found in your data:\n\n"
        )
        report_footer = (
            "\n\n"
            + "Learn about each issue: https://docs.cleanlab.ai/stable/cleanlab/datalab/guide/issue_type_description.html\n"
            + "See which examples in your dataset exhibit each issue via: `datalab.get_issues(<ISSUE_NAME>)`\n\n"
            + "Data indices corresponding to top examples of each issue are shown below.\n\n\n"
        )

        if self.show_summary_score:
            return (
                report_header
                + summary.to_string(index=False)
                + "\n\n"
                + "(Note: A lower score indicates a more severe issue across all examples in the dataset.)"
                + report_footer
            )

        return (
            report_header + summary.drop(columns=["score"]).to_string(index=False) + report_footer
        )


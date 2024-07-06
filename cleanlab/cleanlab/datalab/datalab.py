def get_issue_summary(self, issue_name: Optional[str] = None) -> pd.DataFrame:
        """Summarize the issues found in dataset of a particular type,
        including how severe this type of issue is overall across the dataset.

        See the documentation of the ``issue_summary`` attribute to learn more.

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
            The quality scores lie between 0-1 and are directly comparable between multiple datasets (for the same issue type), but not across different issue types.
        """
        return self.data_issues.get_issue_summary(issue_name=issue_name)

def get_info(self, issue_name: Optional[str] = None) -> Dict[str, Any]:
        """Get the info for the issue_name key.

        This function is used to get the info for a specific issue_name. If the info is not computed yet, it will raise an error.

        Parameters
        ----------
        issue_name :
            The issue name for which the info is required.

        Returns
        -------
        :py:meth:`info <cleanlab.datalab.internal.data_issues.DataIssues.get_info>` :
            The info for the issue_name.
        """
        return self.data_issues.get_info(issue_name)

def save(self, path: str, force: bool = False) -> None:
        """Saves this Datalab object to file (all files are in folder at `path/`).
        We do not guarantee saved Datalab can be loaded from future versions of cleanlab.

        Parameters
        ----------
        path :
            Folder in which all information about this Datalab should be saved.

        force :
            If ``True``, overwrites any existing files in the folder at `path`. Use this with caution!

        NOTE
        ----
        You have to save the Dataset yourself separately if you want it saved to file.
        """
        _Serializer.serialize(path=path, datalab=self, force=force)
        save_message = f"Saved Datalab to folder: {path}"
        print(save_message)

def get_issue_summary(self, issue_name: Optional[str] = None) -> pd.DataFrame:
        """Summarize the issues found in dataset of a particular type,
        including how severe this type of issue is overall across the dataset.

        See the documentation of the ``issue_summary`` attribute to learn more.

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
            The quality scores lie between 0-1 and are directly comparable between multiple datasets (for the same issue type), but not across different issue types.
        """
        return self.data_issues.get_issue_summary(issue_name=issue_name)


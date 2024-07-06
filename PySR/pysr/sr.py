def latex_table(
        self,
        indices=None,
        precision=3,
        columns=["equation", "complexity", "loss", "score"],
    ):
        """Create a LaTeX/booktabs table for all, or some, of the equations.

        Parameters
        ----------
        indices : list[int] | list[list[int]]
            If you wish to select a particular subset of equations from
            `self.equations_`, give the row numbers here. By default,
            all equations will be used. If there are multiple output
            features, then pass a list of lists.
        precision : int
            The number of significant figures shown in the LaTeX
            representations.
            Default is `3`.
        columns : list[str]
            Which columns to include in the table.
            Default is `["equation", "complexity", "loss", "score"]`.

        Returns
        -------
        latex_table_str : str
            A string that will render a table in LaTeX of the equations.
        """
        self.refresh()

        if isinstance(self.equations_, list):
            if indices is not None:
                assert isinstance(indices, list)
                assert isinstance(indices[0], list)
                assert len(indices) == self.nout_

            table_string = sympy2multilatextable(
                self.equations_, indices=indices, precision=precision, columns=columns
            )
        elif isinstance(self.equations_, pd.DataFrame):
            if indices is not None:
                assert isinstance(indices, list)
                assert isinstance(indices[0], int)

            table_string = sympy2latextable(
                self.equations_, indices=indices, precision=precision, columns=columns
            )
        else:
            raise ValueError(
                "Invalid type for equations_ to pass to `latex_table`. "
                "Expected a DataFrame or a list of DataFrames."
            )

        return with_preamble(table_string)


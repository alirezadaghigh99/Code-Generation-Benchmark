class SepDataFrame:
    """
    (Sep)erate DataFrame
    We usually concat multiple dataframe to be processed together(Such as feature, label, weight, filter).
    However, they are usually be used separately at last.
    This will result in extra cost for concatenating and splitting data(reshaping and copying data in the memory is very expensive)

    SepDataFrame tries to act like a DataFrame whose column with multiindex
    """

    # TODO:
    # SepDataFrame try to behave like pandas dataframe,  but it is still not them same
    # Contributions are welcome to make it more complete.

    def __init__(self, df_dict: Dict[str, pd.DataFrame], join: str, skip_align=False):
        """
        initialize the data based on the dataframe dictionary

        Parameters
        ----------
        df_dict : Dict[str, pd.DataFrame]
            dataframe dictionary
        join : str
            how to join the data
            It will reindex the dataframe based on the join key.
            If join is None, the reindex step will be skipped

        skip_align :
            for some cases, we can improve performance by skipping aligning index
        """
        self.join = join

        if skip_align:
            self._df_dict = df_dict
        else:
            self._df_dict = align_index(df_dict, join)

    @property
    def loc(self):
        return SDFLoc(self, join=self.join)

    @property
    def index(self):
        return self._df_dict[self.join].index

    def apply_each(self, method: str, skip_align=True, *args, **kwargs):
        """
        Assumptions:
        - inplace methods will return None
        """
        inplace = False
        df_dict = {}
        for k, df in self._df_dict.items():
            df_dict[k] = getattr(df, method)(*args, **kwargs)
            if df_dict[k] is None:
                inplace = True
        if not inplace:
            return SepDataFrame(df_dict=df_dict, join=self.join, skip_align=skip_align)

    def sort_index(self, *args, **kwargs):
        return self.apply_each("sort_index", True, *args, **kwargs)

    def copy(self, *args, **kwargs):
        return self.apply_each("copy", True, *args, **kwargs)

    def _update_join(self):
        if self.join not in self:
            if len(self._df_dict) > 0:
                self.join = next(iter(self._df_dict.keys()))
            else:
                # NOTE: this will change the behavior of previous reindex when all the keys are empty
                self.join = None

    def __getitem__(self, item):
        # TODO: behave more like pandas when multiindex
        return self._df_dict[item]

    def __setitem__(self, item: str, df: Union[pd.DataFrame, pd.Series]):
        # TODO: consider the join behavior
        if not isinstance(item, tuple):
            self._df_dict[item] = df
        else:
            # NOTE: corner case of MultiIndex
            _df_dict_key, *col_name = item
            col_name = tuple(col_name)
            if _df_dict_key in self._df_dict:
                if len(col_name) == 1:
                    col_name = col_name[0]
                self._df_dict[_df_dict_key][col_name] = df
            else:
                if isinstance(df, pd.Series):
                    if len(col_name) == 1:
                        col_name = col_name[0]
                    self._df_dict[_df_dict_key] = df.to_frame(col_name)
                else:
                    df_copy = df.copy()  # avoid changing df
                    df_copy.columns = pd.MultiIndex.from_tuples([(*col_name, *idx) for idx in df.columns.to_list()])
                    self._df_dict[_df_dict_key] = df_copy

    def __delitem__(self, item: str):
        del self._df_dict[item]
        self._update_join()

    def __contains__(self, item):
        return item in self._df_dict

    def __len__(self):
        return len(self._df_dict[self.join])

    def droplevel(self, *args, **kwargs):
        raise NotImplementedError(f"Please implement the `droplevel` method")

    @property
    def columns(self):
        dfs = []
        for k, df in self._df_dict.items():
            df = df.head(0)
            df.columns = pd.MultiIndex.from_product([[k], df.columns])
            dfs.append(df)
        return pd.concat(dfs, axis=1).columns

    # Useless methods
    @staticmethod
    def merge(df_dict: Dict[str, pd.DataFrame], join: str):
        all_df = df_dict[join]
        for k, df in df_dict.items():
            if k != join:
                all_df = all_df.join(df)
        return all_df


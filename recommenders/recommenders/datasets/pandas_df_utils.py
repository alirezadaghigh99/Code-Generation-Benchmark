class LibffmConverter:
    """Converts an input dataframe to another dataframe in libffm format. A text file of the converted
    Dataframe is optionally generated.

    Note:

        The input dataframe is expected to represent the feature data in the following schema:

        .. code-block:: python

            |field-1|field-2|...|field-n|rating|
            |feature-1-1|feature-2-1|...|feature-n-1|1|
            |feature-1-2|feature-2-2|...|feature-n-2|0|
            ...
            |feature-1-i|feature-2-j|...|feature-n-k|0|

        Where
        1. each `field-*` is the column name of the dataframe (column of label/rating is excluded), and
        2. `feature-*-*` can be either a string or a numerical value, representing the categorical variable or
        actual numerical variable of the feature value in the field, respectively.
        3. If there are ordinal variables represented in int types, users should make sure these columns
        are properly converted to string type.

        The above data will be converted to the libffm format by following the convention as explained in
        `this paper <https://www.csie.ntu.edu.tw/~r01922136/slides/ffm.pdf>`_.

        i.e. `<field_index>:<field_feature_index>:1` or `<field_index>:<field_feature_index>:<field_feature_value>`,
        depending on the data type of the features in the original dataframe.

    Args:
        filepath (str): path to save the converted data.

    Attributes:
        field_count (int): count of field in the libffm format data
        feature_count (int): count of feature in the libffm format data
        filepath (str or None): file path where the output is stored - it can be None or a string

    Examples:
        >>> import pandas as pd
        >>> df_feature = pd.DataFrame({
                'rating': [1, 0, 0, 1, 1],
                'field1': ['xxx1', 'xxx2', 'xxx4', 'xxx4', 'xxx4'],
                'field2': [3, 4, 5, 6, 7],
                'field3': [1.0, 2.0, 3.0, 4.0, 5.0],
                'field4': ['1', '2', '3', '4', '5']
            })
        >>> converter = LibffmConverter().fit(df_feature, col_rating='rating')
        >>> df_out = converter.transform(df_feature)
        >>> df_out
            rating field1 field2   field3 field4
        0       1  1:1:1  2:4:3  3:5:1.0  4:6:1
        1       0  1:2:1  2:4:4  3:5:2.0  4:7:1
        2       0  1:3:1  2:4:5  3:5:3.0  4:8:1
        3       1  1:3:1  2:4:6  3:5:4.0  4:9:1
        4       1  1:3:1  2:4:7  3:5:5.0  4:10:1
    """

    def __init__(self, filepath=None):
        self.filepath = filepath
        self.col_rating = None
        self.field_names = None
        self.field_count = None
        self.feature_count = None

    def fit(self, df, col_rating=DEFAULT_RATING_COL):
        """Fit the dataframe for libffm format.
        This method does nothing but check the validity of the input columns

        Args:
            df (pandas.DataFrame): input Pandas dataframe.
            col_rating (str): rating of the data.

        Return:
            object: the instance of the converter
        """

        # Check column types.
        types = df.dtypes
        if not all(
            [
                x == object or np.issubdtype(x, np.integer) or x == np.float
                for x in types
            ]
        ):
            raise TypeError("Input columns should be only object and/or numeric types.")

        if col_rating not in df.columns:
            raise TypeError(
                "Column of {} is not in input dataframe columns".format(col_rating)
            )

        self.col_rating = col_rating
        self.field_names = list(df.drop(col_rating, axis=1).columns)

        return self

    def transform(self, df):
        """Tranform an input dataset with the same schema (column names and dtypes) to libffm format
        by using the fitted converter.

        Args:
            df (pandas.DataFrame): input Pandas dataframe.

        Return:
            pandas.DataFrame: Output libffm format dataframe.
        """
        if self.col_rating not in df.columns:
            raise ValueError(
                "Input dataset does not contain the label column {} in the fitting dataset".format(
                    self.col_rating
                )
            )

        if not all([x in df.columns for x in self.field_names]):
            raise ValueError(
                "Not all columns in the input dataset appear in the fitting dataset"
            )

        # Encode field-feature.
        idx = 1
        self.field_feature_dict = {}
        for field in self.field_names:
            for feature in df[field].values:
                # Check whether (field, feature) tuple exists in the dict or not.
                # If not, put them into the key-values of the dict and count the index.
                if (field, feature) not in self.field_feature_dict:
                    self.field_feature_dict[(field, feature)] = idx
                    if df[field].dtype == object:
                        idx += 1
            if df[field].dtype != object:
                idx += 1

        self.field_count = len(self.field_names)
        self.feature_count = idx - 1

        def _convert(field, feature, field_index, field_feature_index_dict):
            field_feature_index = field_feature_index_dict[(field, feature)]
            if isinstance(feature, str):
                feature = 1
            return "{}:{}:{}".format(field_index, field_feature_index, feature)

        for col_index, col in enumerate(self.field_names):
            df[col] = df[col].apply(
                lambda x: _convert(col, x, col_index + 1, self.field_feature_dict)
            )

        # Move rating column to the first.
        column_names = self.field_names[:]
        column_names.insert(0, self.col_rating)
        df = df[column_names]

        if self.filepath is not None:
            np.savetxt(self.filepath, df.values, delimiter=" ", fmt="%s")

        return df

    def fit_transform(self, df, col_rating=DEFAULT_RATING_COL):
        """Do fit and transform in a row

        Args:
            df (pandas.DataFrame): input Pandas dataframe.
            col_rating (str): rating of the data.

        Return:
            pandas.DataFrame: Output libffm format dataframe.
        """
        return self.fit(df, col_rating=col_rating).transform(df)

    def get_params(self):
        """Get parameters (attributes) of the libffm converter

        Return:
            dict: A dictionary that contains parameters field count, feature count, and file path.
        """
        return {
            "field count": self.field_count,
            "feature count": self.feature_count,
            "file path": self.filepath,
        }


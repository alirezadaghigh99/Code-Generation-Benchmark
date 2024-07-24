class OneHotEncoding(Encoding):
    def __init__(self, categories: List):
        """
        Initializes an instance of OneHotEncoding class
        and generates one hot encodings for given categories.
        Categories are assigned 1's in the order they appear in the provided list.

        An example reference about one hot encoding:
        https://www.kaggle.com/dansbecker/using-categorical-data-with-one-hot-encoding

        :param categories: List of categories to encode
        """
        encodings = []
        for i, _ in enumerate(categories):
            e = [0] * len(categories)
            e[i] = 1
            encodings.append(e)

        super(OneHotEncoding, self).__init__(categories, encodings)

    def round_row(self, x_row):
        """
        Rounds the given row. The highest value is rounded to 1
        all other values are rounded to 0

        :param x_row: A row to round.
        :returns: A rounded row.
        """
        idx = x_row.argmax()
        row_rounded = np.zeros(x_row.shape)
        row_rounded[idx] = 1
        return row_rounded

class OrdinalEncoding(Encoding):
    def __init__(self, categories: List):
        """
        Initializes an instance of OrdinalEncoding class
        and generates ordinal encodings for given categories.
        The encoding is a list of integer numbers [1, .. , n]
        where n is the number of categories.
        Categories are assigned codes in the order they appear in the provided list.

        Note that encoding categories with ordinal encoding is effectively the same as
        treating them as a discrete variable.

        :param categories: List of categories to encode
        """
        encodings = [[i + 1] for i, _ in enumerate(categories)]
        super(OrdinalEncoding, self).__init__(categories, encodings)

    def round_row(self, x_row):
        # since we used just one column for this encoding
        # x_row should contain a single number

        if x_row.shape[0] != 1:
            raise ValueError("Expected a single valued array, got array of {}" + str(x_row.shape))

        x_value = x_row[0]
        if x_value < 1:
            x_value = 1
        if x_value > len(self.categories):
            x_value = len(self.categories)

        rounded_value = int(round(x_value))

        return np.array([rounded_value])


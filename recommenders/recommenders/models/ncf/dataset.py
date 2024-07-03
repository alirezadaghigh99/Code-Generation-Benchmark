class DataFile:
    """
    DataFile class for NCF. Iterator to read data from a csv file.
    Data must be sorted by user. Includes utilities for loading user data from
    file, formatting it and returning a Pandas dataframe.
    """

    def __init__(
        self, filename, col_user, col_item, col_rating, col_test_batch=None, binary=True
    ):
        """Constructor

        Args:
            filename (str): Path to file to be processed.
            col_user (str): User column name.
            col_item (str): Item column name.
            col_rating (str): Rating column name.
            col_test_batch (str): Test batch column name.
            binary (bool): If true, set rating > 0 to rating = 1.
        """
        self.filename = filename
        self.col_user = col_user
        self.col_item = col_item
        self.col_rating = col_rating
        self.col_test_batch = col_test_batch
        self.expected_fields = [self.col_user, self.col_item, self.col_rating]
        if self.col_test_batch is not None:
            self.expected_fields.append(self.col_test_batch)
        self.binary = binary
        self._init_data()
        self.id2user = {self.user2id[k]: k for k in self.user2id}
        self.id2item = {self.item2id[k]: k for k in self.item2id}

    @property
    def users(self):
        return self.user2id.keys()

    @property
    def items(self):
        return self.item2id.keys()

    @property
    def end_of_file(self):
        return (self.line_num > 0) and self.next_row is None

    def __iter__(self):
        return self

    def __enter__(self, *args):
        self.file = open(self.filename, "r", encoding="UTF8")
        self.reader = csv.DictReader(self.file)
        self._check_for_missing_fields(self.expected_fields)
        self.line_num = 0
        self.row, self.next_row = None, None
        return self

    def __exit__(self, *args):
        self.file.close()
        self.reader = None
        self.line_num = 0
        self.row, self.next_row = None, None

    def __next__(self):
        if self.next_row:
            self.row = self.next_row
        elif self.line_num == 0:
            self.row = self._extract_row_data(next(self.reader, None))
            if self.row is None:
                raise EmptyFileException("{} is empty.".format(self.filename))
        else:
            raise StopIteration  # end of file
        self.next_row = self._extract_row_data(next(self.reader, None))
        self.line_num += 1

        return self.row

    def _check_for_missing_fields(self, fields_to_check):
        missing_fields = set(fields_to_check).difference(set(self.reader.fieldnames))
        if len(missing_fields):
            raise MissingFieldsException(
                "Columns {} not in header of file {}".format(
                    missing_fields, self.filename
                )
            )

    def _extract_row_data(self, row):
        if row is None:
            return row
        user = int(row[self.col_user])
        item = int(row[self.col_item])
        rating = float(row[self.col_rating])
        if self.binary:
            rating = float(rating > 0)
        test_batch = None
        if self.col_test_batch:
            test_batch = int(row[self.col_test_batch])
        return {
            self.col_user: user,
            self.col_item: item,
            self.col_rating: rating,
            self.col_test_batch: test_batch,
        }

    def _init_data(self):
        # Compile lists of unique users and items, assign IDs to users and items,
        # and ensure file is sorted by user (and batch index if test set)
        logger.info("Indexing {} ...".format(self.filename))
        with self:
            user_items = []
            self.item2id, self.user2id = OrderedDict(), OrderedDict()
            batch_index = 0
            for _ in self:
                item = self.row[self.col_item]
                user = self.row[self.col_user]
                test_batch = self.row[self.col_test_batch]
                if not self.end_of_file:
                    next_user = self.next_row[self.col_user]
                    next_test_batch = self.next_row[self.col_test_batch]
                if item not in self.items:
                    self.item2id[item] = len(self.item2id)
                user_items.append(item)

                if (next_user != user) or self.next_row is None:
                    if not self.end_of_file:
                        if next_user in self.users:
                            raise FileNotSortedException(
                                "File {} is not sorted by user".format(self.filename)
                            )
                    self.user2id[user] = len(self.user2id)
                if self.col_test_batch:
                    if (next_test_batch != test_batch) or self.next_row is None:
                        if not self.end_of_file:
                            if next_test_batch < batch_index:
                                raise FileNotSortedException(
                                    "File {} is not sorted by {}".format(
                                        self.filename, self.col_test_batch
                                    )
                                )
                        batch_index += 1
            self.batch_indices_range = range(0, batch_index)
            self.data_len = self.line_num

    def load_data(self, key, by_user=True):
        """Load data for a specified user or test batch

        Args:
            key (int): user or test batch index
            by_user (bool): load data by usr if True, else by test batch

        Returns:
            pandas.DataFrame
        """
        records = []
        key_col = self.col_user if by_user else self.col_test_batch

        # fast forward in file to user/test batch
        while (self.line_num == 0) or (self.row[key_col] != key):
            if self.end_of_file:
                raise MissingUserException(
                    "User {} not in file {}".format(key, self.filename)
                )
            next(self)
        # collect user/test batch data
        while self.row[key_col] == key:
            row = self.row
            if self.col_test_batch in row:
                del row[self.col_test_batch]
            records.append(row)
            if not self.end_of_file:
                next(self)
            else:
                break
        return pd.DataFrame.from_records(records)
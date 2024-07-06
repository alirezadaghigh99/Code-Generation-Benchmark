def check_embeddings(path: str, remove_additional_columns: bool = False) -> None:
    """Raises an error if the embeddings csv file has not the correct format

    Use this check whenever you want to upload an embedding to the Lightly
    Platform.
    This method only checks whether the header row matches the specs:
    https://docs.lightly.ai/self-supervised-learning/getting_started/command_line_tool.html#id1

    Args:
        path:
            Path to the embedding csv file
        remove_additional_columns:
            If True, all additional columns
            which are not in {filenames, embeddings_x, labels} are removed.
            If false, they are kept unchanged.

    Raises:
        RuntimeError
    """
    with open(path, "r", newline="") as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        header: List[str] = next(reader)

        # check for whitespace in the header (we don't allow this)
        if any(x != x.strip() for x in header):
            raise RuntimeError("Embeddings csv file must not contain whitespaces.")

        # first col is `filenames`
        if header[0] != "filenames":
            raise RuntimeError(
                f"Embeddings csv file must start with `filenames` "
                f"column but had {header[0]} instead."
            )

        # `labels` exists
        try:
            header_labels_idx = header.index("labels")
        except ValueError:
            raise RuntimeError(f"Embeddings csv file has no `labels` column.")

        # cols between first and `labels` are `embedding_x`
        for embedding_header in header[1:header_labels_idx]:
            if not re.match(r"embedding_\d+", embedding_header):
                # check if we have a special column
                if not embedding_header in ["masked", "selected"]:
                    raise RuntimeError(
                        f"Embeddings csv file must have `embedding_x` columns but "
                        f"found {embedding_header} instead."
                    )

        # check for empty rows in the body of the csv file
        for i, row in enumerate(reader):
            if len(row) == 0:
                raise RuntimeError(
                    f"Embeddings csv file must not have empty rows. "
                    f"Found empty row on line {i}."
                )

    if remove_additional_columns:
        new_rows = []
        with open(path, "r", newline="") as csv_file:
            reader = csv.reader(csv_file, delimiter=",")
            header_row = next(reader)

            # create mask of columns to keep only filenames, embedding_ or labels
            regexp = r"filenames|(embedding_\d+)|labels"
            col_mask = []
            for i, col in enumerate(header_row):
                col_mask += [True] if re.match(regexp, col) else [False]

            # add header row manually here since we use an iterator
            new_rows.append(list(compress(header_row, col_mask)))

            for row in reader:
                # apply mask to only use filenames, embedding_ or labels
                new_rows.append(list(compress(row, col_mask)))

        with open(path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            writer.writerows(new_rows)

def save_schema(path: str, task_type: str, ids: List[int], names: List[str]) -> None:
    """Saves a prediction schema in the right format.

    Args:
        path:
            Where to store the schema.
        task_type:
            Task type (e.g. classification, object-detection).
        ids:
            List of category ids.
        names:
            List of category names.
    """
    if len(ids) != len(names):
        raise ValueError("ids and names must have same length!")

    schema = {
        "task_type": task_type,
        "categories": [{"id": id, "name": name} for id, name in zip(ids, names)],
    }
    with open(path, "w") as f:
        json.dump(schema, f)


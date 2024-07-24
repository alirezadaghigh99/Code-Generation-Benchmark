class _RemainderColsList(UserList):
    """A list that raises a warning whenever items are accessed.

    It is used to store the columns handled by the "remainder" entry of
    ``ColumnTransformer.transformers_``, ie ``transformers_[-1][-1]``.

    For some values of the ``ColumnTransformer`` ``transformers`` parameter,
    this list of indices will be replaced by either a list of column names or a
    boolean mask; in those cases we emit a ``FutureWarning`` the first time an
    element is accessed.

    Parameters
    ----------
    columns : list of int
        The remainder columns.

    future_dtype : {'str', 'bool'}, default=None
        The dtype that will be used by a ColumnTransformer with the same inputs
        in a future release. There is a default value because providing a
        constructor that takes a single argument is a requirement for
        subclasses of UserList, but we do not use it in practice. It would only
        be used if a user called methods that return a new list such are
        copying or concatenating `_RemainderColsList`.

    warning_was_emitted : bool, default=False
       Whether the warning for that particular list was already shown, so we
       only emit it once.

    warning_enabled : bool, default=True
        When False, the list never emits the warning nor updates
        `warning_was_emitted``. This is used to obtain a quiet copy of the list
        for use by the `ColumnTransformer` itself, so that the warning is only
        shown when a user accesses it directly.
    """

    def __init__(
        self,
        columns,
        *,
        future_dtype=None,
        warning_was_emitted=False,
        warning_enabled=True,
    ):
        super().__init__(columns)
        self.future_dtype = future_dtype
        self.warning_was_emitted = warning_was_emitted
        self.warning_enabled = warning_enabled

    def __getitem__(self, index):
        self._show_remainder_cols_warning()
        return super().__getitem__(index)

    def _show_remainder_cols_warning(self):
        if self.warning_was_emitted or not self.warning_enabled:
            return
        self.warning_was_emitted = True
        future_dtype_description = {
            "str": "column names (of type str)",
            "bool": "a mask array (of type bool)",
            # shouldn't happen because we always initialize it with a
            # non-default future_dtype
            None: "a different type depending on the ColumnTransformer inputs",
        }.get(self.future_dtype, self.future_dtype)

        # TODO(1.7) Update the warning to say that the old behavior will be
        # removed in 1.9.
        warnings.warn(
            (
                "\nThe format of the columns of the 'remainder' transformer in"
                " ColumnTransformer.transformers_ will change in version 1.7 to"
                " match the format of the other transformers.\nAt the moment the"
                " remainder columns are stored as indices (of type int). With the same"
                " ColumnTransformer configuration, in the future they will be stored"
                f" as {future_dtype_description}.\nTo use the new behavior now and"
                " suppress this warning, use"
                " ColumnTransformer(force_int_remainder_cols=False).\n"
            ),
            category=FutureWarning,
        )

    def _repr_pretty_(self, printer, *_):
        """Override display in ipython console, otherwise the class name is shown."""
        printer.text(repr(self.data))


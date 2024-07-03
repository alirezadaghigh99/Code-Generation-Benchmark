class ViewField(ViewExpression):
    """A :class:`ViewExpression` that refers to a field or embedded field of a
    document.

    You can use
    `dot notation <https://docs.mongodb.com/manual/core/document/#dot-notation>`_
    to refer to subfields of embedded objects within fields.

    When you create a :class:`ViewField` using a string field like
    ``ViewField("embedded.field.name")``, the meaning of this field is
    interpreted relative to the context in which the :class:`ViewField` object
    is used. For example, when passed to the :meth:`ViewExpression.map` method,
    this object will refer to the ``embedded.field.name`` object of the array
    element being processed.

    In other cases, you may wish to create a :class:`ViewField` that always
    refers to the root document. You can do this by prepending ``"$"`` to the
    name of the field, as in ``ViewField("$embedded.field.name")``.

    Examples::

        from fiftyone import ViewField as F

        # Reference the root of the current context
        F()

        # Reference the `ground_truth` field in the current context
        F("ground_truth")

        # Reference the `label` field of the `ground_truth` object in the
        # current context
        F("ground_truth.label")

        # Reference the root document in any context
        F("$")

        # Reference the `label` field of the root document in any context
        F("$label")

        # Reference the `label` field of the `ground_truth` object in the root
        # document in any context
        F("$ground_truth.label")

    .. automethod:: __eq__
    .. automethod:: __ge__
    .. automethod:: __gt__
    .. automethod:: __le__
    .. automethod:: __lt__
    .. automethod:: __ne__
    .. automethod:: __and__
    .. automethod:: __invert__
    .. automethod:: __or__
    .. automethod:: __abs__
    .. automethod:: __add__
    .. automethod:: __ceil__
    .. automethod:: __floor__
    .. automethod:: __round__
    .. automethod:: __mod__
    .. automethod:: __mul__
    .. automethod:: __pow__
    .. automethod:: __sub__
    .. automethod:: __truediv__
    .. automethod:: __getitem__

    Args:
        name (None): the name of the field, with an optional "$" prepended if
            you wish to freeze this field to the root document
    """

    def __init__(self, name=None):
        if name is None:
            name = ""

        should_freeze = name.startswith("$")
        if should_freeze:
            name = name[1:]

        super().__init__(name)

        if should_freeze:
            self._freeze_prefix("")

    def __deepcopy__(self, memo):
        obj = self.__class__()
        obj._expr = deepcopy(self._expr, memo)
        obj._prefix = deepcopy(self._prefix, memo)
        return obj

    def to_mongo(self, prefix=None):
        """Returns a MongoDB representation of the field.

        Args:
            prefix (None): an optional prefix to prepend to the field name

        Returns:
            a string
        """
        if self.is_frozen:
            prefix = self._prefix

        if prefix:
            return prefix + "." + self._expr if self._expr else prefix

        if self._expr:
            return "$" + self._expr

        if self.is_frozen:
            return "$$ROOT"

        return "$$CURRENT"class ViewField(ViewExpression):
    """A :class:`ViewExpression` that refers to a field or embedded field of a
    document.

    You can use
    `dot notation <https://docs.mongodb.com/manual/core/document/#dot-notation>`_
    to refer to subfields of embedded objects within fields.

    When you create a :class:`ViewField` using a string field like
    ``ViewField("embedded.field.name")``, the meaning of this field is
    interpreted relative to the context in which the :class:`ViewField` object
    is used. For example, when passed to the :meth:`ViewExpression.map` method,
    this object will refer to the ``embedded.field.name`` object of the array
    element being processed.

    In other cases, you may wish to create a :class:`ViewField` that always
    refers to the root document. You can do this by prepending ``"$"`` to the
    name of the field, as in ``ViewField("$embedded.field.name")``.

    Examples::

        from fiftyone import ViewField as F

        # Reference the root of the current context
        F()

        # Reference the `ground_truth` field in the current context
        F("ground_truth")

        # Reference the `label` field of the `ground_truth` object in the
        # current context
        F("ground_truth.label")

        # Reference the root document in any context
        F("$")

        # Reference the `label` field of the root document in any context
        F("$label")

        # Reference the `label` field of the `ground_truth` object in the root
        # document in any context
        F("$ground_truth.label")

    .. automethod:: __eq__
    .. automethod:: __ge__
    .. automethod:: __gt__
    .. automethod:: __le__
    .. automethod:: __lt__
    .. automethod:: __ne__
    .. automethod:: __and__
    .. automethod:: __invert__
    .. automethod:: __or__
    .. automethod:: __abs__
    .. automethod:: __add__
    .. automethod:: __ceil__
    .. automethod:: __floor__
    .. automethod:: __round__
    .. automethod:: __mod__
    .. automethod:: __mul__
    .. automethod:: __pow__
    .. automethod:: __sub__
    .. automethod:: __truediv__
    .. automethod:: __getitem__

    Args:
        name (None): the name of the field, with an optional "$" prepended if
            you wish to freeze this field to the root document
    """

    def __init__(self, name=None):
        if name is None:
            name = ""

        should_freeze = name.startswith("$")
        if should_freeze:
            name = name[1:]

        super().__init__(name)

        if should_freeze:
            self._freeze_prefix("")

    def __deepcopy__(self, memo):
        obj = self.__class__()
        obj._expr = deepcopy(self._expr, memo)
        obj._prefix = deepcopy(self._prefix, memo)
        return obj

    def to_mongo(self, prefix=None):
        """Returns a MongoDB representation of the field.

        Args:
            prefix (None): an optional prefix to prepend to the field name

        Returns:
            a string
        """
        if self.is_frozen:
            prefix = self._prefix

        if prefix:
            return prefix + "." + self._expr if self._expr else prefix

        if self._expr:
            return "$" + self._expr

        if self.is_frozen:
            return "$$ROOT"

        return "$$CURRENT"
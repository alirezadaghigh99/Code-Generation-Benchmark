class Space(AppComponent):
    """Document for configuration of a Space in the App.

    Args:
        component_id: the component's ID
        children: the list of :class:`Component` children of this space, if any
        orientation (["horizontal", "vertical"]): the orientation of this
            space's children
        active_child: the ``component_id`` of this space's currently active
            child
        sizes: the ordered list of relative sizes for children of a space in
            ``[0, 1]``
    """

    meta = {"strict": False, "allow_inheritance": True}

    children = ListField(
        EmbeddedDocumentField(AppComponent),
        validation=_validate_children,
    )
    orientation = StringField(choices=["horizontal", "vertical"], default=None)
    active_child = StringField(default=None)
    sizes = ListField(FloatField(), default=None)

    # Private name field and read-only 'name' property.
    #   Only the top-level child of a WorkspaceDocument should have a name.
    _name = StringField(default=None)

    @property
    def name(self):
        return self._name


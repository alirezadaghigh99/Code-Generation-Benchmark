def setup_modifier_factory():
    ModifierFactory.refresh()
    assert ModifierFactory._loaded, "ModifierFactory not loaded"


def build_transform_gen(
    cfg: CfgNode, is_train: bool, tfm_gen_repr_list: Optional[List[str]] = None
) -> List[d2T.Transform]:
    """
    This function builds a list of TransformGen or Transform objects using a list of
    strings (`tfm_gen_repr_list). If list is not provided, cfg.D2GO_DATA.AUG_OPS.TRAIN/TEST is used.
    Each string (aka. `tfm_gen_repr`) will be split into `name` and `arg_str` (separated by "::");
    the `name` will be used to lookup the registry while `arg_str` will be used as argument.

    Each function in registry needs to take `cfg`, `arg_str` and `is_train` as
    input, and return a list of TransformGen or Transform objects.
    """
    tfm_gen_repr_list = tfm_gen_repr_list or (
        cfg.D2GO_DATA.AUG_OPS.TRAIN if is_train else cfg.D2GO_DATA.AUG_OPS.TEST
    )
    tfm_gens = [
        TRANSFORM_OP_REGISTRY.get(name)(cfg, arg_str, is_train)
        for name, arg_str in [
            parse_tfm_gen_repr(tfm_gen_repr) for tfm_gen_repr in tfm_gen_repr_list
        ]
    ]
    assert all(isinstance(gens, list) for gens in tfm_gens)
    tfm_gens = [gen for gens in tfm_gens for gen in gens]
    assert all(isinstance(gen, (d2T.Transform, d2T.TransformGen)) for gen in tfm_gens)

    return tfm_gens
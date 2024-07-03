def terms_from_trace(tr):
    """Helper function to extract elbo components from execution traces."""
    # data structure containing densities, measures, scales, and identification
    # of free variables as either product (plate) variables or sum (measure) variables
    terms = {
        "log_factors": [],
        "log_measures": [],
        "scale": to_funsor(1.0),
        "plate_vars": frozenset(),
        "measure_vars": frozenset(),
        "plate_to_step": dict(),
    }
    for name, node in tr.nodes.items():
        # add markov dimensions to the plate_to_step dictionary
        if node["type"] == "markov_chain":
            terms["plate_to_step"][node["name"]] = node["value"]
            # ensure previous step variables are added to measure_vars
            for step in node["value"]:
                terms["measure_vars"] |= frozenset(
                    {
                        var
                        for var in step[1:-1]
                        if tr.nodes[var]["funsor"].get("log_measure", None) is not None
                    }
                )
        if (
            node["type"] != "sample"
            or type(node["fn"]).__name__ == "_Subsample"
            or node["infer"].get("_do_not_score", False)
        ):
            continue
        # grab plate dimensions from the cond_indep_stack
        terms["plate_vars"] |= frozenset(
            f.name for f in node["cond_indep_stack"] if f.vectorized
        )
        # grab the log-measure, found only at sites that are not replayed or observed
        if node["funsor"].get("log_measure", None) is not None:
            terms["log_measures"].append(node["funsor"]["log_measure"])
            # sum (measure) variables: the fresh non-plate variables at a site
            terms["measure_vars"] |= (
                frozenset(node["funsor"]["value"].inputs) | {name}
            ) - terms["plate_vars"]
        # grab the scale, assuming a common subsampling scale
        if (
            node.get("replay_active", False)
            and set(node["funsor"]["log_prob"].inputs) & terms["measure_vars"]
            and float(to_data(node["funsor"]["scale"])) != 1.0
        ):
            # model site that depends on enumerated variable: common scale
            terms["scale"] = node["funsor"]["scale"]
        else:  # otherwise: default scale behavior
            node["funsor"]["log_prob"] = (
                node["funsor"]["log_prob"] * node["funsor"]["scale"]
            )
        # grab the log-density, found at all sites except those that are not replayed
        if node["is_observed"] or not node.get("replay_skipped", False):
            terms["log_factors"].append(node["funsor"]["log_prob"])
    # add plate dimensions to the plate_to_step dictionary
    terms["plate_to_step"].update(
        {plate: terms["plate_to_step"].get(plate, {}) for plate in terms["plate_vars"]}
    )
    return terms
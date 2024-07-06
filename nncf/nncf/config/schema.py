def validate_single_compression_algo_schema(single_compression_algo_dict: Dict, ref_vs_algo_schema: Dict):
    """single_compression_algo_dict must conform to BASIC_COMPRESSION_ALGO_SCHEMA (and possibly has other
    algo-specific properties"""
    algo_name = single_compression_algo_dict["algorithm"]
    if algo_name not in ref_vs_algo_schema:
        raise jsonschema.ValidationError(
            f"Incorrect algorithm name - must be one of {str(list(ref_vs_algo_schema.keys()))}"
        )
    try:
        jsonschema.validate(single_compression_algo_dict, schema=ref_vs_algo_schema[algo_name])
    except jsonschema.ValidationError as e:
        e.message = (
            f"While validating the config for algorithm '{algo_name}' , got:\n"
            + e.message
            + f"\nRefer to the algorithm subschema definition at {SCHEMA_VISUALIZATION_URL}\n"
        )
        if algo_name in ALGO_NAME_VS_README_URL:
            e.message += (
                f"or to the algorithm documentation for examples of the configs: "
                f"{ALGO_NAME_VS_README_URL[algo_name]}"
            )
        raise e

def validate_accuracy_aware_training_schema(single_compression_algo_dict: Dict):
    """
    Checks accuracy_aware_training section.
    """
    jsonschema.validate(single_compression_algo_dict, schema=ACCURACY_AWARE_TRAINING_SCHEMA)
    accuracy_aware_mode = single_compression_algo_dict.get("mode")
    if accuracy_aware_mode not in ACCURACY_AWARE_MODES_VS_SCHEMA:
        raise jsonschema.ValidationError(
            "Incorrect Accuracy Aware mode - must be one of ({})".format(
                ", ".join(ACCURACY_AWARE_MODES_VS_SCHEMA.keys())
            )
        )
    try:
        jsonschema.validate(single_compression_algo_dict, schema=ACCURACY_AWARE_MODES_VS_SCHEMA[accuracy_aware_mode])
    except Exception as e:
        raise e


def json_ai_from_problem(df: pd.DataFrame, problem_definition: Union[ProblemDefinition, dict]) -> JsonAI:
    """
    Creates a JsonAI from your raw data and problem definition. Usually you would use this when you want to subsequently edit the JsonAI, the easiest way to do this is to unload it to a dictionary via `to_dict`, modify it, and then create a new object from it using `lightwood.JsonAI.from_dict`. It's usually better to generate the JsonAI using this function rather than writing it from scratch.

    :param df: The raw data
    :param problem_definition: The manual specifications for your predictive problem

    :returns: A ``JsonAI`` object generated based on your data and problem specifications
    """ # noqa
    if not isinstance(problem_definition, ProblemDefinition):
        problem_definition = ProblemDefinition.from_dict(problem_definition)

    started = time.time()

    if problem_definition.ignore_features:
        log.info(f'Dropping features: {problem_definition.ignore_features}')
        df = df.drop(columns=problem_definition.ignore_features)

    type_information = infer_types(df, config={'engine': 'rule_based', 'pct_invalid': problem_definition.pct_invalid})
    stats = statistical_analysis(
        df, type_information.dtypes, problem_definition.to_dict(), type_information.identifiers)

    duration = time.time() - started
    if problem_definition.time_aim is not None:
        problem_definition.time_aim -= duration
        if problem_definition.time_aim < 10:
            problem_definition.time_aim = 10

    # Assume that the stuff besides encoder and mixers takes about as long as analyzing did... bad, but let's see
    if problem_definition.expected_additional_time is None:
        problem_definition.expected_additional_time = duration
    json_ai = generate_json_ai(
        type_information=type_information, statistical_analysis=stats,
        problem_definition=problem_definition)

    return json_ai
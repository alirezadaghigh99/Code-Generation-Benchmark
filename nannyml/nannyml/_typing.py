def parse(problem_type: str):
        if problem_type in 'classification_binary':
            return ProblemType.CLASSIFICATION_BINARY
        elif problem_type in 'classification_multiclass':
            return ProblemType.CLASSIFICATION_MULTICLASS
        elif problem_type in 'regression':
            return ProblemType.REGRESSION
        else:
            raise InvalidArgumentsException(
                f"unknown value for problem_type '{problem_type}'. Value should be one of "
                f"{[pt.value for pt in ProblemType]}"
            )

def class_labels(model_outputs: ModelOutputsType) -> List[str]:
    if isinstance(model_outputs, Dict):
        return sorted(list(model_outputs.keys()))
    else:
        raise InvalidArgumentsException(
            f"received object of type {type(model_outputs)}. Multiclass ModelOutputsType should be a 'Dict[str, str]'"
        )


def mark_requirement(requirement):
    return _CombinedDecorator(
        [
            *_SHARED_DECORATORS,
            pytest.mark.reqids(requirement),
        ]
    )


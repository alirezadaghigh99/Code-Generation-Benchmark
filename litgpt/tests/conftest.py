def RunIf(thunder: Optional[bool] = None, **kwargs):
    reasons, marker_kwargs = _runif_reasons(**kwargs)

    if thunder is not None:
        thunder_available = bool(RequirementCache("lightning-thunder", "thunder"))
        if thunder and not thunder_available:
            reasons.append("Thunder")
        elif not thunder and thunder_available:
            reasons.append("not Thunder")

    return pytest.mark.skipif(condition=len(reasons) > 0, reason=f"Requires: [{' + '.join(reasons)}]", **marker_kwargs)


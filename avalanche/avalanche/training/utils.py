def trigger_plugins(strategy, event, **kwargs):
    """Call plugins on a specific callback

    :return:
    """
    for p in strategy.plugins:
        if hasattr(p, event):
            getattr(p, event)(strategy, **kwargs)
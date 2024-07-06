def make_debug_pipeline(*steps, **kwargs):
    """Construct a `DebugPipeline` from the given estimators.

    This is a shorthand for the `DebugPipeline` constructor; it does not require, and does not permit, naming the
    estimators. Instead, their names will be set to the lowercase of their types automatically.

    Parameters
    ----------
    *steps : list
        List of estimators to be included in the pipeline.
    **kwargs : dict
        Additional keyword arguments passed to the `DebugPipeline` constructor.
        Possible arguments are `memory`, `verbose` and `log_callback`:

        - `memory` : str | object with the joblib.Memory interface, default=None

            Used to cache the fitted transformers of the pipeline. The last step will never be cached, even if it is a
            transformer. By default, no caching is performed. If a string is given, it is the path to the caching
            directory. Enabling caching triggers a clone of the transformers before fitting. Therefore, the transformer
            instance given to the pipeline cannot be inspected directly. Use the attribute `named_steps` or `steps` to
            inspect estimators within the pipeline. Caching the transformers is advantageous when fitting is time
            consuming.

        - `verbose` : bool, default=False

            If True, the time elapsed while fitting each step will be printed as it is completed.

        - `log_callback` : str | Callable | None, default=None.

            The callback function that logs information in between each intermediate step. If set to `"default"`,
            `default_log_callback` is used.

    Returns
    -------
    DebugPipeline
        Instance with given steps, `memory`, `verbose` and `log_callback`.

    Examples
    --------
    ```py
    from sklearn.naive_bayes import GaussianNB
    from sklearn.preprocessing import StandardScaler

    make_debug_pipeline(StandardScaler(), GaussianNB(priors=None))
    # DebugPipeline(steps=[("standardscaler", StandardScaler()),
    #                 ("gaussiannb", GaussianNB())])
    ```

    See Also
    --------
    sklego.pipeline.DebugPipeline : Class for creating a pipeline of transforms with a final estimator.
    """
    memory = kwargs.pop("memory", None)
    verbose = kwargs.pop("verbose", False)
    log_callback = kwargs.pop("log_callback", None)
    if kwargs:
        raise TypeError('Unknown keyword arguments: "{}"'.format(list(kwargs.keys())[0]))
    return DebugPipeline(
        _name_estimators(steps),
        memory=memory,
        verbose=verbose,
        log_callback=log_callback,
    )


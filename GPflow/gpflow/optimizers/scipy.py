def minimize(
        self,
        closure: LossClosure,
        variables: Sequence[tf.Variable],
        method: Optional[str] = "L-BFGS-B",
        step_callback: Optional[StepCallback] = None,
        compile: bool = True,
        allow_unused_variables: bool = False,
        tf_fun_args: Optional[Mapping[str, Any]] = None,
        track_loss_history: bool = False,
        **scipy_kwargs: Any,
    ) -> OptimizeResult:
        """
        Minimize `closure`.

        Minimize is a wrapper around the `scipy.optimize.minimize` function handling the packing and
        unpacking of a list of shaped variables on the TensorFlow side vs. the flat numpy array
        required on the Scipy side.

        :param closure: A closure that re-evaluates the model, returning the loss to be minimized.
        :param variables: The list (tuple) of variables to be optimized
            (typically `model.trainable_variables`)
        :param method: The type of solver to use in SciPy. Defaults to "L-BFGS-B".
        :param step_callback: If not None, a callable that gets called once after each optimisation
            step. The callable is passed the arguments `step`, `variables`, and `values`. `step` is
            the optimisation step counter, `variables` is the list of trainable variables as above,
            and `values` is the corresponding list of tensors of matching shape that contains their
            value at this optimisation step.
        :param compile: If True, wraps the evaluation function (the passed `closure` as well as its
            gradient computation) inside a `tf.function()`, which will improve optimization speed in
            most cases.
        :param allow_unused_variables: Whether to allow variables that are not actually used in the
            closure.
        :param tf_fun_args: Arguments passed through to `tf.function()` when `compile` is True.
            For example, to enable XLA compilation::

                opt = gpflow.optimizers.Scipy()
                opt.minimize(..., compile=True, tf_fun_args=dict(jit_compile=True))
        :param track_loss_history: Whether to track the training loss history and return it in
            the optimization result.
        :param scipy_kwargs: Arguments passed through to `scipy.optimize.minimize`.
            Note that Scipy's minimize() takes a `callback` argument, but you probably want to use
            our wrapper and pass in `step_callback`.
        :returns:
            The optimization result represented as a Scipy ``OptimizeResult`` object.
            See the Scipy documentation for description of attributes.
        """
        if tf_fun_args is None:
            tf_fun_args = {}
        if not callable(closure):
            raise TypeError(
                "The 'closure' argument is expected to be a callable object."
            )  # pragma: no cover
        variables = tuple(variables)
        if not all(isinstance(v, tf.Variable) for v in variables):
            raise TypeError(
                "The 'variables' argument is expected to only contain tf.Variable instances"
                " (use model.trainable_variables, not model.trainable_parameters)"
            )  # pragma: no cover
        if not compile and len(tf_fun_args) > 0:
            raise ValueError("`tf_fun_args` should only be set when `compile` is True")
        initial_params = self.initial_parameters(variables)

        func = self.eval_func(
            closure,
            variables,
            compile=compile,
            allow_unused_variables=allow_unused_variables,
            tf_fun_args=tf_fun_args,
        )

        if step_callback is not None:
            if "callback" in scipy_kwargs:
                raise ValueError("Callback passed both via `step_callback` and `callback`")
            callback = self.callback_func(variables, step_callback)
            scipy_kwargs["callback"] = callback
        history: List[AnyNDArray] = []
        if track_loss_history:
            callback = self.loss_history_callback_func(func, history, scipy_kwargs.get("callback"))
            scipy_kwargs["callback"] = callback

        opt_result = scipy.optimize.minimize(
            func, initial_params, jac=True, method=method, **scipy_kwargs
        )

        if track_loss_history:
            opt_result["loss_history"] = history

        values = self.unpack_tensors(variables, opt_result.x)
        self.assign_tensors(variables, values)
        return opt_result


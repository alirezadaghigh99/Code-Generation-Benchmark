class ExecutionContext(object):
    """Represents the execution context of an operator.

    Operators can use the execution context to access the view, dataset, and
    selected samples, as well as to trigger other operators.

    Args:
        request_params (None): a optional dictionary of request parameters
        executor (None): an optional :class:`Executor` instance
        set_progress (None): an optional function to set the progress of the
            current operation
        delegated_operation_id (None): an optional ID of the delegated
            operation
        operator_uri (None): the unique id of the operator
        required_secrets (None): the list of required secrets from the
            plugin's definition
    """

    def __init__(
        self,
        request_params=None,
        executor=None,
        set_progress=None,
        delegated_operation_id=None,
        operator_uri=None,
        required_secrets=None,
    ):
        self.request_params = request_params or {}
        self.params = self.request_params.get("params", {})
        self.executor = executor

        self._dataset = None
        self._view = None
        self._ops = Operations(self)

        self._set_progress = set_progress
        self._delegated_operation_id = delegated_operation_id
        self._operator_uri = operator_uri
        self._secrets = {}
        self._secrets_client = PluginSecretsResolver()
        self._required_secret_keys = required_secrets
        if self._required_secret_keys:
            self._secrets_client.register_operator(
                operator_uri=self._operator_uri,
                required_secrets=self._required_secret_keys,
            )

    @property
    def dataset(self):
        """The :class:`fiftyone.core.dataset.Dataset` being operated on."""
        if self._dataset is not None:
            return self._dataset
        # Since dataset may have been renamed, always resolve the dataset by
        # id if it is available
        uid = self.request_params.get("dataset_id", None)
        if uid:
            self._dataset = focu.load_dataset(id=uid)
            # Set the dataset_name using the dataset object in case the dataset
            # has been renamed or changed since the context was created
            self.request_params["dataset_name"] = self._dataset.name
        else:
            uid = self.request_params.get("dataset_name", None)
            if uid:
                self._dataset = focu.load_dataset(name=uid)
        # TODO: refactor so that this additional reload post-load is not
        #  required
        if self._dataset is not None:
            self._dataset.reload()
        return self._dataset

    @property
    def dataset_name(self):
        """The name of the :class:`fiftyone.core.dataset.Dataset` being
        operated on.
        """
        return self.request_params.get("dataset_name", None)

    @property
    def dataset_id(self):
        """The ID of the :class:`fiftyone.core.dataset.Dataset` being operated
        on.
        """
        return self.request_params.get("dataset_id", None)

    @property
    def view(self):
        """The :class:`fiftyone.core.view.DatasetView` being operated on."""
        if self._view is not None:
            return self._view

        # Always derive the view from the context's dataset
        dataset = self.dataset
        view_name = self.request_params.get("view_name", None)
        stages = self.request_params.get("view", None)
        filters = self.request_params.get("filters", None)
        extended = self.request_params.get("extended", None)

        if dataset is None:
            return None

        if view_name is None:
            self._view = fosv.get_view(
                dataset,
                stages=stages,
                filters=filters,
                extended_stages=extended,
                reload=False,
            )
        else:
            self._view = dataset.load_saved_view(view_name)

        return self._view

    def target_view(self, param_name="view_target"):
        """The target :class:`fiftyone.core.view.DatasetView` for the operator
        being executed.
        """
        target = self.params.get(param_name, None)
        if target == "SELECTED_SAMPLES":
            return self.view.select(self.selected)
        if target == "DATASET":
            return self.dataset
        return self.view

    @property
    def has_custom_view(self):
        """Whether the operator has a custom view."""
        stages = self.request_params.get("view", None)
        filters = self.request_params.get("filters", None)
        extended = self.request_params.get("extended", None)
        has_stages = stages is not None and stages != [] and stages != {}
        has_filters = filters is not None and filters != [] and filters != {}
        has_extended = (
            extended is not None and extended != [] and extended != {}
        )
        return has_stages or has_filters or has_extended

    @property
    def selected(self):
        """The list of selected sample IDs (if any)."""
        return self.request_params.get("selected", [])

    @property
    def selected_labels(self):
        """A list of selected labels (if any).

        Items are dictionaries with the following keys:

        -   ``label_id``: the ID of the label
        -   ``sample_id``: the ID of the sample containing the label
        -   ``field``: the field name containing the label
        -   ``frame_number``: the frame number containing the label (only
            applicable to video samples)
        """
        return self.request_params.get("selected_labels", [])

    @property
    def current_sample(self):
        """The ID of the current sample being processed (if any).

        When executed via the FiftyOne App, this is set when the user opens a
        sample in the modal.
        """
        return self.request_params.get("current_sample", None)

    @property
    def delegated(self):
        """Whether delegated execution has been forced for the operation."""
        return self.request_params.get("delegated", False)

    @property
    def requesting_delegated_execution(self):
        """Whether delegated execution has been requested for the operation."""
        return self.request_params.get("request_delegation", False)

    @property
    def delegation_target(self):
        """The orchestrator to which the operation was delegated (if any)."""
        return self.request_params.get("delegation_target", None)

    @property
    def results(self):
        """A ``dict`` of results for the current operation."""
        return self.request_params.get("results", {})

    @property
    def secrets(self):
        """A read-only mapping of keys to their resolved values."""
        return SecretsDictionary(
            self._secrets,
            operator_uri=self._operator_uri,
            resolver_fn=self._secrets_client.get_secret_sync,
            required_keys=self._required_secret_keys,
        )

    @property
    def ops(self):
        """A :class:`fiftyone.operators.operations.Operations` instance that
        you can use to trigger builtin operations on the current context.
        """
        return self._ops

    def secret(self, key):
        """Retrieves the secret with the given key.

        Args:
            key: a secret key

        Returns:
            the secret value
        """
        if key not in self._secrets:
            try:
                secret = self._secrets_client.get_secret_sync(
                    key, self._operator_uri
                )
                if secret:
                    self._secrets[secret.key] = secret.value

            except KeyError:
                logging.debug(f"Failed to resolve value for secret `{key}`")
        return self._secrets.get(key, None)

    async def resolve_secret_values(self, keys, **kwargs):
        """Resolves the values of the given secrets keys.

        Args:
            keys: a list of secret keys
            **kwargs: additional keyword arguments to pass to the secrets
                client for authentication if required
        """
        if None in (self._secrets_client, keys):
            return None

        for key in keys:
            secret = await self._secrets_client.get_secret(
                key, self._operator_uri, **kwargs
            )
            if secret:
                self._secrets[secret.key] = secret.value

    def trigger(self, operator_name, params=None):
        """Triggers an invocation of the operator with the given name.

        This method is only available when the operator is invoked via the
        FiftyOne App. You can check this via ``ctx.executor``.

        Example::

            def execute(self, ctx):
                # Trigger the `reload_dataset` operator after this operator
                # finishes executing
                ctx.trigger("reload_dataset")

                # Immediately trigger the `reload_dataset` operator while a
                # generator operator is executing
                yield ctx.trigger("reload_dataset")

        Args:
            operator_name: the name of the operator
            params (None): a dictionary of parameters for the operator

        Returns:
            a :class:`fiftyone.operators.message.GeneratedMessage` containing
            instructions for the FiftyOne App to invoke the operator
        """
        if self.executor is None:
            raise ValueError("No executor available")

        return self.executor.trigger(operator_name, params)

    def log(self, message):
        """Logs a message to the browser console.

        .. note::

            This method is only available to non-delegated operators. You can
            only use this method during the execution of an operator.

        Args:
            message: a message to log

        Returns:
            a :class:`fiftyone.operators.message.GeneratedMessage` containing
            instructions for the FiftyOne App to invoke the operator
        """
        return self.trigger("console_log", {"message": message})

    def serialize(self):
        """Serializes the execution context.

        Returns:
            a JSON dict
        """
        return {
            "request_params": self.request_params,
            "params": self.params,
        }

    def to_dict(self):
        """Returns the properties of the execution context as a dict."""
        return {
            k: v for k, v in self.__dict__.items() if not k.startswith("_")
        }

    def set_progress(self, progress=None, label=None):
        """Sets the progress of the current operation.

        Args:
            progress (None): an optional float between 0 and 1 (0% to 100%)
            label (None): an optional label to display
        """
        if self._set_progress:
            self._set_progress(
                self._delegated_operation_id,
                ExecutionProgress(progress, label),
            )
        else:
            self.log(f"Progress: {progress} - {label}")class ExecutionContext(object):
    """Represents the execution context of an operator.

    Operators can use the execution context to access the view, dataset, and
    selected samples, as well as to trigger other operators.

    Args:
        request_params (None): a optional dictionary of request parameters
        executor (None): an optional :class:`Executor` instance
        set_progress (None): an optional function to set the progress of the
            current operation
        delegated_operation_id (None): an optional ID of the delegated
            operation
        operator_uri (None): the unique id of the operator
        required_secrets (None): the list of required secrets from the
            plugin's definition
    """

    def __init__(
        self,
        request_params=None,
        executor=None,
        set_progress=None,
        delegated_operation_id=None,
        operator_uri=None,
        required_secrets=None,
    ):
        self.request_params = request_params or {}
        self.params = self.request_params.get("params", {})
        self.executor = executor

        self._dataset = None
        self._view = None
        self._ops = Operations(self)

        self._set_progress = set_progress
        self._delegated_operation_id = delegated_operation_id
        self._operator_uri = operator_uri
        self._secrets = {}
        self._secrets_client = PluginSecretsResolver()
        self._required_secret_keys = required_secrets
        if self._required_secret_keys:
            self._secrets_client.register_operator(
                operator_uri=self._operator_uri,
                required_secrets=self._required_secret_keys,
            )

    @property
    def dataset(self):
        """The :class:`fiftyone.core.dataset.Dataset` being operated on."""
        if self._dataset is not None:
            return self._dataset
        # Since dataset may have been renamed, always resolve the dataset by
        # id if it is available
        uid = self.request_params.get("dataset_id", None)
        if uid:
            self._dataset = focu.load_dataset(id=uid)
            # Set the dataset_name using the dataset object in case the dataset
            # has been renamed or changed since the context was created
            self.request_params["dataset_name"] = self._dataset.name
        else:
            uid = self.request_params.get("dataset_name", None)
            if uid:
                self._dataset = focu.load_dataset(name=uid)
        # TODO: refactor so that this additional reload post-load is not
        #  required
        if self._dataset is not None:
            self._dataset.reload()
        return self._dataset

    @property
    def dataset_name(self):
        """The name of the :class:`fiftyone.core.dataset.Dataset` being
        operated on.
        """
        return self.request_params.get("dataset_name", None)

    @property
    def dataset_id(self):
        """The ID of the :class:`fiftyone.core.dataset.Dataset` being operated
        on.
        """
        return self.request_params.get("dataset_id", None)

    @property
    def view(self):
        """The :class:`fiftyone.core.view.DatasetView` being operated on."""
        if self._view is not None:
            return self._view

        # Always derive the view from the context's dataset
        dataset = self.dataset
        view_name = self.request_params.get("view_name", None)
        stages = self.request_params.get("view", None)
        filters = self.request_params.get("filters", None)
        extended = self.request_params.get("extended", None)

        if dataset is None:
            return None

        if view_name is None:
            self._view = fosv.get_view(
                dataset,
                stages=stages,
                filters=filters,
                extended_stages=extended,
                reload=False,
            )
        else:
            self._view = dataset.load_saved_view(view_name)

        return self._view

    def target_view(self, param_name="view_target"):
        """The target :class:`fiftyone.core.view.DatasetView` for the operator
        being executed.
        """
        target = self.params.get(param_name, None)
        if target == "SELECTED_SAMPLES":
            return self.view.select(self.selected)
        if target == "DATASET":
            return self.dataset
        return self.view

    @property
    def has_custom_view(self):
        """Whether the operator has a custom view."""
        stages = self.request_params.get("view", None)
        filters = self.request_params.get("filters", None)
        extended = self.request_params.get("extended", None)
        has_stages = stages is not None and stages != [] and stages != {}
        has_filters = filters is not None and filters != [] and filters != {}
        has_extended = (
            extended is not None and extended != [] and extended != {}
        )
        return has_stages or has_filters or has_extended

    @property
    def selected(self):
        """The list of selected sample IDs (if any)."""
        return self.request_params.get("selected", [])

    @property
    def selected_labels(self):
        """A list of selected labels (if any).

        Items are dictionaries with the following keys:

        -   ``label_id``: the ID of the label
        -   ``sample_id``: the ID of the sample containing the label
        -   ``field``: the field name containing the label
        -   ``frame_number``: the frame number containing the label (only
            applicable to video samples)
        """
        return self.request_params.get("selected_labels", [])

    @property
    def current_sample(self):
        """The ID of the current sample being processed (if any).

        When executed via the FiftyOne App, this is set when the user opens a
        sample in the modal.
        """
        return self.request_params.get("current_sample", None)

    @property
    def delegated(self):
        """Whether delegated execution has been forced for the operation."""
        return self.request_params.get("delegated", False)

    @property
    def requesting_delegated_execution(self):
        """Whether delegated execution has been requested for the operation."""
        return self.request_params.get("request_delegation", False)

    @property
    def delegation_target(self):
        """The orchestrator to which the operation was delegated (if any)."""
        return self.request_params.get("delegation_target", None)

    @property
    def results(self):
        """A ``dict`` of results for the current operation."""
        return self.request_params.get("results", {})

    @property
    def secrets(self):
        """A read-only mapping of keys to their resolved values."""
        return SecretsDictionary(
            self._secrets,
            operator_uri=self._operator_uri,
            resolver_fn=self._secrets_client.get_secret_sync,
            required_keys=self._required_secret_keys,
        )

    @property
    def ops(self):
        """A :class:`fiftyone.operators.operations.Operations` instance that
        you can use to trigger builtin operations on the current context.
        """
        return self._ops

    def secret(self, key):
        """Retrieves the secret with the given key.

        Args:
            key: a secret key

        Returns:
            the secret value
        """
        if key not in self._secrets:
            try:
                secret = self._secrets_client.get_secret_sync(
                    key, self._operator_uri
                )
                if secret:
                    self._secrets[secret.key] = secret.value

            except KeyError:
                logging.debug(f"Failed to resolve value for secret `{key}`")
        return self._secrets.get(key, None)

    async def resolve_secret_values(self, keys, **kwargs):
        """Resolves the values of the given secrets keys.

        Args:
            keys: a list of secret keys
            **kwargs: additional keyword arguments to pass to the secrets
                client for authentication if required
        """
        if None in (self._secrets_client, keys):
            return None

        for key in keys:
            secret = await self._secrets_client.get_secret(
                key, self._operator_uri, **kwargs
            )
            if secret:
                self._secrets[secret.key] = secret.value

    def trigger(self, operator_name, params=None):
        """Triggers an invocation of the operator with the given name.

        This method is only available when the operator is invoked via the
        FiftyOne App. You can check this via ``ctx.executor``.

        Example::

            def execute(self, ctx):
                # Trigger the `reload_dataset` operator after this operator
                # finishes executing
                ctx.trigger("reload_dataset")

                # Immediately trigger the `reload_dataset` operator while a
                # generator operator is executing
                yield ctx.trigger("reload_dataset")

        Args:
            operator_name: the name of the operator
            params (None): a dictionary of parameters for the operator

        Returns:
            a :class:`fiftyone.operators.message.GeneratedMessage` containing
            instructions for the FiftyOne App to invoke the operator
        """
        if self.executor is None:
            raise ValueError("No executor available")

        return self.executor.trigger(operator_name, params)

    def log(self, message):
        """Logs a message to the browser console.

        .. note::

            This method is only available to non-delegated operators. You can
            only use this method during the execution of an operator.

        Args:
            message: a message to log

        Returns:
            a :class:`fiftyone.operators.message.GeneratedMessage` containing
            instructions for the FiftyOne App to invoke the operator
        """
        return self.trigger("console_log", {"message": message})

    def serialize(self):
        """Serializes the execution context.

        Returns:
            a JSON dict
        """
        return {
            "request_params": self.request_params,
            "params": self.params,
        }

    def to_dict(self):
        """Returns the properties of the execution context as a dict."""
        return {
            k: v for k, v in self.__dict__.items() if not k.startswith("_")
        }

    def set_progress(self, progress=None, label=None):
        """Sets the progress of the current operation.

        Args:
            progress (None): an optional float between 0 and 1 (0% to 100%)
            label (None): an optional label to display
        """
        if self._set_progress:
            self._set_progress(
                self._delegated_operation_id,
                ExecutionProgress(progress, label),
            )
        else:
            self.log(f"Progress: {progress} - {label}")
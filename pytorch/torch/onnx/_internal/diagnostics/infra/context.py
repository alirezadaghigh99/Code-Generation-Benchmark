class Diagnostic:
    rule: infra.Rule
    level: infra.Level
    message: Optional[str] = None
    locations: List[infra.Location] = dataclasses.field(default_factory=list)
    stacks: List[infra.Stack] = dataclasses.field(default_factory=list)
    graphs: List[infra.Graph] = dataclasses.field(default_factory=list)
    thread_flow_locations: List[infra.ThreadFlowLocation] = dataclasses.field(
        default_factory=list
    )
    additional_messages: List[str] = dataclasses.field(default_factory=list)
    tags: List[infra.Tag] = dataclasses.field(default_factory=list)
    source_exception: Optional[Exception] = None
    """The exception that caused this diagnostic to be created."""
    logger: logging.Logger = dataclasses.field(init=False, default=diagnostic_logger)
    """The logger for this diagnostic. Defaults to 'diagnostic_logger' which has the same
    log level setting with `DiagnosticOptions.verbosity_level`."""
    _current_log_section_depth: int = 0

    def __post_init__(self) -> None:
        pass

    def sarif(self) -> sarif.Result:
        """Returns the SARIF Result representation of this diagnostic."""
        message = self.message or self.rule.message_default_template
        if self.additional_messages:
            additional_message = "\n".join(self.additional_messages)
            message_markdown = (
                f"{message}\n\n## Additional Message:\n\n{additional_message}"
            )
        else:
            message_markdown = message

        kind: Literal["informational", "fail"] = (
            "informational" if self.level == infra.Level.NONE else "fail"
        )

        sarif_result = sarif.Result(
            message=sarif.Message(text=message, markdown=message_markdown),
            level=self.level.name.lower(),  # type: ignore[arg-type]
            rule_id=self.rule.id,
            kind=kind,
        )
        sarif_result.locations = [location.sarif() for location in self.locations]
        sarif_result.stacks = [stack.sarif() for stack in self.stacks]
        sarif_result.graphs = [graph.sarif() for graph in self.graphs]
        sarif_result.code_flows = [
            sarif.CodeFlow(
                thread_flows=[
                    sarif.ThreadFlow(
                        locations=[loc.sarif() for loc in self.thread_flow_locations]
                    )
                ]
            )
        ]
        sarif_result.properties = sarif.PropertyBag(
            tags=[tag.value for tag in self.tags]
        )
        return sarif_result

    def with_location(self: Self, location: infra.Location) -> Self:
        """Adds a location to the diagnostic."""
        self.locations.append(location)
        return self

    def with_thread_flow_location(
        self: Self, location: infra.ThreadFlowLocation
    ) -> Self:
        """Adds a thread flow location to the diagnostic."""
        self.thread_flow_locations.append(location)
        return self

    def with_stack(self: Self, stack: infra.Stack) -> Self:
        """Adds a stack to the diagnostic."""
        self.stacks.append(stack)
        return self

    def with_graph(self: Self, graph: infra.Graph) -> Self:
        """Adds a graph to the diagnostic."""
        self.graphs.append(graph)
        return self

    @contextlib.contextmanager
    def log_section(
        self, level: int, message: str, *args, **kwargs
    ) -> Generator[None, None, None]:
        """
        Context manager for a section of log messages, denoted by a title message and increased indentation.

        Same api as `logging.Logger.log`.

        This context manager logs the given title at the specified log level, increases the current
        section depth for subsequent log messages, and ensures that the section depth is decreased
        again when exiting the context.

        Args:
            level: The log level.
            message: The title message to log.
            *args: The arguments to the message. Use `LazyString` to defer the
                expensive evaluation of the arguments until the message is actually logged.
            **kwargs: The keyword arguments for `logging.Logger.log`.

        Yields:
            None: This context manager does not yield any value.

        Example:
            >>> with DiagnosticContext("DummyContext", "1.0"):
            ...     rule = infra.Rule("RuleID", "DummyRule", "Rule message")
            ...     diagnostic = Diagnostic(rule, infra.Level.WARNING)
            ...     with diagnostic.log_section(logging.INFO, "My Section"):
            ...         diagnostic.log(logging.INFO, "My Message")
            ...         with diagnostic.log_section(logging.INFO, "My Subsection"):
            ...             diagnostic.log(logging.INFO, "My Submessage")
            ...     diagnostic.additional_messages
            ['## My Section', 'My Message', '### My Subsection', 'My Submessage']
        """
        if self.logger.isEnabledFor(level):
            indented_format_message = (
                f"##{'#' * self._current_log_section_depth } {message}"
            )
            self.log(
                level,
                indented_format_message,
                *args,
                **kwargs,
            )
        self._current_log_section_depth += 1
        try:
            yield
        finally:
            self._current_log_section_depth -= 1

    def log(self, level: int, message: str, *args, **kwargs) -> None:
        """Logs a message within the diagnostic. Same api as `logging.Logger.log`.

        If logger is not enabled for the given level, the message will not be logged.
        Otherwise, the message will be logged and also added to the diagnostic's additional_messages.

        The default setting for `DiagnosticOptions.verbosity_level` is `logging.INFO`. Based on this default,
        the log level recommendations are as follows. If you've set a different default verbosity level in your
        application, please adjust accordingly:

        - logging.ERROR: Log any events leading to application failure.
        - logging.WARNING: Log events that might result in application issues or failures, although not guaranteed.
        - logging.INFO: Log general useful information, ensuring minimal performance overhead.
        - logging.DEBUG: Log detailed debug information, which might affect performance when logged.

        Args:
            level: The log level.
            message: The message to log.
            *args: The arguments to the message. Use `LazyString` to defer the
                expensive evaluation of the arguments until the message is actually logged.
            **kwargs: The keyword arguments for `logging.Logger.log`.
        """
        if self.logger.isEnabledFor(level):
            formatted_message = message % args
            self.logger.log(level, formatted_message, **kwargs)
            self.additional_messages.append(formatted_message)

    def debug(self, message: str, *args, **kwargs) -> None:
        """Logs a debug message within the diagnostic. Same api as logging.Logger.debug.

        Checkout `log` for more details.
        """
        self.log(logging.DEBUG, message, *args, **kwargs)

    def info(self, message: str, *args, **kwargs) -> None:
        """Logs an info message within the diagnostic. Same api as logging.Logger.info.

        Checkout `log` for more details.
        """
        self.log(logging.INFO, message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs) -> None:
        """Logs a warning message within the diagnostic. Same api as logging.Logger.warning.

        Checkout `log` for more details.
        """
        self.log(logging.WARNING, message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs) -> None:
        """Logs an error message within the diagnostic. Same api as logging.Logger.error.

        Checkout `log` for more details.
        """
        self.log(logging.ERROR, message, *args, **kwargs)

    def log_source_exception(self, level: int, exception: Exception) -> None:
        """Logs a source exception within the diagnostic.

        Invokes `log_section` and `log` to log the exception in markdown section format.
        """
        self.source_exception = exception
        with self.log_section(level, "Exception log"):
            self.log(level, "%s", formatter.lazy_format_exception(exception))

    def record_python_call_stack(self, frames_to_skip: int) -> infra.Stack:
        """Records the current Python call stack."""
        frames_to_skip += 1  # Skip this function.
        stack = utils.python_call_stack(frames_to_skip=frames_to_skip)
        self.with_stack(stack)
        if len(stack.frames) > 0:
            self.with_location(stack.frames[0].location)
        return stack

    def record_python_call(
        self,
        fn: Callable,
        state: Mapping[str, str],
        message: Optional[str] = None,
        frames_to_skip: int = 0,
    ) -> infra.ThreadFlowLocation:
        """Records a python call as one thread flow step."""
        frames_to_skip += 1  # Skip this function.
        stack = utils.python_call_stack(frames_to_skip=frames_to_skip, frames_to_log=5)
        location = utils.function_location(fn)
        location.message = message
        # Add function location to the top of the stack.
        stack.frames.insert(0, infra.StackFrame(location=location))
        thread_flow_location = infra.ThreadFlowLocation(
            location=location,
            state=state,
            index=len(self.thread_flow_locations),
            stack=stack,
        )
        self.with_thread_flow_location(thread_flow_location)
        return thread_flow_locationclass Diagnostic:
    rule: infra.Rule
    level: infra.Level
    message: Optional[str] = None
    locations: List[infra.Location] = dataclasses.field(default_factory=list)
    stacks: List[infra.Stack] = dataclasses.field(default_factory=list)
    graphs: List[infra.Graph] = dataclasses.field(default_factory=list)
    thread_flow_locations: List[infra.ThreadFlowLocation] = dataclasses.field(
        default_factory=list
    )
    additional_messages: List[str] = dataclasses.field(default_factory=list)
    tags: List[infra.Tag] = dataclasses.field(default_factory=list)
    source_exception: Optional[Exception] = None
    """The exception that caused this diagnostic to be created."""
    logger: logging.Logger = dataclasses.field(init=False, default=diagnostic_logger)
    """The logger for this diagnostic. Defaults to 'diagnostic_logger' which has the same
    log level setting with `DiagnosticOptions.verbosity_level`."""
    _current_log_section_depth: int = 0

    def __post_init__(self) -> None:
        pass

    def sarif(self) -> sarif.Result:
        """Returns the SARIF Result representation of this diagnostic."""
        message = self.message or self.rule.message_default_template
        if self.additional_messages:
            additional_message = "\n".join(self.additional_messages)
            message_markdown = (
                f"{message}\n\n## Additional Message:\n\n{additional_message}"
            )
        else:
            message_markdown = message

        kind: Literal["informational", "fail"] = (
            "informational" if self.level == infra.Level.NONE else "fail"
        )

        sarif_result = sarif.Result(
            message=sarif.Message(text=message, markdown=message_markdown),
            level=self.level.name.lower(),  # type: ignore[arg-type]
            rule_id=self.rule.id,
            kind=kind,
        )
        sarif_result.locations = [location.sarif() for location in self.locations]
        sarif_result.stacks = [stack.sarif() for stack in self.stacks]
        sarif_result.graphs = [graph.sarif() for graph in self.graphs]
        sarif_result.code_flows = [
            sarif.CodeFlow(
                thread_flows=[
                    sarif.ThreadFlow(
                        locations=[loc.sarif() for loc in self.thread_flow_locations]
                    )
                ]
            )
        ]
        sarif_result.properties = sarif.PropertyBag(
            tags=[tag.value for tag in self.tags]
        )
        return sarif_result

    def with_location(self: Self, location: infra.Location) -> Self:
        """Adds a location to the diagnostic."""
        self.locations.append(location)
        return self

    def with_thread_flow_location(
        self: Self, location: infra.ThreadFlowLocation
    ) -> Self:
        """Adds a thread flow location to the diagnostic."""
        self.thread_flow_locations.append(location)
        return self

    def with_stack(self: Self, stack: infra.Stack) -> Self:
        """Adds a stack to the diagnostic."""
        self.stacks.append(stack)
        return self

    def with_graph(self: Self, graph: infra.Graph) -> Self:
        """Adds a graph to the diagnostic."""
        self.graphs.append(graph)
        return self

    @contextlib.contextmanager
    def log_section(
        self, level: int, message: str, *args, **kwargs
    ) -> Generator[None, None, None]:
        """
        Context manager for a section of log messages, denoted by a title message and increased indentation.

        Same api as `logging.Logger.log`.

        This context manager logs the given title at the specified log level, increases the current
        section depth for subsequent log messages, and ensures that the section depth is decreased
        again when exiting the context.

        Args:
            level: The log level.
            message: The title message to log.
            *args: The arguments to the message. Use `LazyString` to defer the
                expensive evaluation of the arguments until the message is actually logged.
            **kwargs: The keyword arguments for `logging.Logger.log`.

        Yields:
            None: This context manager does not yield any value.

        Example:
            >>> with DiagnosticContext("DummyContext", "1.0"):
            ...     rule = infra.Rule("RuleID", "DummyRule", "Rule message")
            ...     diagnostic = Diagnostic(rule, infra.Level.WARNING)
            ...     with diagnostic.log_section(logging.INFO, "My Section"):
            ...         diagnostic.log(logging.INFO, "My Message")
            ...         with diagnostic.log_section(logging.INFO, "My Subsection"):
            ...             diagnostic.log(logging.INFO, "My Submessage")
            ...     diagnostic.additional_messages
            ['## My Section', 'My Message', '### My Subsection', 'My Submessage']
        """
        if self.logger.isEnabledFor(level):
            indented_format_message = (
                f"##{'#' * self._current_log_section_depth } {message}"
            )
            self.log(
                level,
                indented_format_message,
                *args,
                **kwargs,
            )
        self._current_log_section_depth += 1
        try:
            yield
        finally:
            self._current_log_section_depth -= 1

    def log(self, level: int, message: str, *args, **kwargs) -> None:
        """Logs a message within the diagnostic. Same api as `logging.Logger.log`.

        If logger is not enabled for the given level, the message will not be logged.
        Otherwise, the message will be logged and also added to the diagnostic's additional_messages.

        The default setting for `DiagnosticOptions.verbosity_level` is `logging.INFO`. Based on this default,
        the log level recommendations are as follows. If you've set a different default verbosity level in your
        application, please adjust accordingly:

        - logging.ERROR: Log any events leading to application failure.
        - logging.WARNING: Log events that might result in application issues or failures, although not guaranteed.
        - logging.INFO: Log general useful information, ensuring minimal performance overhead.
        - logging.DEBUG: Log detailed debug information, which might affect performance when logged.

        Args:
            level: The log level.
            message: The message to log.
            *args: The arguments to the message. Use `LazyString` to defer the
                expensive evaluation of the arguments until the message is actually logged.
            **kwargs: The keyword arguments for `logging.Logger.log`.
        """
        if self.logger.isEnabledFor(level):
            formatted_message = message % args
            self.logger.log(level, formatted_message, **kwargs)
            self.additional_messages.append(formatted_message)

    def debug(self, message: str, *args, **kwargs) -> None:
        """Logs a debug message within the diagnostic. Same api as logging.Logger.debug.

        Checkout `log` for more details.
        """
        self.log(logging.DEBUG, message, *args, **kwargs)

    def info(self, message: str, *args, **kwargs) -> None:
        """Logs an info message within the diagnostic. Same api as logging.Logger.info.

        Checkout `log` for more details.
        """
        self.log(logging.INFO, message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs) -> None:
        """Logs a warning message within the diagnostic. Same api as logging.Logger.warning.

        Checkout `log` for more details.
        """
        self.log(logging.WARNING, message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs) -> None:
        """Logs an error message within the diagnostic. Same api as logging.Logger.error.

        Checkout `log` for more details.
        """
        self.log(logging.ERROR, message, *args, **kwargs)

    def log_source_exception(self, level: int, exception: Exception) -> None:
        """Logs a source exception within the diagnostic.

        Invokes `log_section` and `log` to log the exception in markdown section format.
        """
        self.source_exception = exception
        with self.log_section(level, "Exception log"):
            self.log(level, "%s", formatter.lazy_format_exception(exception))

    def record_python_call_stack(self, frames_to_skip: int) -> infra.Stack:
        """Records the current Python call stack."""
        frames_to_skip += 1  # Skip this function.
        stack = utils.python_call_stack(frames_to_skip=frames_to_skip)
        self.with_stack(stack)
        if len(stack.frames) > 0:
            self.with_location(stack.frames[0].location)
        return stack

    def record_python_call(
        self,
        fn: Callable,
        state: Mapping[str, str],
        message: Optional[str] = None,
        frames_to_skip: int = 0,
    ) -> infra.ThreadFlowLocation:
        """Records a python call as one thread flow step."""
        frames_to_skip += 1  # Skip this function.
        stack = utils.python_call_stack(frames_to_skip=frames_to_skip, frames_to_log=5)
        location = utils.function_location(fn)
        location.message = message
        # Add function location to the top of the stack.
        stack.frames.insert(0, infra.StackFrame(location=location))
        thread_flow_location = infra.ThreadFlowLocation(
            location=location,
            state=state,
            index=len(self.thread_flow_locations),
            stack=stack,
        )
        self.with_thread_flow_location(thread_flow_location)
        return thread_flow_location
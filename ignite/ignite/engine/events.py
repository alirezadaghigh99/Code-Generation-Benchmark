class State:
    """An object that is used to pass internal and user-defined state between event handlers. By default, state
    contains the following attributes:

    .. code-block:: python

        state.iteration         # 1-based, the first iteration is 1
        state.epoch             # 1-based, the first epoch is 1
        state.seed              # seed to set at each epoch
        state.dataloader        # data passed to engine
        state.epoch_length      # optional length of an epoch
        state.max_epochs        # number of epochs to run
        state.max_iters         # number of iterations to run
        state.batch             # batch passed to `process_function`
        state.output            # output of `process_function` after a single iteration
        state.metrics           # dictionary with defined metrics if any
        state.times             # dictionary with total and per-epoch times fetched on
                                # keys: Events.EPOCH_COMPLETED.name and Events.COMPLETED.name

    Args:
        kwargs: keyword arguments to be defined as State attributes.
    """

    event_to_attr: Dict[Union[str, "Events", "CallableEventWithFilter"], str] = {
        Events.GET_BATCH_STARTED: "iteration",
        Events.GET_BATCH_COMPLETED: "iteration",
        Events.ITERATION_STARTED: "iteration",
        Events.ITERATION_COMPLETED: "iteration",
        Events.EPOCH_STARTED: "epoch",
        Events.EPOCH_COMPLETED: "epoch",
        Events.STARTED: "epoch",
        Events.COMPLETED: "epoch",
    }

    def __init__(self, **kwargs: Any) -> None:
        self.iteration = 0
        self.epoch = 0
        self.epoch_length: Optional[int] = None
        self.max_epochs: Optional[int] = None
        self.max_iters: Optional[int] = None
        self.output: Optional[int] = None
        self.batch: Optional[int] = None
        self.metrics: Dict[str, Any] = {}
        self.dataloader: Optional[Union[DataLoader, Iterable[Any]]] = None
        self.seed: Optional[int] = None
        self.times: Dict[str, Optional[float]] = {
            Events.EPOCH_COMPLETED.name: None,
            Events.COMPLETED.name: None,
        }

        for k, v in kwargs.items():
            setattr(self, k, v)

        self._update_attrs()

    def _update_attrs(self) -> None:
        for value in self.event_to_attr.values():
            if not hasattr(self, value):
                setattr(self, value, 0)

    def get_event_attrib_value(self, event_name: Union[str, Events, CallableEventWithFilter]) -> int:
        """Get the value of Event attribute with given `event_name`."""
        if event_name not in State.event_to_attr:
            raise RuntimeError(f"Unknown event name '{event_name}'")
        return getattr(self, State.event_to_attr[event_name])

    def __repr__(self) -> str:
        s = "State:\n"
        for attr, value in self.__dict__.items():
            if not isinstance(value, (numbers.Number, str)):
                value = type(value)
            s += f"\t{attr}: {value}\n"
        return s

class EventsList:
    """Collection of events stacked by operator `__or__`.

    .. code-block:: python

        events = Events.STARTED | Events.COMPLETED
        events |= Events.ITERATION_STARTED(every=3)

        engine = ...

        @engine.on(events)
        def call_on_events(engine):
            # do something

    or

    .. code-block:: python

        @engine.on(Events.STARTED | Events.COMPLETED | Events.ITERATION_STARTED(every=3))
        def call_on_events(engine):
            # do something

    """

    def __init__(self) -> None:
        self._events: List[Union[Events, CallableEventWithFilter]] = []

    def _append(self, event: Union[Events, CallableEventWithFilter]) -> None:
        if not isinstance(event, (Events, CallableEventWithFilter)):
            raise TypeError(f"Argument event should be Events or CallableEventWithFilter, got: {type(event)}")
        self._events.append(event)

    def __getitem__(self, item: int) -> Union[Events, CallableEventWithFilter]:
        return self._events[item]

    def __iter__(self) -> Iterator[Union[Events, CallableEventWithFilter]]:
        return iter(self._events)

    def __len__(self) -> int:
        return len(self._events)

    def __or__(self, other: Union[Events, CallableEventWithFilter]) -> "EventsList":
        self._append(event=other)
        return self

class EventsList:
    """Collection of events stacked by operator `__or__`.

    .. code-block:: python

        events = Events.STARTED | Events.COMPLETED
        events |= Events.ITERATION_STARTED(every=3)

        engine = ...

        @engine.on(events)
        def call_on_events(engine):
            # do something

    or

    .. code-block:: python

        @engine.on(Events.STARTED | Events.COMPLETED | Events.ITERATION_STARTED(every=3))
        def call_on_events(engine):
            # do something

    """

    def __init__(self) -> None:
        self._events: List[Union[Events, CallableEventWithFilter]] = []

    def _append(self, event: Union[Events, CallableEventWithFilter]) -> None:
        if not isinstance(event, (Events, CallableEventWithFilter)):
            raise TypeError(f"Argument event should be Events or CallableEventWithFilter, got: {type(event)}")
        self._events.append(event)

    def __getitem__(self, item: int) -> Union[Events, CallableEventWithFilter]:
        return self._events[item]

    def __iter__(self) -> Iterator[Union[Events, CallableEventWithFilter]]:
        return iter(self._events)

    def __len__(self) -> int:
        return len(self._events)

    def __or__(self, other: Union[Events, CallableEventWithFilter]) -> "EventsList":
        self._append(event=other)
        return self


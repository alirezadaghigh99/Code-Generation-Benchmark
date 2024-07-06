def on(self, event_name: Any, *args: Any, **kwargs: Any) -> Callable:
        """Decorator shortcut for :meth:`~ignite.engine.engine.Engine.add_event_handler`.

        Args:
            event_name: An event to attach the handler to. Valid events are from :class:`~ignite.engine.events.Events`
                or any ``event_name`` added by :meth:`~ignite.engine.engine.Engine.register_events`.
            args: optional args to be passed to `handler`.
            kwargs: optional keyword args to be passed to `handler`.

        Examples:
            .. code-block:: python

                engine = Engine(process_function)

                @engine.on(Events.EPOCH_COMPLETED)
                def print_epoch():
                    print(f"Epoch: {engine.state.epoch}")

                @engine.on(Events.EPOCH_COMPLETED | Events.COMPLETED)
                def execute_something():
                    # do some thing not related to engine
                    pass
        """

        def decorator(f: Callable) -> Callable:
            self.add_event_handler(event_name, f, *args, **kwargs)
            return f

        return decorator

def terminate(self) -> None:
        """Sends terminate signal to the engine, so that it terminates completely the run. The run is
        terminated after the event on which ``terminate`` method was called. The following events are triggered:

        - ...
        - Terminating event
        - :attr:`~ignite.engine.events.Events.TERMINATE`
        - :attr:`~ignite.engine.events.Events.COMPLETED`


        Examples:
            .. testcode::

                from ignite.engine import Engine, Events

                def func(engine, batch):
                    print(engine.state.epoch, engine.state.iteration, " | ", batch)

                max_epochs = 4
                data = range(10)
                engine = Engine(func)

                @engine.on(Events.ITERATION_COMPLETED(once=14))
                def terminate():
                    print(f"-> terminate at iteration: {engine.state.iteration}")
                    engine.terminate()

                print("Start engine run:")
                state = engine.run(data, max_epochs=max_epochs)
                print("1 Engine run is terminated at ", state.epoch, state.iteration)
                state = engine.run(data, max_epochs=max_epochs)
                print("2 Engine ended the run at ", state.epoch, state.iteration)

            .. dropdown:: Output

                .. testoutput::

                    Start engine run:
                    1 1  |  0
                    1 2  |  1
                    1 3  |  2
                    1 4  |  3
                    1 5  |  4
                    1 6  |  5
                    1 7  |  6
                    1 8  |  7
                    1 9  |  8
                    1 10  |  9
                    2 11  |  0
                    2 12  |  1
                    2 13  |  2
                    2 14  |  3
                    -> terminate at iteration: 14
                    1 Engine run is terminated at  2 14
                    3 15  |  0
                    3 16  |  1
                    3 17  |  2
                    3 18  |  3
                    3 19  |  4
                    3 20  |  5
                    3 21  |  6
                    3 22  |  7
                    3 23  |  8
                    3 24  |  9
                    4 25  |  0
                    4 26  |  1
                    4 27  |  2
                    4 28  |  3
                    4 29  |  4
                    4 30  |  5
                    4 31  |  6
                    4 32  |  7
                    4 33  |  8
                    4 34  |  9
                    2 Engine ended the run at  4 34

        .. versionchanged:: 0.4.10
            Behaviour changed, for details see https://github.com/pytorch/ignite/issues/2669

        """
        self.logger.info("Terminate signaled. Engine will stop after current iteration is finished.")
        self.should_terminate = True

def on(self, event_name: Any, *args: Any, **kwargs: Any) -> Callable:
        """Decorator shortcut for :meth:`~ignite.engine.engine.Engine.add_event_handler`.

        Args:
            event_name: An event to attach the handler to. Valid events are from :class:`~ignite.engine.events.Events`
                or any ``event_name`` added by :meth:`~ignite.engine.engine.Engine.register_events`.
            args: optional args to be passed to `handler`.
            kwargs: optional keyword args to be passed to `handler`.

        Examples:
            .. code-block:: python

                engine = Engine(process_function)

                @engine.on(Events.EPOCH_COMPLETED)
                def print_epoch():
                    print(f"Epoch: {engine.state.epoch}")

                @engine.on(Events.EPOCH_COMPLETED | Events.COMPLETED)
                def execute_something():
                    # do some thing not related to engine
                    pass
        """

        def decorator(f: Callable) -> Callable:
            self.add_event_handler(event_name, f, *args, **kwargs)
            return f

        return decorator

def terminate_epoch(self) -> None:
        """Sends terminate signal to the engine, so that it terminates the current epoch. The run
        continues from the next epoch. The following events are triggered:

        - ...
        - Event on which ``terminate_epoch`` method is called
        - :attr:`~ignite.engine.events.Events.TERMINATE_SINGLE_EPOCH`
        - :attr:`~ignite.engine.events.Events.EPOCH_COMPLETED`
        - :attr:`~ignite.engine.events.Events.EPOCH_STARTED`
        - ...
        """
        self.logger.info(
            "Terminate current epoch is signaled. "
            "Current epoch iteration will stop after current iteration is finished."
        )
        self.should_terminate_single_epoch = True


def set_bypass_mode(cls, bypass: bool) -> None:
        """
        Set ``clearml.Task`` to offline mode.
        Will bypass all outside communication, and will save all data and logs to a local session folder.
        Should only be used in "standalone mode", when there is no access to the *clearml-server*.

        Args:
            bypass: If ``True``, all outside communication is skipped.
                Data and logs will be stored in a local session folder.
                For more information, please refer to `ClearML docs
                <https://clear.ml/docs/latest/docs/clearml_sdk/task_sdk/#offline-mode>`_.
        """
        from clearml import Task

        setattr(cls, "_bypass", bypass)
        Task.set_offline(offline_mode=bypass)


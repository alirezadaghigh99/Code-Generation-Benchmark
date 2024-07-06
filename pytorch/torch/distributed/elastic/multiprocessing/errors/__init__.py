def signal_name(self) -> str:
        if self.exitcode < 0:
            # We don't want to kill the parent process trying to find the signal name.
            # if the signal doesn't map to a known name, use not available.
            try:
                return signal.Signals(-self.exitcode).name
            except Exception:
                return _NOT_AVAILABLE
        else:
            return _NOT_AVAILABLE


    def run(self) -> None:
        signal.signal(signal.SIGINT, ignore_signal)
        signal.signal(signal.SIGTERM, self._handle_termination_signal)
        while not self._stop:
            command: Optional[Tuple[str, dict]] = self._command_queue.get()
            if command is None:
                break
            request_id, payload = command
            self._handle_command(request_id=request_id, payload=payload)
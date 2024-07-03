class IPCServer(socketserver.TCPServer):
    """Server that listens for inter-process messages.

    Messages are exchanged as pickle-encoded objects.
    """

    timeout = 2

    def __init__(self, on_message):
        """Constructs (but does not start) a server.

        Args:
            on_message (function): callback that takes a single argument (any
                Python object sent by a client) and returns an object in
                response
        """
        super().__init__(
            ("localhost", 0), IPCRequestHandler, bind_and_activate=True
        )
        self.on_message = on_message
        self.__lock = threading.Lock()
        self.__in_serve_forever = False

    def stop(self):
        """Stops and shuts down the server.

        Interrupts serve_forever() if necessary."""
        with self.__lock:
            if self.__in_serve_forever:
                self.shutdown()
        self.server_close()

    def serve_forever(self):
        """Handles one request at a time until the server shuts down.

        This is a simple wrapper around TCPServer.serve_forever that keeps
        track of whether it was called.
        """
        try:
            with self.__lock:
                self.__in_serve_forever = True
            super().serve_forever()
        finally:
            with self.__lock:
                self.__in_serve_forever = False

    @property
    def port(self):
        return self.server_address[1]

    @classmethod
    def run_in_background(cls, on_message):
        """Convenience wrapper that creates a new server instance and calls
        serve_forever in a background thread.

        The server will be listening before this function returns.

        Returns:
            the IPCServer instance
        """
        server = cls(on_message)
        thread = threading.Thread(target=server.serve_forever)
        thread.daemon = True
        thread.start()
        return server

    # backport for Python 3.5 - these were added in Python 3.6
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.server_close()
    def get_client(cls, client_type: str, port: int = 4000):
        try:
            return next(cl for cl in cls.__subclasses__() if cl.client_type == client_type)(port)
        except StopIteration:
            raise ValueError("Parameter server mode has to be either `http` or `socket`, "
                             "got {}".format(client_type))
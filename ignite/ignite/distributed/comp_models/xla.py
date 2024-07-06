def create_from_backend(backend: str = XLA_TPU, **kwargs: Any) -> "_XlaDistModel":
            if backend not in _XlaDistModel.available_backends:
                raise ValueError(f"Backend should be one of '{_XlaDistModel.available_backends}'")

            return _XlaDistModel(backend=backend, **kwargs)

def create_from_context() -> Optional["_XlaDistModel"]:
            return _XlaDistModel()

def create_from_backend(backend: str = XLA_TPU, **kwargs: Any) -> "_XlaDistModel":
            if backend not in _XlaDistModel.available_backends:
                raise ValueError(f"Backend should be one of '{_XlaDistModel.available_backends}'")

            return _XlaDistModel(backend=backend, **kwargs)

def create_from_context() -> Optional["_XlaDistModel"]:
            return _XlaDistModel()

def create_from_context() -> Optional["_XlaDistModel"]:
            return _XlaDistModel()

def create_from_backend(backend: str = XLA_TPU, **kwargs: Any) -> "_XlaDistModel":
            if backend not in _XlaDistModel.available_backends:
                raise ValueError(f"Backend should be one of '{_XlaDistModel.available_backends}'")

            return _XlaDistModel(backend=backend, **kwargs)

